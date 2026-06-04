// On-device streaming speech recognition via macOS 26's SpeechAnalyzer / SpeechTranscriber, exposed to
// Rust over a small C ABI. The transcriber runs Apple's built-in models (no download of *our* models;
// the OS fetches the per-language asset on first use), streams volatile (partial) + final results, and
// we forward each as (text, isFinal) to a Rust callback.
//
// Bridging notes: @_cdecl entry points are synchronous, but the Speech API is async. `start` returns a
// handle immediately and kicks the (async) setup — model install, analyzer start, result consumption —
// onto a Task. `feed` yields audio into an AsyncStream the analyzer consumes; audio that arrives before
// setup finishes is buffered briefly and flushed once the analyzer's audio format is known.

import AVFoundation
import Foundation
import Speech

/// Whether on-device transcription is usable on this OS (macOS 26+ with the SpeechTranscriber API).
@_cdecl("wisp_applespeech_available")
public func wisp_applespeech_available() -> Bool {
    if #available(macOS 26.0, *) { return true }
    return false
}

@available(macOS 26.0, *)
final class WispSpeechSession {
    private let callback: @convention(c) (UnsafeMutableRawPointer?, UnsafePointer<CChar>?, Bool) -> Void
    private let ctx: UnsafeMutableRawPointer?

    private let transcriber: SpeechTranscriber
    private let analyzer: SpeechAnalyzer
    private let continuation: AsyncStream<AnalyzerInput>.Continuation
    private let inputStream: AsyncStream<AnalyzerInput>

    /// The audio format the analyzer wants; nil until setup resolves it. Audio fed before then is held.
    private var analyzerFormat: AVAudioFormat?
    private var converter: AVAudioConverter?
    private var pending: [(samples: [Float], rate: Double)] = []
    private let lock = NSLock()

    init(locale: String,
         ctx: UnsafeMutableRawPointer?,
         callback: @escaping @convention(c) (UnsafeMutableRawPointer?, UnsafePointer<CChar>?, Bool) -> Void) {
        self.ctx = ctx
        self.callback = callback
        // `.volatileResults` streams partial hypotheses as you speak; `.fastResults` biases the
        // transcriber toward low-latency emission over maximal accuracy — the right trade for a live
        // caption (the accuracy-first models are the other picker options).
        self.transcriber = SpeechTranscriber(
            locale: Locale(identifier: locale),
            transcriptionOptions: [],
            reportingOptions: [.volatileResults, .fastResults],
            attributeOptions: [])
        // User-initiated priority keeps the analyzer responsive under load; `.lingering` keeps the model
        // warm between sessions so a restart doesn't pay the cold-load latency again.
        self.analyzer = SpeechAnalyzer(
            modules: [transcriber],
            options: SpeechAnalyzer.Options(priority: .userInitiated, modelRetention: .lingering))
        (self.inputStream, self.continuation) = AsyncStream<AnalyzerInput>.makeStream()
    }

    func begin() {
        Task { [weak self] in
            guard let self else { return }
            do {
                if let request = try await AssetInventory.assetInstallationRequest(supporting: [self.transcriber]) {
                    try await request.downloadAndInstall()
                }
                let format = await SpeechAnalyzer.bestAvailableAudioFormat(compatibleWith: [self.transcriber])
                self.setFormat(format)
                try await self.analyzer.start(inputSequence: self.inputStream)
                for try await result in self.transcriber.results {
                    let text = String(result.text.characters)
                    text.withCString { cstr in self.callback(self.ctx, cstr, result.isFinal) }
                }
            } catch {
                "".withCString { cstr in self.callback(self.ctx, cstr, true) }
            }
        }
    }

    private func setFormat(_ format: AVAudioFormat?) {
        lock.lock()
        analyzerFormat = format
        let held = pending
        pending.removeAll()
        lock.unlock()
        for chunk in held { yield(chunk.samples, rate: chunk.rate) }
    }

    func feed(_ samples: [Float], rate: Double) {
        lock.lock()
        let ready = analyzerFormat != nil
        if !ready {
            // Cap the pre-setup hold to ~2 s so a slow first-run model install can't grow it unbounded.
            if pending.reduce(0, { $0 + $1.samples.count }) < Int(rate * 2) {
                pending.append((samples, rate))
            }
            lock.unlock()
            return
        }
        lock.unlock()
        yield(samples, rate: rate)
    }

    private func yield(_ samples: [Float], rate: Double) {
        guard let target = analyzerFormat else { return }
        guard let source = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: rate, channels: 1, interleaved: false) else { return }
        guard let inBuf = makeBuffer(samples, format: source) else { return }

        let out: AVAudioPCMBuffer
        if source.sampleRate == target.sampleRate && source.commonFormat == target.commonFormat {
            out = inBuf
        } else {
            if converter == nil || converter?.outputFormat != target { converter = AVAudioConverter(from: source, to: target) }
            guard let converter else { return }
            let ratio = target.sampleRate / source.sampleRate
            let cap = AVAudioFrameCount(Double(inBuf.frameLength) * ratio) + 16
            guard let outBuf = AVAudioPCMBuffer(pcmFormat: target, frameCapacity: cap) else { return }
            var supplied = false
            var err: NSError?
            converter.convert(to: outBuf, error: &err) { _, status in
                if supplied { status.pointee = .noDataNow; return nil }
                supplied = true
                status.pointee = .haveData
                return inBuf
            }
            if err != nil { return }
            out = outBuf
        }
        continuation.yield(AnalyzerInput(buffer: out))
    }

    private func makeBuffer(_ samples: [Float], format: AVAudioFormat) -> AVAudioPCMBuffer? {
        guard let buf = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(samples.count)) else { return nil }
        buf.frameLength = AVAudioFrameCount(samples.count)
        if let ch = buf.floatChannelData { samples.withUnsafeBufferPointer { ch[0].update(from: $0.baseAddress!, count: samples.count) } }
        return buf
    }

    func stop() {
        continuation.finish()
        Task { [analyzer] in try? await analyzer.finalizeAndFinishThroughEndOfInput() }
    }
}

/// Starts a session for `locale` (BCP-47, e.g. "en-US"/"zh-CN"/"yue-Hant"); returns an opaque handle or
/// null on an unsupported OS. `callback(ctx, text, isFinal)` fires for each partial/final result.
@_cdecl("wisp_applespeech_start")
public func wisp_applespeech_start(
    _ locale: UnsafePointer<CChar>?,
    _ ctx: UnsafeMutableRawPointer?,
    _ callback: @escaping @convention(c) (UnsafeMutableRawPointer?, UnsafePointer<CChar>?, Bool) -> Void
) -> UnsafeMutableRawPointer? {
    guard #available(macOS 26.0, *) else { return nil }
    let requested = locale.map { String(cString: $0) } ?? "en-US"
    // "auto" means "follow the system language" — resolve it to the host's current locale.
    let id = requested == "auto" ? Locale.current.identifier : requested
    let session = WispSpeechSession(locale: id, ctx: ctx, callback: callback)
    session.begin()
    return Unmanaged.passRetained(session).toOpaque()
}

/// Feeds `count` mono Float32 samples at `rate` Hz into the session.
@_cdecl("wisp_applespeech_feed")
public func wisp_applespeech_feed(_ handle: UnsafeMutableRawPointer?, _ samples: UnsafePointer<Float>?, _ count: Int, _ rate: Double) {
    guard #available(macOS 26.0, *), let handle, let samples else { return }
    let session = Unmanaged<WispSpeechSession>.fromOpaque(handle).takeUnretainedValue()
    session.feed(Array(UnsafeBufferPointer(start: samples, count: count)), rate: rate)
}

/// Stops the session and releases the handle.
@_cdecl("wisp_applespeech_stop")
public func wisp_applespeech_stop(_ handle: UnsafeMutableRawPointer?) {
    guard #available(macOS 26.0, *), let handle else { return }
    let session = Unmanaged<WispSpeechSession>.fromOpaque(handle).takeRetainedValue()
    // The Rust caller invokes this while joining the capture thread that owns the engine. Finishing
    // the analyzer and releasing it (the ARC deinit on return) can block, which would stall that join
    // and make "Stop" hang. Hand the session to a background queue: stop() and the deinit run there,
    // and this @_cdecl call returns immediately so the capture thread joins at once.
    DispatchQueue.global(qos: .utility).async {
        session.stop()
    }
}
