//! Background session lifecycle around a [`Pipeline`].

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{self, Sender};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use wisp_audio::{Resampler, TARGET_SAMPLE_RATE};
use wisp_core::audio::AudioSource;
use wisp_core::denoise::Denoiser;
use wisp_core::engine::StreamingAsrEngine;
use wisp_core::error::{Result, WispError};
use wisp_core::transcript::{AudioSourceKind, SegmentStatus, TranscriptEvent, TranscriptSegment};

use crate::pipeline::Pipeline;
use crate::segmenter::{Segmenter, Utterance};
use crate::transcriber::Transcriber;

/// A boxed, `Send` consumer of transcript events (e.g. one that forwards them to the UI).
pub type EventSink = Box<dyn FnMut(TranscriptEvent) + Send>;

/// A unit of work handed from a (real-time) capture thread to the (slow) transcription thread. Each
/// job carries the [`AudioSourceKind`] of the stream that produced it, so the shared transcriber
/// labels the resulting segment by source even though mic and system audio share one engine.
#[derive(Debug)]
enum Job {
    /// A provisional decode of the still-open utterance, sharing the final's id so the UI updates
    /// one row in place. Dropped if a newer partial or the final supersedes it before it runs.
    Partial(u64, AudioSourceKind, Utterance),
    /// The closed, authoritative utterance.
    Final(u64, AudioSourceKind, Utterance),
}

/// One live audio stream feeding a shared-engine session: its capture source, the segmenter that
/// turns its frames into utterances, an optional per-stream denoiser, and the [`AudioSourceKind`]
/// every segment cut from it is tagged with. [`Session::spawn_live_multi`] runs one capture thread
/// per stream against a single shared [`Transcriber`].
pub struct LiveStream {
    pub segmenter: Box<dyn Segmenter>,
    pub source: Box<dyn AudioSource>,
    pub denoiser: Option<Box<dyn Denoiser>>,
    pub kind: AudioSourceKind,
}

/// Bounds on how much captured audio to accumulate between provisional partial decodes. The actual
/// interval *adapts to the engine's measured decode time*: a fast Mac gets snappier partials near
/// the floor, a slow one backs off toward the ceiling so it never builds a backlog (stale partials
/// are dropped regardless). [`PARTIAL_DEFAULT`] is used until the first decode is timed.
const PARTIAL_FLOOR: Duration = Duration::from_millis(300);
const PARTIAL_CEIL: Duration = Duration::from_millis(1_500);
const PARTIAL_DEFAULT: Duration = Duration::from_millis(500);

/// An utterance paired with the id its segment will carry and the kind it was captured from — the
/// unit a transcribe pass consumes.
type Decode = (u64, AudioSourceKind, Utterance);

/// A transcription run executing on background thread(s).
///
/// The application's start/stop commands are thin wrappers over this — `start` →
/// [`Session::spawn_live`], `stop` → [`Session::stop`] — keeping the lifecycle logic off the UI
/// layer.
pub struct Session {
    stop: Arc<AtomicBool>,
    handles: Vec<JoinHandle<Result<()>>>,
}

impl Session {
    /// Spawns a synchronous [`Pipeline`] on one background thread, driving `source` and forwarding
    /// events to `sink` until the source ends or [`stop`](Self::stop) is called.
    ///
    /// Segmentation and (slow) transcription share this thread, so it's meant for finite sources
    /// (e.g. a file) where transcription latency doesn't stall live capture. Live microphone /
    /// system audio should use [`spawn_live`](Self::spawn_live).
    pub fn spawn(
        mut pipeline: Pipeline,
        mut source: Box<dyn AudioSource>,
        mut sink: EventSink,
    ) -> Self {
        let stop = Arc::new(AtomicBool::new(false));
        let stop_for_thread = Arc::clone(&stop);

        let handle =
            thread::spawn(move || pipeline.run_until(&mut *source, &mut *sink, &stop_for_thread));

        Self {
            stop,
            handles: vec![handle],
        }
    }

    /// Spawns a *decoupled* live session over one audio stream — the common single-stream case (just
    /// the mic, or just system audio), tagged with the transcriber's
    /// [`source_kind`](Transcriber::source_kind). A thin wrapper over [`spawn_live_multi`].
    pub fn spawn_live(
        segmenter: Box<dyn Segmenter>,
        transcriber: Transcriber,
        source: Box<dyn AudioSource>,
        sink: EventSink,
        denoiser: Option<Box<dyn Denoiser>>,
    ) -> Self {
        let kind = transcriber.source_kind();
        Self::spawn_live_multi(
            vec![LiveStream {
                segmenter,
                source,
                denoiser,
                kind,
            }],
            transcriber,
            sink,
        )
    }

    /// Spawns a *decoupled* live session over N audio streams sharing **one** ASR engine.
    ///
    /// Each stream runs its own capture+segmentation thread at real-time — converting frames to 16 kHz
    /// mono, segmenting (and optionally denoising) them — and hands each finished [`Utterance`] over a
    /// shared queue to a *single* transcription thread, so a slow engine can never stall capture or
    /// drop audio. Running mic + system audio through one model halves the memory of two engines and
    /// gives one coherent rolling-context and one consistent language; each job is stamped with its
    /// stream's [`AudioSourceKind`] so the segment is still labelled by source. Ids come from one
    /// shared counter, so a partial and its final share an id across all streams and the UI updates a
    /// single row in place. On stop each capture thread flushes its buffered utterance and drops its
    /// sender; the transcription thread finishes the backlog once every capture thread has.
    pub fn spawn_live_multi(
        streams: Vec<LiveStream>,
        mut transcriber: Transcriber,
        mut sink: EventSink,
    ) -> Self {
        let stop = Arc::new(AtomicBool::new(false));
        let (job_tx, job_rx) = mpsc::channel::<Job>();

        // Most-recent partial-decode time (ms), shared so the capture threads can pace partials to
        // what this machine actually decodes — snappier on a fast Mac, backing off on a slow one.
        let decode_ms = Arc::new(AtomicU64::new(PARTIAL_DEFAULT.as_millis() as u64));
        // One id space across all streams, so ids stay globally unique on the shared sink.
        let next_id = Arc::new(AtomicU64::new(0));

        // Transcription thread: drains jobs and forwards segments, tagging each with the kind its
        // capture thread stamped on the job. Ends when every capture thread has dropped its sender. A
        // transcription error on one job is logged and skipped rather than killing the session. Each
        // burst is coalesced first — per id, so two streams' open utterances never evict each other.
        let stop_for_transcribe = Arc::clone(&stop);
        let decode_ms_for_transcribe = Arc::clone(&decode_ms);
        let transcribe_handle = thread::spawn(move || {
            while let Ok(first) = job_rx.recv() {
                let mut batch = vec![first];
                while let Ok(next) = job_rx.try_recv() {
                    batch.push(next);
                }
                let (finals, partials) = coalesce(batch);

                // Finals are the authoritative transcript — always transcribe them, even on stop. The
                // capture threads flush their trailing utterance as a Final on stop; dropping it (and
                // any backlog of real finals a slow engine fell behind on) would silently lose the
                // last sentence(s) of the recording.
                for (id, kind, utterance) in finals {
                    match transcriber.transcribe_utterance(
                        id,
                        kind,
                        &utterance,
                        SegmentStatus::Final,
                    ) {
                        Ok(Some(segment)) => sink(TranscriptEvent::Segment(segment)),
                        Ok(None) => {}
                        Err(e) => eprintln!("wisp: transcription error: {e}"),
                    }
                }

                // On stop, skip provisional partials — a final supersedes them and decoding would only
                // slow the shutdown — and stop pulling new work: drain the queued finals above, then
                // exit when the capture threads drop their senders.
                if stop_for_transcribe.load(Ordering::Relaxed) {
                    continue;
                }

                for (id, kind, utterance) in partials {
                    let started = Instant::now();
                    match transcriber.transcribe_utterance(
                        id,
                        kind,
                        &utterance,
                        SegmentStatus::Partial,
                    ) {
                        Ok(Some(segment)) => sink(TranscriptEvent::Segment(segment)),
                        Ok(None) => {}
                        Err(e) => eprintln!("wisp: partial transcription error: {e}"),
                    }
                    // Feed the measured decode cost back to the capture threads' pacing.
                    decode_ms_for_transcribe
                        .store(started.elapsed().as_millis() as u64, Ordering::Relaxed);
                }
            }
            Ok(())
        });

        // One capture + segmentation thread per stream; none ever blocks on the engine. Each clones
        // the job sender, the pacing gauge, the stop flag, and the shared id counter.
        let mut handles = vec![transcribe_handle];
        for stream in streams {
            let job_tx = job_tx.clone();
            let decode_ms = Arc::clone(&decode_ms);
            let stop = Arc::clone(&stop);
            let next_id = Arc::clone(&next_id);
            handles.push(thread::spawn(move || {
                run_capture(stream, &job_tx, &decode_ms, &stop, &next_id)
            }));
        }
        // Drop the original sender so the transcription thread ends once every capture thread's clone
        // has dropped (all sources exhausted or stopped).
        drop(job_tx);

        Self { stop, handles }
    }

    /// Spawns a *streaming* live session: a single thread feeds captured audio chunk-by-chunk to an
    /// online `engine` that emits a growing hypothesis and finalizes on its own endpoint detection.
    ///
    /// Unlike [`spawn_live`](Self::spawn_live) — which VAD-segments utterances and transcribes each
    /// whole on a decoupled thread — a streaming engine decodes incrementally and cheaply per chunk,
    /// so capture and recognition share one thread without stalling. Each non-empty hypothesis is
    /// forwarded as a `Partial` segment the UI updates in place by id; on an endpoint it's sent
    /// `Final` and the next utterance gets a fresh id. `denoiser`, when present, cleans each chunk
    /// first. `kind` tags every segment with which source it came from.
    pub fn spawn_streaming(
        mut engine: Box<dyn StreamingAsrEngine>,
        mut source: Box<dyn AudioSource>,
        mut sink: EventSink,
        denoiser: Option<Box<dyn Denoiser>>,
        kind: AudioSourceKind,
    ) -> Self {
        let stop = Arc::new(AtomicBool::new(false));
        let stop_for_thread = Arc::clone(&stop);

        let handle = thread::spawn(move || {
            run_streaming(
                &mut *engine,
                &mut *source,
                denoiser,
                &mut sink,
                kind,
                &stop_for_thread,
            )
        });

        Self {
            stop,
            handles: vec![handle],
        }
    }

    /// Signals the session to stop *without* waiting — its thread(s) observe the flag on their next
    /// frame and wind down (flushing the trailing utterance). Call this up front, before a bounded
    /// teardown, so a session stops capturing/emitting even if a later join wedges on a stuck device.
    pub fn signal_stop(&self) {
        self.stop.store(true, Ordering::Relaxed);
    }

    /// Signals the session to stop and waits for its thread(s) to finish, returning the first
    /// error encountered. Use this for live sources (e.g. a microphone) that never end on their
    /// own.
    pub fn stop(mut self) -> Result<()> {
        self.stop.store(true, Ordering::Relaxed);
        self.take_join()
    }

    /// Waits for the session to finish on its own — e.g. a finite/file source reaching its end —
    /// without signalling stop, returning the first error encountered.
    pub fn join(mut self) -> Result<()> {
        self.take_join()
    }

    fn take_join(&mut self) -> Result<()> {
        let mut first_error = None;
        for handle in std::mem::take(&mut self.handles) {
            let result = handle
                .join()
                .unwrap_or_else(|_| Err(WispError::Engine("transcription thread panicked".into())));
            if let Err(e) = result {
                first_error.get_or_insert(e);
            }
        }
        match first_error {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }
}

/// Capture loop for one [`LiveStream`] in [`Session::spawn_live_multi`]: pull frames, segment them,
/// and queue closed utterances as `Final` jobs until `stop` is set or the source ends; then flush any
/// buffered utterance. Every job is tagged with the stream's [`AudioSourceKind`] so the shared
/// transcriber labels each segment by source.
///
/// Between finals, on an interval that adapts to the engine's measured decode speed (bounded by
/// [`PARTIAL_FLOOR`]/[`PARTIAL_CEIL`] via `decode_ms`), it also queues a `Partial` job of the
/// still-open utterance (when the segmenter exposes one), so the UI can show provisional text before
/// the speaker pauses. Ids come from the session's shared `next_id`, so a `Partial` and the `Final`
/// that closes the same utterance share one id — unique across sibling streams — and the UI updates a
/// single row in place.
fn run_capture(
    stream: LiveStream,
    job_tx: &Sender<Job>,
    decode_ms: &AtomicU64,
    stop: &AtomicBool,
    next_id: &AtomicU64,
) -> Result<()> {
    let LiveStream {
        mut segmenter,
        mut source,
        mut denoiser,
        kind,
    } = stream;
    let mut open_id: Option<u64> = None;
    let mut since_partial = Duration::ZERO;
    let mut resampler = Resampler::new(TARGET_SAMPLE_RATE);
    // A partial re-decodes the whole open utterance every cadence; cap it to a trailing window so the
    // cost stays bounded on a long monologue (the closing Final still decodes the full utterance).
    let max_partial_samples = partial_window_samples();

    while !stop.load(Ordering::Relaxed) {
        match source.next_frame()? {
            Some(frame) => {
                let mono = resampler.process(&frame);
                let mono = match &mut denoiser {
                    Some(d) => d.denoise(&mono, TARGET_SAMPLE_RATE),
                    None => mono,
                };

                for utterance in segmenter.push(&mono, frame.timestamp, frame.duration()) {
                    let id = open_id.take().unwrap_or_else(|| alloc_id(next_id));
                    let _ = job_tx.send(Job::Final(id, kind, utterance));
                    since_partial = Duration::ZERO;
                }

                since_partial += frame.duration();
                if since_partial >= partial_target(decode_ms) {
                    since_partial = Duration::ZERO;
                    if let Some(utterance) = segmenter.partial() {
                        let id = *open_id.get_or_insert_with(|| alloc_id(next_id));
                        let utterance =
                            cap_partial(utterance, max_partial_samples, TARGET_SAMPLE_RATE);
                        let _ = job_tx.send(Job::Partial(id, kind, utterance));
                    }
                }
            }
            None => break,
        }
    }
    for utterance in segmenter.flush() {
        let id = open_id.take().unwrap_or_else(|| alloc_id(next_id));
        let _ = job_tx.send(Job::Final(id, kind, utterance));
    }
    Ok(())
}

/// Streaming capture+recognition loop for [`Session::spawn_streaming`]: pull frames, resample each to
/// the engine's own input rate as mono (optionally denoised), feed it to `engine`, and forward the
/// growing hypothesis as `Partial`/`Final` segments until `stop` is set or the source ends.
///
/// All partials of one utterance share an id with the `Final` that closes it, so the UI updates a
/// single row in place; the engine's own endpoint detection commits the utterance and advances to a
/// fresh id and start time.
fn run_streaming(
    engine: &mut dyn StreamingAsrEngine,
    source: &mut dyn AudioSource,
    mut denoiser: Option<Box<dyn Denoiser>>,
    sink: &mut EventSink,
    kind: AudioSourceKind,
    stop: &AtomicBool,
) -> Result<()> {
    // Most engines want 16 kHz; a cloud engine that accepts richer audio asks for more (e.g. 24 kHz),
    // so we resample each frame to the engine's own rate rather than always downsampling to 16 kHz.
    let rate = engine.input_sample_rate();
    let mut id: u64 = 0;
    let mut utterance_start: Option<Duration> = None;
    let mut resampler = Resampler::new(rate);
    // The open utterance's latest partial, kept so stopping (or the source ending) mid-utterance
    // promotes it to a `Final` instead of dropping the last words — a streaming engine commits text
    // only on its own endpoint, which a stop pre-empts.
    let mut pending_final: Option<TranscriptSegment> = None;

    while !stop.load(Ordering::Relaxed) {
        let Some(frame) = source.next_frame()? else {
            break;
        };
        let start = *utterance_start.get_or_insert(frame.timestamp);

        let mono = resampler.process(&frame);
        let mono = match &mut denoiser {
            Some(d) => d.denoise(&mono, rate),
            None => mono,
        };

        let result = engine.accept_waveform(rate, &mono);

        // A dual-stream engine (cloud model session) may carry a parallel `aux` rendering (a
        // translation) that leads or lags the verbatim text, so emit when either side has content.
        let aux = result.aux.filter(|a| !a.is_empty());
        let emitted = !result.text.is_empty() || aux.is_some();
        if emitted {
            let mut segment = TranscriptSegment::new(
                id,
                &result.text,
                start..(frame.timestamp + frame.duration()),
                kind,
            )
            .with_aux_text(aux);
            segment.status = if result.is_endpoint {
                SegmentStatus::Final
            } else {
                SegmentStatus::Partial
            };
            // Hold the open partial so the post-loop flush can finalize it; a final closes it now.
            pending_final = (!result.is_endpoint).then(|| segment.clone());
            sink(TranscriptEvent::Segment(segment));
        }
        if result.is_endpoint {
            // A committed utterance (one that produced text) advances to a fresh id; an empty
            // endpoint just resets the start so the next utterance reuses this id.
            if emitted {
                id += 1;
            }
            utterance_start = None;
            pending_final = None;
        }
    }

    // The loop exited mid-utterance (stop, or the source ended): promote the still-open partial to a
    // `Final` so the user's last sentence survives even though the engine never endpointed it.
    if let Some(mut segment) = pending_final {
        segment.status = SegmentStatus::Final;
        sink(TranscriptEvent::Segment(segment));
    }
    Ok(())
}

/// Trailing-window cap (seconds) for a live *partial* re-decode: bounds the per-partial cost on a long
/// utterance instead of re-decoding the whole thing every cadence. Override with the env var below.
const PARTIAL_WINDOW_SECS_ENV: &str = "WISP_PARTIAL_WINDOW_SECS";
const DEFAULT_PARTIAL_WINDOW_SECS: f64 = 8.0;

/// The partial trailing-window length in 16 kHz samples — read once per session from the environment.
fn partial_window_samples() -> usize {
    let secs = std::env::var(PARTIAL_WINDOW_SECS_ENV)
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .filter(|&s| s > 0.0)
        .unwrap_or(DEFAULT_PARTIAL_WINDOW_SECS);
    (secs * f64::from(TARGET_SAMPLE_RATE)) as usize
}

/// Caps a live partial's audio to its trailing `max_samples`, shifting `start` forward by the dropped
/// span so the shorter re-decode stays timestamp-consistent. Left unchanged when it already fits; the
/// closing `Final` always decodes the whole utterance, so nothing is lost.
fn cap_partial(mut utterance: Utterance, max_samples: usize, rate: u32) -> Utterance {
    if utterance.audio.len() > max_samples {
        let dropped = utterance.audio.len() - max_samples;
        utterance.audio.drain(..dropped);
        utterance.start += Duration::from_secs_f64(dropped as f64 / f64::from(rate));
    }
    utterance
}

/// The current partial cadence: the engine's last measured partial-decode time, clamped to the sane
/// [`PARTIAL_FLOOR`]..=[`PARTIAL_CEIL`] bounds — so fast Macs emit partials more often and slow ones
/// back off, but neither runs away.
fn partial_target(decode_ms: &AtomicU64) -> Duration {
    let ms = decode_ms.load(Ordering::Relaxed).clamp(
        PARTIAL_FLOOR.as_millis() as u64,
        PARTIAL_CEIL.as_millis() as u64,
    );
    Duration::from_millis(ms)
}

/// Returns the next id and advances the shared counter. Sharing one counter across a session's
/// capture threads keeps ids globally unique, so a partial and its final still meet on one id while
/// sibling streams never collide.
fn alloc_id(next_id: &AtomicU64) -> u64 {
    next_id.fetch_add(1, Ordering::Relaxed)
}

/// Collapses a drained burst of jobs into the finals to transcribe (in order) and the surviving
/// partials — at most one per id, the newest not yet closed by a final. Sibling streams share the
/// queue but hold globally-unique ids, so keying by id keeps each open utterance's latest
/// provisional view alive while a partial that its own final closes, or that a newer partial for the
/// same id replaces, is stale and dropped.
fn coalesce(batch: Vec<Job>) -> (Vec<Decode>, Vec<Decode>) {
    let mut finals = Vec::new();
    let mut partials: Vec<Decode> = Vec::new();
    for job in batch {
        match job {
            Job::Final(id, kind, utterance) => {
                partials.retain(|(pid, _, _)| *pid != id);
                finals.push((id, kind, utterance));
            }
            Job::Partial(id, kind, utterance) => {
                match partials.iter_mut().find(|(pid, _, _)| *pid == id) {
                    Some(slot) => *slot = (id, kind, utterance),
                    None => partials.push((id, kind, utterance)),
                }
            }
        }
    }
    (finals, partials)
}

impl Drop for Session {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        let _ = self.take_join();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::segmenter::EnergySegmenter;
    use crate::vad::EnergyVad;
    use std::sync::mpsc;
    use std::time::Duration;
    use wisp_core::audio::{AudioFrame, AudioSourceInfo};
    use wisp_core::engine::TranscriptionResult;
    use wisp_core::testing::{MockAsrEngine, MockAudioSource};
    use wisp_core::transcript::{AudioSourceKind, TranscriptSegment};

    fn frame(amp: f32, t_ms: u64) -> AudioFrame {
        AudioFrame::new(vec![amp; 1_600], 16_000, 1, Duration::from_millis(t_ms))
    }

    fn canned(text: &str) -> TranscriptionResult {
        TranscriptionResult {
            segments: vec![TranscriptSegment::new(
                0,
                text,
                Duration::ZERO..Duration::from_millis(50),
                AudioSourceKind::File,
            )],
        }
    }

    fn pipeline(results: Vec<TranscriptionResult>) -> Pipeline {
        Pipeline::new(
            Box::new(MockAsrEngine::new(results)),
            Box::new(EnergyVad::new(0.01)),
            AudioSourceKind::Microphone,
            Duration::from_millis(150),
        )
    }

    #[test]
    fn finite_source_delivers_events_then_finishes() {
        let frames = vec![
            frame(0.5, 0),
            frame(0.5, 100),
            frame(0.0, 200),
            frame(0.0, 300),
        ];
        let source: Box<dyn AudioSource> = Box::new(MockAudioSource::new(frames));

        let (tx, rx) = mpsc::channel();
        let sink: EventSink = Box::new(move |event| {
            let _ = tx.send(event);
        });

        let session = Session::spawn(pipeline(vec![canned("hi")]), source, sink);
        session.join().unwrap();

        let events: Vec<_> = rx.try_iter().collect();
        assert_eq!(events.len(), 1);
        match &events[0] {
            TranscriptEvent::Segment(s) => assert_eq!(s.text, "hi"),
            _ => panic!("unexpected event"),
        }
    }

    #[test]
    fn stop_halts_an_endless_source() {
        struct Endless;
        impl AudioSource for Endless {
            fn info(&self) -> AudioSourceInfo {
                AudioSourceInfo {
                    kind: AudioSourceKind::Microphone,
                    name: "endless".to_owned(),
                }
            }
            fn next_frame(&mut self) -> Result<Option<AudioFrame>> {
                Ok(Some(AudioFrame::new(
                    vec![0.0; 160],
                    16_000,
                    1,
                    Duration::ZERO,
                )))
            }
        }

        let (tx, _rx) = mpsc::channel();
        let sink: EventSink = Box::new(move |event| {
            let _ = tx.send(event);
        });

        // Endless yields silence forever; stop() must break the loop and join cleanly.
        let session = Session::spawn(pipeline(vec![]), Box::new(Endless), sink);
        session.stop().unwrap();
    }

    #[test]
    fn spawn_live_stop_keeps_the_flushed_trailing_final() {
        // The user stops mid/just-after a sentence: the capture thread flushes that buffered utterance
        // as a Final on stop. The transcribe loop must still decode it — dropping it would silently lose
        // the last sentence of every recording. (This segmenter never finalizes mid-stream and exposes
        // no partial, so the *only* job is the flushed Final, isolating the stop path.)
        struct BufferingSegmenter;
        impl Segmenter for BufferingSegmenter {
            fn push(&mut self, _m: &[f32], _t: Duration, _d: Duration) -> Vec<Utterance> {
                Vec::new()
            }
            fn flush(&mut self) -> Vec<Utterance> {
                vec![Utterance {
                    audio: vec![0.3; 1_600],
                    start: Duration::ZERO,
                }]
            }
            fn partial(&self) -> Option<Utterance> {
                None
            }
        }

        struct Endless;
        impl AudioSource for Endless {
            fn info(&self) -> AudioSourceInfo {
                AudioSourceInfo {
                    kind: AudioSourceKind::Microphone,
                    name: "endless".to_owned(),
                }
            }
            fn next_frame(&mut self) -> Result<Option<AudioFrame>> {
                Ok(Some(AudioFrame::new(
                    vec![0.0; 160],
                    16_000,
                    1,
                    Duration::ZERO,
                )))
            }
        }

        let (tx, rx) = mpsc::channel();
        let sink: EventSink = Box::new(move |event| {
            let _ = tx.send(event);
        });

        let session = Session::spawn_live(
            Box::new(BufferingSegmenter),
            transcriber(vec![canned("the trailing sentence")]),
            Box::new(Endless),
            sink,
            None,
        );
        session.stop().unwrap();

        let texts: Vec<String> = rx
            .try_iter()
            .filter_map(|event| match event {
                TranscriptEvent::Segment(s) => Some(s.text),
                _ => None,
            })
            .collect();
        assert_eq!(
            texts,
            vec!["the trailing sentence".to_owned()],
            "the utterance flushed on stop must still be transcribed, not dropped"
        );
    }

    fn segmenter() -> Box<dyn Segmenter> {
        Box::new(EnergySegmenter::new(
            Box::new(EnergyVad::new(0.01)),
            Duration::from_millis(150),
        ))
    }

    fn transcriber(results: Vec<TranscriptionResult>) -> Transcriber {
        Transcriber::new(
            Box::new(MockAsrEngine::new(results)),
            AudioSourceKind::Microphone,
        )
    }

    #[test]
    fn spawn_live_transcribes_all_queued_utterances_in_order() {
        // Two utterances separated by trailing silence; the decoupled path must transcribe both,
        // in order, with sequential ids — even though the engine runs on a separate thread.
        let frames = vec![
            frame(0.5, 0),
            frame(0.5, 100),
            frame(0.0, 200),
            frame(0.0, 300), // closes utterance 1 (start 0)
            frame(0.5, 400),
            frame(0.0, 500),
            frame(0.0, 600), // closes utterance 2 (start 400)
        ];
        let source: Box<dyn AudioSource> = Box::new(MockAudioSource::new(frames));

        let (tx, rx) = mpsc::channel();
        let sink: EventSink = Box::new(move |event| {
            let _ = tx.send(event);
        });

        let session = Session::spawn_live(
            segmenter(),
            transcriber(vec![canned("one"), canned("two")]),
            source,
            sink,
            None,
        );
        session.join().unwrap();

        let segments: Vec<(u64, String)> = rx
            .try_iter()
            .map(|event| match event {
                TranscriptEvent::Segment(s) => (s.id, s.text),
                _ => panic!("unexpected event"),
            })
            .collect();
        assert_eq!(segments, vec![(0, "one".to_owned()), (1, "two".to_owned())]);
    }

    #[test]
    fn spawn_live_multi_tags_each_stream_with_its_own_kind_and_unique_ids() {
        // Two streams — mic and system — share one engine and one sink. Each yields a single final;
        // spawn_live_multi must tag every segment with its own stream's kind (overriding the shared
        // transcriber's default) and hand out globally-unique ids from the shared counter, even
        // though both drain through one transcribe thread in non-deterministic order.
        let one_utterance = || {
            vec![
                frame(0.5, 0),
                frame(0.5, 100),
                frame(0.0, 200),
                frame(0.0, 300),
            ]
        };
        let mic = LiveStream {
            segmenter: segmenter(),
            source: Box::new(MockAudioSource::new(one_utterance())),
            denoiser: None,
            kind: AudioSourceKind::Microphone,
        };
        let system = LiveStream {
            segmenter: segmenter(),
            source: Box::new(MockAudioSource::new(one_utterance())),
            denoiser: None,
            kind: AudioSourceKind::System,
        };

        let (tx, rx) = mpsc::channel();
        let sink: EventSink = Box::new(move |event| {
            let _ = tx.send(event);
        });

        let session = Session::spawn_live_multi(
            vec![mic, system],
            transcriber(vec![canned("one"), canned("two")]),
            sink,
        );
        session.join().unwrap();

        let mut segments: Vec<(u64, AudioSourceKind)> = rx
            .try_iter()
            .map(|event| match event {
                TranscriptEvent::Segment(s) => (s.id, s.source),
                _ => panic!("unexpected event"),
            })
            .collect();
        segments.sort_by_key(|(id, _)| *id);

        let ids: Vec<u64> = segments.iter().map(|(id, _)| *id).collect();
        assert_eq!(
            ids,
            vec![0, 1],
            "ids stay globally unique across both streams"
        );

        let mic_count = segments
            .iter()
            .filter(|(_, k)| *k == AudioSourceKind::Microphone)
            .count();
        let system_count = segments
            .iter()
            .filter(|(_, k)| *k == AudioSourceKind::System)
            .count();
        assert_eq!(
            mic_count, 1,
            "the mic stream's segment is tagged Microphone"
        );
        assert_eq!(
            system_count, 1,
            "the system stream's segment is tagged System"
        );
    }

    #[test]
    fn spawn_live_stop_halts_an_endless_source() {
        struct Endless;
        impl AudioSource for Endless {
            fn info(&self) -> AudioSourceInfo {
                AudioSourceInfo {
                    kind: AudioSourceKind::Microphone,
                    name: "endless".to_owned(),
                }
            }
            fn next_frame(&mut self) -> Result<Option<AudioFrame>> {
                Ok(Some(AudioFrame::new(
                    vec![0.0; 160],
                    16_000,
                    1,
                    Duration::ZERO,
                )))
            }
        }

        let (tx, _rx) = mpsc::channel();
        let sink: EventSink = Box::new(move |event| {
            let _ = tx.send(event);
        });

        // The capture thread must observe stop, close the queue, and let both threads join.
        let session = Session::spawn_live(
            segmenter(),
            transcriber(vec![]),
            Box::new(Endless),
            sink,
            None,
        );
        session.stop().unwrap();
    }

    #[test]
    fn spawn_live_denoises_each_captured_frame() {
        use std::sync::atomic::AtomicUsize;

        // Passes audio through unchanged while counting how many frames it cleaned.
        struct CountingDenoiser(Arc<AtomicUsize>);
        impl Denoiser for CountingDenoiser {
            fn denoise(&mut self, audio: &[f32], _sample_rate: u32) -> Vec<f32> {
                self.0.fetch_add(1, Ordering::Relaxed);
                audio.to_vec()
            }
        }

        let calls = Arc::new(AtomicUsize::new(0));
        let frames = vec![
            frame(0.5, 0),
            frame(0.5, 100),
            frame(0.0, 200),
            frame(0.0, 300),
        ];
        let source: Box<dyn AudioSource> = Box::new(MockAudioSource::new(frames));
        let (tx, _rx) = mpsc::channel();
        let sink: EventSink = Box::new(move |event| {
            let _ = tx.send(event);
        });

        let session = Session::spawn_live(
            segmenter(),
            transcriber(vec![canned("x")]),
            source,
            sink,
            Some(Box::new(CountingDenoiser(Arc::clone(&calls)))),
        );
        session.join().unwrap();

        assert_eq!(
            calls.load(Ordering::Relaxed),
            4,
            "the denoiser runs once per captured frame"
        );
    }

    #[test]
    fn spawn_streaming_shares_an_id_across_partials_then_finalizes_per_utterance() {
        use std::collections::VecDeque;
        use wisp_core::engine::{StreamingAsrEngine, StreamingResult};
        use wisp_core::transcript::SegmentStatus;

        // A scripted streaming engine: one result per `accept_waveform` call.
        struct Scripted(VecDeque<StreamingResult>);
        impl StreamingAsrEngine for Scripted {
            fn accept_waveform(&mut self, _rate: u32, _samples: &[f32]) -> StreamingResult {
                self.0.pop_front().unwrap_or_default()
            }
            fn reset(&mut self) {}
        }

        let frames = vec![
            frame(0.5, 0),
            frame(0.5, 100),
            frame(0.5, 200),
            frame(0.5, 300),
        ];
        let source: Box<dyn AudioSource> = Box::new(MockAudioSource::new(frames));
        let results = VecDeque::from(vec![
            StreamingResult::new("he", false),
            StreamingResult::new("hello", false),
            StreamingResult::new("hello.", true),
            StreamingResult::new("next", false),
        ]);

        let (tx, rx) = mpsc::channel();
        let sink: EventSink = Box::new(move |event| {
            let _ = tx.send(event);
        });

        let session = Session::spawn_streaming(
            Box::new(Scripted(results)),
            source,
            sink,
            None,
            AudioSourceKind::Microphone,
        );
        session.join().unwrap();

        let segs: Vec<(u64, String, SegmentStatus)> = rx
            .try_iter()
            .map(|e| match e {
                TranscriptEvent::Segment(s) => (s.id, s.text, s.status),
                _ => panic!("unexpected event"),
            })
            .collect();
        assert_eq!(
            segs,
            vec![
                (0, "he".to_owned(), SegmentStatus::Partial),
                (0, "hello".to_owned(), SegmentStatus::Partial),
                (0, "hello.".to_owned(), SegmentStatus::Final),
                (1, "next".to_owned(), SegmentStatus::Partial),
                (1, "next".to_owned(), SegmentStatus::Final),
            ],
            "partials share the utterance id, the endpoint finalizes it, the next utterance gets a fresh id, and the open partial is flushed to a Final when the source ends"
        );
    }

    #[test]
    fn spawn_streaming_promotes_the_open_partial_to_final_when_the_stream_ends() {
        use std::collections::VecDeque;
        use wisp_core::engine::{StreamingAsrEngine, StreamingResult};
        use wisp_core::transcript::SegmentStatus;

        // The engine never reaches its own endpoint before the source ends (a Stop mid-sentence): the
        // last words only ever surface as a Partial, so the loop must promote them to a Final on exit
        // rather than drop the user's last spoken sentence.
        struct Scripted(VecDeque<StreamingResult>);
        impl StreamingAsrEngine for Scripted {
            fn accept_waveform(&mut self, _rate: u32, _samples: &[f32]) -> StreamingResult {
                self.0.pop_front().unwrap_or_default()
            }
            fn reset(&mut self) {}
        }

        let frames = vec![frame(0.5, 0), frame(0.5, 100)];
        let source: Box<dyn AudioSource> = Box::new(MockAudioSource::new(frames));
        let results = VecDeque::from(vec![
            StreamingResult::new("good", false),
            StreamingResult::new("good morning", false),
        ]);

        let (tx, rx) = mpsc::channel();
        let sink: EventSink = Box::new(move |event| {
            let _ = tx.send(event);
        });
        let session = Session::spawn_streaming(
            Box::new(Scripted(results)),
            source,
            sink,
            None,
            AudioSourceKind::Microphone,
        );
        session.join().unwrap();

        let segs: Vec<(u64, String, SegmentStatus)> = rx
            .try_iter()
            .map(|e| match e {
                TranscriptEvent::Segment(s) => (s.id, s.text, s.status),
                _ => panic!("unexpected event"),
            })
            .collect();
        assert_eq!(
            segs,
            vec![
                (0, "good".to_owned(), SegmentStatus::Partial),
                (0, "good morning".to_owned(), SegmentStatus::Partial),
                (0, "good morning".to_owned(), SegmentStatus::Final),
            ],
            "the open utterance's last partial is re-emitted as a Final (same id) when the stream ends"
        );
    }

    #[test]
    fn streaming_threads_the_aux_translation_onto_segments() {
        use std::collections::VecDeque;
        use wisp_core::engine::{StreamingAsrEngine, StreamingResult};
        use wisp_core::transcript::SegmentStatus;

        struct Scripted(VecDeque<StreamingResult>);
        impl StreamingAsrEngine for Scripted {
            fn accept_waveform(&mut self, _rate: u32, _samples: &[f32]) -> StreamingResult {
                self.0.pop_front().unwrap_or_default()
            }
            fn reset(&mut self) {}
        }

        // A dual-stream engine (cloud model session): a parallel `aux` rendering (the translation)
        // that can lead the verbatim text and rides each segment through to the UI.
        let frames = vec![frame(0.5, 0), frame(0.5, 100), frame(0.5, 200)];
        let source: Box<dyn AudioSource> = Box::new(MockAudioSource::new(frames));
        let results = VecDeque::from(vec![
            StreamingResult {
                text: String::new(),
                is_endpoint: false,
                aux: Some("Hel".to_owned()),
            },
            StreamingResult {
                text: "你好".to_owned(),
                is_endpoint: false,
                aux: Some("Hello".to_owned()),
            },
            StreamingResult {
                text: "你好。".to_owned(),
                is_endpoint: true,
                aux: Some("Hello.".to_owned()),
            },
        ]);

        let (tx, rx) = mpsc::channel();
        let sink: EventSink = Box::new(move |event| {
            let _ = tx.send(event);
        });
        let session = Session::spawn_streaming(
            Box::new(Scripted(results)),
            source,
            sink,
            None,
            AudioSourceKind::Microphone,
        );
        session.join().unwrap();

        let segs: Vec<(String, Option<String>, SegmentStatus)> = rx
            .try_iter()
            .map(|e| match e {
                TranscriptEvent::Segment(s) => (s.text, s.aux_text, s.status),
                _ => panic!("unexpected event"),
            })
            .collect();
        assert_eq!(
            segs,
            vec![
                (
                    String::new(),
                    Some("Hel".to_owned()),
                    SegmentStatus::Partial
                ),
                (
                    "你好".to_owned(),
                    Some("Hello".to_owned()),
                    SegmentStatus::Partial
                ),
                (
                    "你好。".to_owned(),
                    Some("Hello.".to_owned()),
                    SegmentStatus::Final
                ),
            ],
            "the translation rides each segment; an aux-only frame (empty verbatim) still emits"
        );
    }

    fn utt(len: usize) -> Utterance {
        Utterance {
            audio: vec![0.1; len],
            start: Duration::ZERO,
        }
    }

    #[test]
    fn coalesce_keeps_only_the_newest_partial() {
        let (finals, partials) = coalesce(vec![
            Job::Partial(0, AudioSourceKind::Microphone, utt(1)),
            Job::Partial(0, AudioSourceKind::Microphone, utt(2)),
        ]);
        assert!(finals.is_empty());
        assert_eq!(partials.len(), 1);
        assert_eq!(partials[0].2.audio.len(), 2, "the older partial is dropped");
    }

    #[test]
    fn coalesce_drops_a_partial_superseded_by_its_final() {
        let (finals, partials) = coalesce(vec![
            Job::Partial(0, AudioSourceKind::Microphone, utt(1)),
            Job::Final(0, AudioSourceKind::Microphone, utt(3)),
        ]);
        assert_eq!(finals.len(), 1);
        assert_eq!(finals[0].0, 0);
        assert!(
            partials.is_empty(),
            "the final supersedes the provisional partial"
        );
    }

    #[test]
    fn coalesce_keeps_a_partial_that_follows_the_last_final() {
        let (finals, partials) = coalesce(vec![
            Job::Final(0, AudioSourceKind::Microphone, utt(3)),
            Job::Partial(1, AudioSourceKind::Microphone, utt(1)),
        ]);
        assert_eq!(finals.len(), 1);
        assert_eq!(partials.len(), 1);
        assert_eq!(partials[0].0, 1, "the next utterance's partial survives");
    }

    #[test]
    fn coalesce_preserves_final_order() {
        let (finals, partials) = coalesce(vec![
            Job::Final(0, AudioSourceKind::Microphone, utt(1)),
            Job::Final(1, AudioSourceKind::Microphone, utt(2)),
        ]);
        let ids: Vec<u64> = finals.iter().map(|(id, _, _)| *id).collect();
        assert_eq!(ids, vec![0, 1]);
        assert!(partials.is_empty());
    }

    #[test]
    fn coalesce_keeps_a_partial_per_stream_and_only_its_own_final_clears_it() {
        // Two streams share the queue with globally-unique ids: a mic partial (id 0) and a system
        // partial (id 1) coexist; the mic's final closes only id 0, leaving the system partial alive
        // and still tagged with its own source kind.
        let (finals, partials) = coalesce(vec![
            Job::Partial(0, AudioSourceKind::Microphone, utt(1)),
            Job::Partial(1, AudioSourceKind::System, utt(2)),
            Job::Final(0, AudioSourceKind::Microphone, utt(3)),
        ]);
        assert_eq!(finals.len(), 1);
        assert_eq!(finals[0].0, 0);
        assert_eq!(finals[0].1, AudioSourceKind::Microphone);
        assert_eq!(
            partials.len(),
            1,
            "the sibling stream's partial is untouched"
        );
        assert_eq!(partials[0].0, 1);
        assert_eq!(partials[0].1, AudioSourceKind::System);
    }

    #[test]
    fn capture_emits_a_partial_then_a_final_sharing_one_id() {
        // Never closes on push, always exposes an in-progress partial, closes on flush (stream end)
        // — so we can observe the partial cadence and the shared id deterministically.
        struct OpenSegmenter {
            partial_len: usize,
            final_len: usize,
        }
        impl Segmenter for OpenSegmenter {
            fn push(&mut self, _m: &[f32], _t: Duration, _d: Duration) -> Vec<Utterance> {
                Vec::new()
            }
            fn flush(&mut self) -> Vec<Utterance> {
                vec![Utterance {
                    audio: vec![0.3; self.final_len],
                    start: Duration::ZERO,
                }]
            }
            fn partial(&self) -> Option<Utterance> {
                Some(Utterance {
                    audio: vec![0.2; self.partial_len],
                    start: Duration::ZERO,
                })
            }
        }

        // Seven 100 ms frames cross the 600 ms partial interval exactly once before the source ends.
        let frames: Vec<AudioFrame> = (0u64..7).map(|i| frame(0.5, i * 100)).collect();
        let source = MockAudioSource::new(frames);
        let seg = OpenSegmenter {
            partial_len: 800,
            final_len: 1_600,
        };

        let (tx, rx) = mpsc::channel();
        let stop = AtomicBool::new(false);
        // No transcribe thread here, so the decode time stays at the default (500 ms cadence).
        let decode_ms = AtomicU64::new(PARTIAL_DEFAULT.as_millis() as u64);
        let next_id = AtomicU64::new(0);
        let stream = LiveStream {
            segmenter: Box::new(seg),
            source: Box::new(source),
            denoiser: None,
            kind: AudioSourceKind::Microphone,
        };
        run_capture(stream, &tx, &decode_ms, &stop, &next_id).unwrap();
        drop(tx);

        let jobs: Vec<Job> = rx.try_iter().collect();
        assert_eq!(
            jobs.len(),
            2,
            "one partial (one interval) then the closing final"
        );
        match &jobs[0] {
            Job::Partial(id, kind, u) => {
                assert_eq!(*id, 0);
                assert_eq!(*kind, AudioSourceKind::Microphone);
                assert_eq!(u.audio.len(), 800);
            }
            other => panic!("expected a partial first, got {other:?}"),
        }
        match &jobs[1] {
            Job::Final(id, kind, u) => {
                assert_eq!(*id, 0, "the final reuses the open partial's id");
                assert_eq!(*kind, AudioSourceKind::Microphone);
                assert_eq!(u.audio.len(), 1_600);
            }
            other => panic!("expected a final second, got {other:?}"),
        }
    }

    #[test]
    fn partial_target_clamps_decode_time_to_bounds() {
        // A fast machine (decode under the floor) is held at the floor — no partial spam.
        assert_eq!(partial_target(&AtomicU64::new(50)), PARTIAL_FLOOR);
        // A typical decode passes through unchanged.
        assert_eq!(
            partial_target(&AtomicU64::new(420)),
            Duration::from_millis(420)
        );
        // A slow machine (decode over the ceiling) is capped so it still tries periodically.
        assert_eq!(partial_target(&AtomicU64::new(9_000)), PARTIAL_CEIL);
    }

    #[test]
    fn cap_partial_trims_to_the_trailing_window_and_shifts_start() {
        // 5 s of 16 kHz audio (80 000 samples) starting at t = 1 s, capped to a 2 s (32 000) window.
        let utterance = Utterance {
            audio: (0..80_000).map(|i| i as f32).collect(),
            start: Duration::from_secs(1),
        };
        let capped = cap_partial(utterance, 32_000, 16_000);

        assert_eq!(
            capped.audio.len(),
            32_000,
            "kept exactly the trailing window"
        );
        assert_eq!(
            capped.audio[0], 48_000.0,
            "the window is the last 32 000 samples (80 000 - 32 000)"
        );
        // start advanced by the dropped 48 000 samples = 3 s, so the window opens at 1 s + 3 s = 4 s.
        assert_eq!(capped.start, Duration::from_secs(4));
    }

    #[test]
    fn cap_partial_leaves_an_utterance_within_the_window_untouched() {
        let utterance = Utterance {
            audio: vec![0.1; 1_000],
            start: Duration::from_millis(500),
        };
        assert_eq!(
            cap_partial(utterance.clone(), 32_000, 16_000),
            utterance,
            "an utterance already within the window is returned unchanged"
        );
    }

    #[test]
    fn partial_window_env_var_name_is_pinned() {
        // Operators tune live partial latency via this env var; renaming it silently changes behaviour
        // for anyone who set it, so pin the literal.
        assert_eq!(PARTIAL_WINDOW_SECS_ENV, "WISP_PARTIAL_WINDOW_SECS");
    }
}
