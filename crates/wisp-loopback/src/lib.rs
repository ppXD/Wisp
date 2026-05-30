//! Windows system-audio capture via WASAPI loopback.
//!
//! Records whatever is playing on the default render endpoint (the "loopback" of the speakers) and
//! exposes it as a [`wisp_core::AudioSource`], so Windows gets the same one-click "capture the
//! meeting" Live experience the Mac gets from ScreenCaptureKit — no virtual device, no setup.
//! Windows only.
//!
//! The COM interfaces WASAPI hands back are not `Send`, so (like the macOS ScreenCaptureKit source)
//! they live entirely on a dedicated capture thread that feeds frames to a channel; the public
//! handle holds only the receiver + a stop flag and stays `Send`.

#![cfg(target_os = "windows")]

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Sender};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use windows::core::Result as WinResult;
use windows::Win32::Media::Audio::{
    eConsole, eRender, IAudioCaptureClient, IAudioClient, IMMDeviceEnumerator, MMDeviceEnumerator,
    AUDCLNT_BUFFERFLAGS_SILENT, AUDCLNT_SHAREMODE_SHARED, AUDCLNT_STREAMFLAGS_LOOPBACK,
};
use windows::Win32::System::Com::{
    CoCreateInstance, CoInitializeEx, CoTaskMemFree, CLSCTX_ALL, COINIT_MULTITHREADED,
};

use wisp_core::audio::{AudioFrame, AudioSource, AudioSourceInfo};
use wisp_core::channel::{frame_channel, FrameReceiver, FrameSender};
use wisp_core::error::{Result, WispError};
use wisp_core::transcript::AudioSourceKind;

/// Bounded capacity of the capture→pipeline frame channel. Drop-oldest on overflow keeps capture
/// real-time if the consumer briefly stalls (generous headroom so a slow engine transcribing one
/// utterance doesn't drop the next one's audio).
const FRAME_CHANNEL_CAPACITY: usize = 1024;

/// 100-nanosecond units per second — WASAPI's reference-time unit. A one-second shared buffer is
/// ample headroom for a 10 ms poll.
const REFTIMES_PER_SEC: i64 = 10_000_000;

/// How often the capture thread wakes to drain buffered packets. Well under the buffer length, so
/// loopback audio never overflows while keeping latency low.
const POLL: Duration = Duration::from_millis(10);

/// An [`AudioSource`] capturing system audio through WASAPI loopback.
pub struct WasapiLoopbackSource {
    rx: FrameReceiver,
    stop: Arc<AtomicBool>,
    handle: Option<JoinHandle<()>>,
}

impl WasapiLoopbackSource {
    /// Starts capturing the default render endpoint's audio. Errors if WASAPI can't initialize
    /// (no output device, COM failure) so the caller can degrade to mic-only.
    pub fn new() -> Result<Self> {
        let (tx, rx) = frame_channel(FRAME_CHANNEL_CAPACITY);
        let (ready_tx, ready_rx) = mpsc::channel::<std::result::Result<(), String>>();
        let stop = Arc::new(AtomicBool::new(false));
        let stop_for_thread = Arc::clone(&stop);

        let handle = thread::spawn(move || capture_thread(&tx, &stop_for_thread, &ready_tx));

        match ready_rx.recv() {
            Ok(Ok(())) => Ok(Self {
                rx,
                stop,
                handle: Some(handle),
            }),
            Ok(Err(e)) => Err(WispError::Audio(e)),
            Err(_) => Err(WispError::Audio(
                "WASAPI loopback thread exited before signalling readiness".to_owned(),
            )),
        }
    }
}

impl AudioSource for WasapiLoopbackSource {
    fn info(&self) -> AudioSourceInfo {
        AudioSourceInfo {
            kind: AudioSourceKind::System,
            name: "System audio".to_owned(),
        }
    }

    fn next_frame(&mut self) -> Result<Option<AudioFrame>> {
        Ok(self.rx.recv())
    }
}

impl Drop for WasapiLoopbackSource {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

/// Runs the capture, signalling readiness to `new` once audio is flowing, then closes the channel
/// so a blocked receiver ends cleanly when capture stops or fails.
fn capture_thread(
    tx: &FrameSender,
    stop: &AtomicBool,
    ready: &Sender<std::result::Result<(), String>>,
) {
    // SAFETY: every COM object created here stays on this thread and is released (implicitly on
    // drop, the buffer explicitly) before the thread exits.
    if let Err(e) = unsafe { run_capture(tx, stop, ready) } {
        // Only reached for a failure *before* readiness was signalled; tell `new` to degrade.
        let _ = ready.send(Err(e));
    }
    tx.close();
}

/// Builds the WASAPI loopback client and pumps audio into `tx` until `stop`. Returns `Err` only for
/// a setup failure before `ready` is signalled; once capture starts, mid-stream errors are logged
/// and end the loop with `Ok`.
unsafe fn run_capture(
    tx: &FrameSender,
    stop: &AtomicBool,
    ready: &Sender<std::result::Result<(), String>>,
) -> std::result::Result<(), String> {
    // A fresh thread → initialize COM (multithreaded apartment) for it.
    let _ = CoInitializeEx(None, COINIT_MULTITHREADED);

    let enumerator: IMMDeviceEnumerator = CoCreateInstance(&MMDeviceEnumerator, None, CLSCTX_ALL)
        .map_err(label("device enumerator"))?;
    let device = enumerator
        .GetDefaultAudioEndpoint(eRender, eConsole)
        .map_err(label("default render endpoint"))?;
    let client: IAudioClient = device
        .Activate(CLSCTX_ALL, None)
        .map_err(label("activate audio client"))?;

    // The shared-mode mix format is what the endpoint is actually mixing to — capture matches it,
    // and the pipeline resamples to 16 kHz downstream.
    let format = client.GetMixFormat().map_err(label("mix format"))?;
    let channels = (*format).nChannels as usize;
    let sample_rate = (*format).nSamplesPerSec;
    let bits = (*format).wBitsPerSample;

    client
        .Initialize(
            AUDCLNT_SHAREMODE_SHARED,
            AUDCLNT_STREAMFLAGS_LOOPBACK,
            REFTIMES_PER_SEC,
            0,
            format,
            None,
        )
        .map_err(label("initialize"))?;
    // `Initialize` copied the format; release the buffer `GetMixFormat` allocated.
    CoTaskMemFree(Some(format as *const _));

    let capture: IAudioCaptureClient = client.GetService().map_err(label("capture client"))?;
    client.Start().map_err(label("start"))?;

    let _ = ready.send(Ok(()));

    let start = Instant::now();
    while !stop.load(Ordering::Relaxed) {
        thread::sleep(POLL);
        if let Err(e) = drain(&capture, tx, channels, sample_rate, bits, start) {
            eprintln!("wisp: WASAPI loopback capture error ({e}); stopping system audio");
            break;
        }
    }

    let _ = client.Stop();
    Ok(())
}

/// Drains every packet currently buffered, emitting one mono [`AudioFrame`] per non-silent packet.
unsafe fn drain(
    capture: &IAudioCaptureClient,
    tx: &FrameSender,
    channels: usize,
    sample_rate: u32,
    bits: u16,
    start: Instant,
) -> WinResult<()> {
    loop {
        if capture.GetNextPacketSize()? == 0 {
            return Ok(());
        }

        let mut data: *mut u8 = std::ptr::null_mut();
        let mut frames: u32 = 0;
        let mut flags: u32 = 0;
        capture.GetBuffer(&mut data, &mut frames, &mut flags, None, None)?;

        // The SILENT flag means "treat as silence regardless of bytes" — skip it; the pipeline
        // handles gaps, and emitting nothing is cheaper than emitting zeros.
        let silent = (flags & AUDCLNT_BUFFERFLAGS_SILENT.0 as u32) != 0;
        if !silent && frames > 0 && channels > 0 && !data.is_null() {
            let mono = downmix(data, frames as usize, channels, bits);
            if !mono.is_empty() {
                tx.send(AudioFrame::new(mono, sample_rate, 1, start.elapsed()));
            }
        }

        capture.ReleaseBuffer(frames)?;
    }
}

/// Downmixes one WASAPI packet (interleaved, `bits`-per-sample) to mono `f32`. The shared-mode mix
/// format is 32-bit float in practice; 16-bit PCM is handled defensively. Other depths yield empty.
unsafe fn downmix(data: *const u8, frames: usize, channels: usize, bits: u16) -> Vec<f32> {
    let total = frames * channels;
    match bits {
        32 => {
            let samples = std::slice::from_raw_parts(data as *const f32, total);
            samples
                .chunks(channels)
                .map(|c| c.iter().sum::<f32>() / channels as f32)
                .collect()
        }
        16 => {
            let samples = std::slice::from_raw_parts(data as *const i16, total);
            samples
                .chunks(channels)
                .map(|c| c.iter().map(|&s| s as f32 / 32_768.0).sum::<f32>() / channels as f32)
                .collect()
        }
        _ => Vec::new(),
    }
}

/// Tags a `windows` error with the call that produced it, for a readable degradation notice.
fn label(what: &'static str) -> impl Fn(windows::core::Error) -> String {
    move |e| format!("{what}: {e}")
}
