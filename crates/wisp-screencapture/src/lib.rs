//! macOS system-audio capture via ScreenCaptureKit (no virtual device).
//!
//! Captures system/display audio and exposes it as a [`wisp_core::AudioSource`]. The first use
//! triggers the one-time macOS Screen Recording permission prompt. macOS only.

#![cfg(target_os = "macos")]

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Sender};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use screencapturekit::cm::AudioBufferList;
use screencapturekit::prelude::*;

use wisp_core::audio::{AudioFrame, AudioSource, AudioSourceInfo};
use wisp_core::channel::{frame_channel, FrameReceiver, FrameSender};
use wisp_core::error::{Result, WispError};
use wisp_core::transcript::AudioSourceKind;

/// Rate requested from ScreenCaptureKit; the pipeline resamples to 16 kHz downstream.
const CAPTURE_SAMPLE_RATE: u32 = 48_000;

/// Bounded capacity of the capture→pipeline frame channel. Drop-oldest on overflow keeps capture
/// real-time if the consumer briefly stalls. Generous headroom (~10-20 s) so a slow engine
/// transcribing one utterance doesn't drop the audio of the next.
const FRAME_CHANNEL_CAPACITY: usize = 1024;

/// An [`AudioSource`] capturing system audio through ScreenCaptureKit.
///
/// The `SCStream` (which is not `Send`) is owned by a dedicated thread that feeds frames to a
/// channel; this struct holds only the receiver + a stop flag, so it stays `Send`.
pub struct ScreenCaptureSource {
    rx: FrameReceiver,
    stop: Arc<AtomicBool>,
    handle: Option<JoinHandle<()>>,
}

impl ScreenCaptureSource {
    /// Starts capturing system audio. Prompts for Screen Recording permission on first use;
    /// errors if permission is denied or no display is available.
    pub fn new() -> Result<Self> {
        let (tx, rx) = frame_channel(FRAME_CHANNEL_CAPACITY);
        let (ready_tx, ready_rx) = mpsc::channel::<std::result::Result<(), String>>();
        let stop = Arc::new(AtomicBool::new(false));
        let stop_for_thread = Arc::clone(&stop);

        let handle = thread::spawn(move || capture_loop(tx, &stop_for_thread, &ready_tx));

        match ready_rx.recv() {
            Ok(Ok(())) => Ok(Self {
                rx,
                stop,
                handle: Some(handle),
            }),
            Ok(Err(e)) => Err(WispError::Audio(e)),
            Err(_) => Err(WispError::Audio(
                "screen-capture thread exited before signalling readiness".to_owned(),
            )),
        }
    }
}

impl AudioSource for ScreenCaptureSource {
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

impl Drop for ScreenCaptureSource {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(handle) = self.handle.take() {
            handle.thread().unpark();
            let _ = handle.join();
        }
    }
}

/// ScreenCaptureKit output handler: turns each audio sample buffer into a mono [`AudioFrame`].
///
/// `SCStream` may invoke this from arbitrary dispatch-queue threads, so it must be `Send + Sync`;
/// [`FrameSender`] is already `Sync`, so it is held directly (no `Mutex` needed).
struct AudioHandler {
    tx: FrameSender,
    start: Instant,
}

impl SCStreamOutputTrait for AudioHandler {
    fn did_output_sample_buffer(&self, sample: CMSampleBuffer, of_type: SCStreamOutputType) {
        if !matches!(of_type, SCStreamOutputType::Audio) {
            return;
        }

        let Some(list) = sample.audio_buffer_list() else {
            return;
        };
        let mono = downmix_buffer_list(&list);
        if mono.is_empty() {
            return;
        }

        let frame = AudioFrame::new(mono, CAPTURE_SAMPLE_RATE, 1, self.start.elapsed());
        self.tx.send(frame);
    }
}

fn capture_loop(
    tx: FrameSender,
    stop: &AtomicBool,
    ready: &Sender<std::result::Result<(), String>>,
) {
    match build_and_start(tx) {
        Ok(stream) => {
            let _ = ready.send(Ok(()));
            while !stop.load(Ordering::Relaxed) {
                thread::park_timeout(Duration::from_millis(100));
            }
            let _ = stream.stop_capture();
            drop(stream);
        }
        Err(e) => {
            let _ = ready.send(Err(e));
        }
    }
}

fn build_and_start(tx: FrameSender) -> std::result::Result<SCStream, String> {
    let content = SCShareableContent::get().map_err(|e| format!("shareable content: {e}"))?;
    let displays = content.displays();
    let display = displays
        .first()
        .ok_or_else(|| "no display available".to_owned())?;

    let filter = SCContentFilter::create()
        .with_display(display)
        .with_excluding_windows(&[])
        .build();

    let config = SCStreamConfiguration::new()
        .with_captures_audio(true)
        .with_sample_rate(CAPTURE_SAMPLE_RATE as i32)
        .with_channel_count(2);

    let mut stream = SCStream::new(&filter, &config);
    stream.add_output_handler(
        AudioHandler {
            tx,
            start: Instant::now(),
        },
        SCStreamOutputType::Audio,
    );
    stream
        .start_capture()
        .map_err(|e| format!("start capture: {e}"))?;
    Ok(stream)
}

/// Downmixes a Float32 [`AudioBufferList`] (interleaved or planar) to mono.
fn downmix_buffer_list(list: &AudioBufferList) -> Vec<f32> {
    let buffers = list.num_buffers();
    if buffers == 0 {
        return Vec::new();
    }

    if buffers == 1 {
        let Some(buffer) = list.get(0) else {
            return Vec::new();
        };
        let channels = buffer.number_channels.max(1) as usize;
        let samples = bytes_to_f32(buffer.data());
        if channels <= 1 {
            return samples;
        }
        return samples
            .chunks(channels)
            .map(|c| c.iter().sum::<f32>() / channels as f32)
            .collect();
    }

    // Planar: one channel per buffer — average element-wise.
    let channels: Vec<Vec<f32>> = (0..buffers)
        .filter_map(|i| list.get(i).map(|b| bytes_to_f32(b.data())))
        .collect();
    let len = channels.iter().map(Vec::len).min().unwrap_or(0);
    (0..len)
        .map(|i| channels.iter().map(|c| c[i]).sum::<f32>() / channels.len() as f32)
        .collect()
}

fn bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}
