//! Microphone capture via cpal.
//!
//! cpal's `Stream` is `!Send`, so it is owned by a dedicated capture thread that feeds frames to
//! a channel. [`MicSource`] holds only the receiver and a stop flag, so it stays `Send` and can
//! move onto the pipeline thread. Frames are emitted in the device's native rate/channels;
//! conversion to 16 kHz mono happens downstream in the pipeline.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Sender};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

use wisp_core::audio::{AudioFrame, AudioSource, AudioSourceInfo};
use wisp_core::channel::{frame_channel, FrameReceiver, FrameSender};
use wisp_core::error::{Result, WispError};
use wisp_core::transcript::AudioSourceKind;

/// Bounded capacity of the capture→pipeline frame channel. Drop-oldest on overflow keeps capture
/// real-time if the consumer briefly stalls, instead of letting a backlog grow without bound (which
/// would climb in latency and memory until the stream appears stuck). ~256 cpal buffers is a couple
/// of seconds of audio — far more headroom than a healthy consumer ever needs.
const FRAME_CHANNEL_CAPACITY: usize = 256;

/// Framing metadata shared with the capture callback.
#[derive(Clone, Copy)]
struct Framing {
    sample_rate: u32,
    channels: u16,
    start: Instant,
}

/// An [`AudioSource`] that captures the system default input device through cpal.
pub struct MicSource {
    rx: FrameReceiver,
    stop: Arc<AtomicBool>,
    handle: Option<JoinHandle<()>>,
    info: AudioSourceInfo,
}

impl MicSource {
    /// Opens the system default input device and starts capturing.
    pub fn from_default() -> Result<Self> {
        let device = cpal::default_host()
            .default_input_device()
            .ok_or_else(|| WispError::Audio("no default input device".to_owned()))?;
        Self::from_cpal_device(device)
    }

    /// Opens a specific input device by name and starts capturing.
    ///
    /// Used for loopback/virtual devices (e.g. BlackHole) that carry system/meeting audio.
    pub fn from_device(name: &str) -> Result<Self> {
        let device = cpal::default_host()
            .input_devices()
            .map_err(|e| WispError::Audio(format!("enumerate input devices: {e}")))?
            .find(|d| d.name().is_ok_and(|n| n == name))
            .ok_or_else(|| WispError::Audio(format!("input device '{name}' not found")))?;
        Self::from_cpal_device(device)
    }

    fn from_cpal_device(device: cpal::Device) -> Result<Self> {
        let name = device.name().unwrap_or_else(|_| "input".to_owned());
        let supported = device
            .default_input_config()
            .map_err(|e| WispError::Audio(format!("default input config: {e}")))?;

        let (tx, rx) = frame_channel(FRAME_CHANNEL_CAPACITY);
        let (ready_tx, ready_rx) = mpsc::channel::<std::result::Result<(), String>>();
        let stop = Arc::new(AtomicBool::new(false));
        let stop_for_thread = Arc::clone(&stop);

        let handle =
            thread::spawn(move || capture_loop(device, supported, tx, stop_for_thread, ready_tx));

        match ready_rx.recv() {
            Ok(Ok(())) => Ok(Self {
                rx,
                stop,
                handle: Some(handle),
                info: AudioSourceInfo {
                    kind: AudioSourceKind::Microphone,
                    name,
                },
            }),
            Ok(Err(e)) => Err(WispError::Audio(e)),
            Err(_) => Err(WispError::Audio(
                "capture thread exited before signalling readiness".to_owned(),
            )),
        }
    }
}

impl AudioSource for MicSource {
    fn info(&self) -> AudioSourceInfo {
        self.info.clone()
    }

    fn next_frame(&mut self) -> Result<Option<AudioFrame>> {
        Ok(self.rx.recv())
    }
}

impl Drop for MicSource {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(handle) = self.handle.take() {
            handle.thread().unpark();
            let _ = handle.join();
        }
    }
}

/// Lists the names of available input devices (microphones, loopback/virtual devices, …).
pub fn list_input_devices() -> Vec<String> {
    cpal::default_host()
        .input_devices()
        .map(|devices| devices.filter_map(|d| d.name().ok()).collect())
        .unwrap_or_default()
}

fn capture_loop(
    device: cpal::Device,
    supported: cpal::SupportedStreamConfig,
    tx: FrameSender,
    stop: Arc<AtomicBool>,
    ready: Sender<std::result::Result<(), String>>,
) {
    let format = supported.sample_format();
    let config: cpal::StreamConfig = supported.into();
    let framing = Framing {
        sample_rate: config.sample_rate.0,
        channels: config.channels,
        start: Instant::now(),
    };

    let stream = match build_stream(&device, &config, format, tx, framing) {
        Ok(stream) => stream,
        Err(e) => {
            let _ = ready.send(Err(e));
            return;
        }
    };

    if let Err(e) = stream.play() {
        let _ = ready.send(Err(format!("play stream: {e}")));
        return;
    }

    let _ = ready.send(Ok(()));

    while !stop.load(Ordering::Relaxed) {
        thread::park_timeout(Duration::from_millis(100));
    }

    drop(stream);
}

fn build_stream(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    format: cpal::SampleFormat,
    tx: FrameSender,
    framing: Framing,
) -> std::result::Result<cpal::Stream, String> {
    // Closing the sender on a stream error wakes the blocked consumer, so a device that drops or
    // is toggled mid-session ends that source cleanly instead of hanging on a dead stream.
    let on_error = {
        let tx = tx.clone();
        move |error: cpal::StreamError| {
            eprintln!("wisp: microphone stream error, closing source: {error}");
            tx.close();
        }
    };

    match format {
        cpal::SampleFormat::F32 => to_build_err(device.build_input_stream(
            config,
            move |data: &[f32], _| send_samples(&tx, data.to_vec(), framing),
            on_error,
            None,
        )),
        cpal::SampleFormat::I16 => to_build_err(device.build_input_stream(
            config,
            move |data: &[i16], _| {
                send_samples(
                    &tx,
                    data.iter().map(|s| f32::from(*s) / 32_768.0).collect(),
                    framing,
                );
            },
            on_error,
            None,
        )),
        cpal::SampleFormat::U16 => to_build_err(device.build_input_stream(
            config,
            move |data: &[u16], _| {
                send_samples(
                    &tx,
                    data.iter()
                        .map(|s| (f32::from(*s) - 32_768.0) / 32_768.0)
                        .collect(),
                    framing,
                );
            },
            on_error,
            None,
        )),
        other => Err(format!("unsupported sample format: {other:?}")),
    }
}

fn to_build_err(
    result: std::result::Result<cpal::Stream, cpal::BuildStreamError>,
) -> std::result::Result<cpal::Stream, String> {
    result.map_err(|e| format!("build input stream: {e}"))
}

fn send_samples(tx: &FrameSender, samples: Vec<f32>, framing: Framing) {
    let frame = AudioFrame::new(
        samples,
        framing.sample_rate,
        framing.channels,
        framing.start.elapsed(),
    );
    tx.send(frame);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires an input device and microphone permission; run with --ignored"]
    fn from_default_constructs_or_reports_no_device() {
        match MicSource::from_default() {
            Ok(source) => assert_eq!(source.info().kind, AudioSourceKind::Microphone),
            Err(WispError::Audio(_)) => {}
            Err(other) => panic!("unexpected error: {other}"),
        }
    }
}
