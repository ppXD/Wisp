//! Channel-backed audio source and a fan-out tee.
//!
//! These bridge the [`wisp_core::AudioSource`] interface and the bounded
//! [`frame_channel`](wisp_core::channel::frame_channel): [`ChannelSource`] turns a
//! [`FrameReceiver`] back into an `AudioSource`, and [`tee`] forwards one source to two receivers.
//! Together they let a single capture (e.g. system audio) feed two consumers — the meeting
//! transcription pipeline and the echo-cancellation far-end reference.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};

use wisp_core::audio::{AudioFrame, AudioSource, AudioSourceInfo};
use wisp_core::channel::{frame_channel, FrameReceiver};
use wisp_core::error::Result;

/// Per-branch capacity of a [`tee`] (drop-oldest on overflow, like the capture channel). Generous
/// headroom (~10-20 s) so a slow engine transcribing one utterance doesn't drop the next.
const TEE_CAPACITY: usize = 1024;

/// An [`AudioSource`] that reads frames from a [`FrameReceiver`].
///
/// Bridges a channel — e.g. one branch of a [`tee`] — back into the `AudioSource` interface the
/// pipeline consumes. Ends (`Ok(None)`) once the channel is closed and drained.
pub struct ChannelSource {
    rx: FrameReceiver,
    info: AudioSourceInfo,
}

impl ChannelSource {
    /// Wraps `rx` as a source described by `info`.
    pub fn new(rx: FrameReceiver, info: AudioSourceInfo) -> Self {
        Self { rx, info }
    }
}

impl AudioSource for ChannelSource {
    fn info(&self) -> AudioSourceInfo {
        self.info.clone()
    }

    fn next_frame(&mut self) -> Result<Option<AudioFrame>> {
        Ok(self.rx.recv())
    }
}

/// Handle to a running [`tee`] pump thread. Dropping it (or calling [`stop`](Tee::stop)) stops the
/// pump and closes both branches, so downstream consumers end cleanly.
pub struct Tee {
    stop: Arc<AtomicBool>,
    handle: Option<JoinHandle<()>>,
}

impl Tee {
    /// Signals the pump to stop and waits for it to finish (closing both branches).
    pub fn stop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

impl Drop for Tee {
    fn drop(&mut self) {
        self.stop();
    }
}

/// Forwards one [`AudioSource`] to two [`FrameReceiver`]s.
///
/// A pump thread takes ownership of `source` and sends each frame (cloned) to both branches with
/// drop-oldest backpressure. It stops when the source ends or errors, or when the returned [`Tee`]
/// is stopped/dropped; either way both branches are closed so consumers see end-of-stream.
pub fn tee(mut source: Box<dyn AudioSource>) -> (Tee, FrameReceiver, FrameReceiver) {
    let (tx_a, rx_a) = frame_channel(TEE_CAPACITY);
    let (tx_b, rx_b) = frame_channel(TEE_CAPACITY);
    let stop = Arc::new(AtomicBool::new(false));
    let stop_for_thread = Arc::clone(&stop);

    let handle = thread::spawn(move || {
        while !stop_for_thread.load(Ordering::Relaxed) {
            match source.next_frame() {
                Ok(Some(frame)) => {
                    tx_a.send(frame.clone());
                    tx_b.send(frame);
                }
                Ok(None) => break,
                Err(e) => {
                    eprintln!("wisp: tee source error, closing branches: {e}");
                    break;
                }
            }
        }
        tx_a.close();
        tx_b.close();
    });

    (
        Tee {
            stop,
            handle: Some(handle),
        },
        rx_a,
        rx_b,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;
    use wisp_core::testing::MockAudioSource;
    use wisp_core::transcript::AudioSourceKind;

    fn frame(v: f32) -> AudioFrame {
        AudioFrame::new(vec![v], 16_000, 1, Duration::ZERO)
    }

    /// A source that never ends — used to exercise stopping a live tee.
    struct Endless;

    impl AudioSource for Endless {
        fn info(&self) -> AudioSourceInfo {
            AudioSourceInfo {
                kind: AudioSourceKind::System,
                name: "endless".to_owned(),
            }
        }

        fn next_frame(&mut self) -> Result<Option<AudioFrame>> {
            Ok(Some(frame(0.5)))
        }
    }

    #[test]
    fn channel_source_yields_frames_then_ends_on_close() {
        let (tx, rx) = frame_channel(8);
        tx.send(frame(1.0));
        tx.send(frame(2.0));
        tx.close();

        let info = AudioSourceInfo {
            kind: AudioSourceKind::System,
            name: "sys".to_owned(),
        };
        let mut src = ChannelSource::new(rx, info.clone());

        assert_eq!(src.info(), info);
        assert_eq!(src.next_frame().unwrap().unwrap().samples, vec![1.0]);
        assert_eq!(src.next_frame().unwrap().unwrap().samples, vec![2.0]);
        assert!(src.next_frame().unwrap().is_none());
    }

    #[test]
    fn tee_forwards_every_frame_to_both_branches_then_closes() {
        let frames = vec![frame(1.0), frame(2.0), frame(3.0)];
        let (_tee, rx_a, rx_b) = tee(Box::new(MockAudioSource::new(frames.clone())));

        for expected in &frames {
            assert_eq!(rx_a.recv().unwrap().samples, expected.samples);
        }
        assert!(rx_a.recv().is_none()); // source ended → branch closed

        for expected in &frames {
            assert_eq!(rx_b.recv().unwrap().samples, expected.samples);
        }
        assert!(rx_b.recv().is_none());
    }

    #[test]
    fn tee_stop_closes_both_branches() {
        let (mut t, rx_a, rx_b) = tee(Box::new(Endless));
        assert!(rx_a.recv().is_some()); // pump is running

        t.stop(); // joins the pump, which closes both branches

        while rx_a.recv().is_some() {} // drain buffered, then None (closed)
        while rx_b.recv().is_some() {}
    }
}
