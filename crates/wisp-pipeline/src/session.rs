//! Background session lifecycle around a [`Pipeline`].

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};

use wisp_core::audio::AudioSource;
use wisp_core::error::{Result, WispError};
use wisp_core::transcript::TranscriptEvent;

use crate::pipeline::Pipeline;

/// A boxed, `Send` consumer of transcript events (e.g. one that forwards them to the UI).
pub type EventSink = Box<dyn FnMut(TranscriptEvent) + Send>;

/// A transcription run executing on its own thread.
///
/// The application's start/stop commands are thin wrappers over this — `start` → [`Session::spawn`],
/// `stop` → [`Session::stop`] — keeping the lifecycle logic off the UI layer.
pub struct Session {
    stop: Arc<AtomicBool>,
    handle: Option<JoinHandle<Result<()>>>,
}

impl Session {
    /// Spawns the pipeline on a background thread, driving `source` and forwarding events to
    /// `sink` until the source ends or [`stop`](Self::stop) is called.
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
            handle: Some(handle),
        }
    }

    /// Signals the session to stop and waits for the thread to finish, returning its run result.
    /// Use this for live sources (e.g. a microphone) that never end on their own.
    pub fn stop(mut self) -> Result<()> {
        self.stop.store(true, Ordering::Relaxed);
        self.take_join()
    }

    /// Waits for the session to finish on its own — e.g. a finite/file source reaching its end —
    /// without signalling stop, returning its run result.
    pub fn join(mut self) -> Result<()> {
        self.take_join()
    }

    fn take_join(&mut self) -> Result<()> {
        match self.handle.take() {
            Some(handle) => handle
                .join()
                .map_err(|_| WispError::Engine("transcription thread panicked".to_owned()))?,
            None => Ok(()),
        }
    }
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
}
