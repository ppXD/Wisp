//! Background session lifecycle around a [`Pipeline`].

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Sender};
use std::sync::Arc;
use std::thread::{self, JoinHandle};

use wisp_audio::{to_mono_16k, TARGET_SAMPLE_RATE};
use wisp_core::audio::AudioSource;
use wisp_core::denoise::Denoiser;
use wisp_core::error::{Result, WispError};
use wisp_core::transcript::TranscriptEvent;

use crate::pipeline::Pipeline;
use crate::segmenter::{Segmenter, Utterance};
use crate::transcriber::Transcriber;

/// A boxed, `Send` consumer of transcript events (e.g. one that forwards them to the UI).
pub type EventSink = Box<dyn FnMut(TranscriptEvent) + Send>;

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

    /// Spawns a *decoupled* live session: a capture+segmentation thread feeds complete utterances
    /// over a queue to a separate transcription thread.
    ///
    /// The capture thread runs at real-time — it converts frames to 16 kHz mono, segments them
    /// with `segmenter`, and hands each finished [`Utterance`] to the queue — so a slow engine
    /// can never stall capture or cause dropped audio mid-sentence. The transcription thread
    /// drains the queue, runs `transcriber`, and forwards segments to `sink`. On stop the capture
    /// thread flushes its buffered utterance and closes the queue; the transcription thread then
    /// finishes the backlog before the session joins.
    /// `denoiser`, when present, cleans every captured frame before segmentation, so both the VAD
    /// and the engine see noise-suppressed audio. It is stateful and runs on the capture thread.
    pub fn spawn_live(
        mut segmenter: Box<dyn Segmenter>,
        mut transcriber: Transcriber,
        mut source: Box<dyn AudioSource>,
        mut sink: EventSink,
        denoiser: Option<Box<dyn Denoiser>>,
    ) -> Self {
        let stop = Arc::new(AtomicBool::new(false));
        let stop_for_capture = Arc::clone(&stop);
        let (utterance_tx, utterance_rx) = mpsc::channel::<Utterance>();

        // Transcription thread: drains complete utterances and forwards segments. Ends when the
        // capture thread drops its sender (stop or source exhausted). A transcription error on one
        // utterance is logged and skipped rather than killing the session.
        let transcribe_handle = thread::spawn(move || {
            while let Ok(utterance) = utterance_rx.recv() {
                match transcriber.transcribe(&utterance) {
                    Ok(segments) => {
                        for segment in segments {
                            sink(TranscriptEvent::Segment(segment));
                        }
                    }
                    Err(e) => eprintln!("wisp: transcription error: {e}"),
                }
            }
            Ok(())
        });

        // Capture + segmentation thread: never blocks on the engine.
        let capture_handle = thread::spawn(move || {
            run_capture(
                &mut *segmenter,
                &mut *source,
                denoiser,
                &utterance_tx,
                &stop_for_capture,
            )
            // `utterance_tx` drops here, signalling the transcription thread to finish.
        });

        Self {
            stop,
            handles: vec![capture_handle, transcribe_handle],
        }
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

/// Capture loop for [`Session::spawn_live`]: pull frames, segment them, and queue finished
/// utterances until `stop` is set or the source ends; then flush any buffered utterance.
fn run_capture(
    segmenter: &mut dyn Segmenter,
    source: &mut dyn AudioSource,
    mut denoiser: Option<Box<dyn Denoiser>>,
    utterance_tx: &Sender<Utterance>,
    stop: &AtomicBool,
) -> Result<()> {
    while !stop.load(Ordering::Relaxed) {
        match source.next_frame()? {
            Some(frame) => {
                let mono = to_mono_16k(&frame);
                let mono = match &mut denoiser {
                    Some(d) => d.denoise(&mono, TARGET_SAMPLE_RATE),
                    None => mono,
                };
                for utterance in segmenter.push(&mono, frame.timestamp, frame.duration()) {
                    let _ = utterance_tx.send(utterance);
                }
            }
            None => break,
        }
    }
    for utterance in segmenter.flush() {
        let _ = utterance_tx.send(utterance);
    }
    Ok(())
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
}
