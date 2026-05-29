//! The VAD-segmented transcription pipeline.

use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use wisp_audio::to_mono_16k;
use wisp_core::audio::{AudioFrame, AudioSource};
use wisp_core::engine::AsrEngine;
use wisp_core::error::Result;
use wisp_core::transcript::{AudioSourceKind, TranscriptEvent};

use crate::segmenter::{Segmenter, Utterance};
use crate::transcriber::Transcriber;
use crate::vad::Vad;

/// Default trailing silence that ends an utterance. Generous enough that a brief mid-sentence
/// pause doesn't cut the utterance short (which truncates the transcript).
pub const DEFAULT_SILENCE_HANGOVER: Duration = Duration::from_millis(700);

/// Drives audio through VAD segmentation and an ASR engine, emitting transcript events.
///
/// Frames are converted to 16 kHz mono, accumulated by a [`Segmenter`] while it reports speech,
/// and once trailing silence reaches the hangover the buffered utterance is transcribed by a
/// [`Transcriber`] and emitted as `Final` segments.
///
/// This is the *synchronous* composition — segmentation and (slow) transcription share one thread,
/// which is fine for finite sources (e.g. a file) where latency doesn't matter. Live sources use
/// [`Session::spawn_live`](crate::session::Session::spawn_live) instead, which runs the same two
/// pieces on separate threads so a slow engine never stalls capture.
pub struct Pipeline {
    segmenter: Segmenter,
    transcriber: Transcriber,
}

impl Pipeline {
    /// Creates a pipeline tagging segments with `source_kind` and ending utterances after
    /// `silence_hangover` of trailing silence.
    pub fn new(
        engine: Box<dyn AsrEngine>,
        vad: Box<dyn Vad>,
        source_kind: AudioSourceKind,
        silence_hangover: Duration,
    ) -> Self {
        Self {
            segmenter: Segmenter::new(vad, silence_hangover),
            transcriber: Transcriber::new(engine, source_kind),
        }
    }

    /// Runs until `source` is exhausted, invoking `sink` for every emitted event.
    pub fn run(
        &mut self,
        source: &mut dyn AudioSource,
        sink: &mut dyn FnMut(TranscriptEvent),
    ) -> Result<()> {
        self.run_until(source, sink, &AtomicBool::new(false))
    }

    /// Like [`run`](Self::run) but also stops once `stop` is set — for live sources that never end
    /// on their own. The in-progress utterance is finalized before returning either way.
    pub fn run_until(
        &mut self,
        source: &mut dyn AudioSource,
        sink: &mut dyn FnMut(TranscriptEvent),
        stop: &AtomicBool,
    ) -> Result<()> {
        while !stop.load(Ordering::Relaxed) {
            match source.next_frame()? {
                Some(frame) => self.process_frame(&frame, sink)?,
                None => break,
            }
        }
        self.finalize(sink)
    }

    fn process_frame(
        &mut self,
        frame: &AudioFrame,
        sink: &mut dyn FnMut(TranscriptEvent),
    ) -> Result<()> {
        let mono = to_mono_16k(frame);
        if let Some(utterance) = self
            .segmenter
            .push(&mono, frame.timestamp, frame.duration())
        {
            self.emit(&utterance, sink)?;
        }
        Ok(())
    }

    fn finalize(&mut self, sink: &mut dyn FnMut(TranscriptEvent)) -> Result<()> {
        if let Some(utterance) = self.segmenter.flush() {
            self.emit(&utterance, sink)?;
        }
        Ok(())
    }

    fn emit(&mut self, utterance: &Utterance, sink: &mut dyn FnMut(TranscriptEvent)) -> Result<()> {
        for segment in self.transcriber.transcribe(utterance)? {
            sink(TranscriptEvent::Segment(segment));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vad::EnergyVad;
    use wisp_audio::TARGET_SAMPLE_RATE;
    use wisp_core::engine::TranscriptionResult;
    use wisp_core::testing::{MockAsrEngine, MockAudioSource};
    use wisp_core::transcript::{SegmentStatus, TranscriptSegment};

    /// 0.1 s of 16 kHz mono at amplitude `amp`, stamped at `t_ms`.
    fn frame(amp: f32, t_ms: u64) -> AudioFrame {
        AudioFrame::new(
            vec![amp; 1_600],
            TARGET_SAMPLE_RATE,
            1,
            Duration::from_millis(t_ms),
        )
    }

    fn canned(text: &str) -> TranscriptionResult {
        TranscriptionResult {
            segments: vec![TranscriptSegment::new(
                999,
                text,
                Duration::ZERO..Duration::from_millis(50),
                AudioSourceKind::File,
            )],
        }
    }

    fn run(frames: Vec<AudioFrame>, results: Vec<TranscriptionResult>) -> Vec<TranscriptSegment> {
        let mut source = MockAudioSource::new(frames);
        let engine = Box::new(MockAsrEngine::new(results));
        let vad = Box::new(EnergyVad::new(0.01));
        let mut pipeline = Pipeline::new(
            engine,
            vad,
            AudioSourceKind::Microphone,
            Duration::from_millis(150),
        );

        let mut segments = Vec::new();
        let mut sink = |event: TranscriptEvent| match event {
            TranscriptEvent::Segment(s) => segments.push(s),
            _ => panic!("unexpected event kind"),
        };
        pipeline.run(&mut source, &mut sink).unwrap();
        segments
    }

    #[test]
    fn segments_two_utterances_with_offset_timestamps_and_sequential_ids() {
        // silence, speech, speech, silence, silence(->finalize), speech, silence, silence(->finalize)
        let frames = vec![
            frame(0.0, 0),
            frame(0.5, 100),
            frame(0.5, 200),
            frame(0.0, 300),
            frame(0.0, 400),
            frame(0.5, 500),
            frame(0.0, 600),
            frame(0.0, 700),
        ];
        let segments = run(frames, vec![canned("one"), canned("two")]);

        assert_eq!(segments.len(), 2);

        assert_eq!(segments[0].id, 0);
        assert_eq!(segments[0].text, "one");
        assert_eq!(segments[0].source, AudioSourceKind::Microphone);
        assert_eq!(segments[0].status, SegmentStatus::Final);
        // First utterance starts at the first speech frame (t=100ms); canned seg offset 0..50ms.
        assert_eq!(segments[0].start, Duration::from_millis(100));
        assert_eq!(segments[0].end, Duration::from_millis(150));

        assert_eq!(segments[1].id, 1);
        assert_eq!(segments[1].text, "two");
        assert_eq!(segments[1].start, Duration::from_millis(500));
    }

    #[test]
    fn finalizes_pending_utterance_at_end_of_stream() {
        let frames = vec![frame(0.5, 0), frame(0.5, 100)];
        let segments = run(frames, vec![canned("hello")]);

        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0].text, "hello");
        assert_eq!(segments[0].start, Duration::ZERO);
    }

    #[test]
    fn all_silence_emits_nothing() {
        let frames = vec![frame(0.0, 0), frame(0.0, 100), frame(0.0, 200)];
        let segments = run(frames, vec![canned("unused")]);
        assert!(segments.is_empty());
    }

    #[test]
    fn drops_punctuation_only_transcription() {
        // One spoken utterance whose recognition is just punctuation — an ASR hallucination.
        let frames = vec![frame(0.5, 0), frame(0.5, 100)];
        let segments = run(frames, vec![canned("？")]);
        assert!(segments.is_empty());
    }
}
