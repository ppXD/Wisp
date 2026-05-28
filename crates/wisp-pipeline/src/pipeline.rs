//! The VAD-segmented transcription pipeline.

use std::time::Duration;

use wisp_audio::{to_mono_16k, TARGET_SAMPLE_RATE};
use wisp_core::audio::{AudioFrame, AudioSource};
use wisp_core::engine::AsrEngine;
use wisp_core::error::Result;
use wisp_core::transcript::{AudioSourceKind, SegmentStatus, TranscriptEvent};

use crate::vad::Vad;

/// Default trailing silence that ends an utterance.
pub const DEFAULT_SILENCE_HANGOVER: Duration = Duration::from_millis(400);

/// Drives audio through VAD segmentation and an ASR engine, emitting transcript events.
///
/// Frames are converted to 16 kHz mono, accumulated while the [`Vad`] reports speech, and once
/// trailing silence reaches the hangover the buffered utterance is transcribed and emitted as
/// `Final` segments. Emitting in-progress `Partial` segments is a planned refinement; the event
/// model already supports it.
pub struct Pipeline {
    engine: Box<dyn AsrEngine>,
    vad: Box<dyn Vad>,
    source_kind: AudioSourceKind,
    silence_hangover: Duration,
    next_id: u64,
    utterance: Vec<f32>,
    utterance_start: Option<Duration>,
    silence_accum: Duration,
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
            engine,
            vad,
            source_kind,
            silence_hangover,
            next_id: 0,
            utterance: Vec::new(),
            utterance_start: None,
            silence_accum: Duration::ZERO,
        }
    }

    /// Runs until `source` is exhausted, invoking `sink` for every emitted event.
    pub fn run(
        &mut self,
        source: &mut dyn AudioSource,
        sink: &mut dyn FnMut(TranscriptEvent),
    ) -> Result<()> {
        while let Some(frame) = source.next_frame()? {
            self.process_frame(&frame, sink)?;
        }
        self.finalize(sink)
    }

    fn process_frame(
        &mut self,
        frame: &AudioFrame,
        sink: &mut dyn FnMut(TranscriptEvent),
    ) -> Result<()> {
        let mono = to_mono_16k(frame);

        if self.vad.is_speech(&mono) {
            if self.utterance.is_empty() {
                self.utterance_start = Some(frame.timestamp);
            }
            self.utterance.extend_from_slice(&mono);
            self.silence_accum = Duration::ZERO;
            return Ok(());
        }

        if !self.utterance.is_empty() {
            self.silence_accum += frame.duration();
            if self.silence_accum >= self.silence_hangover {
                self.finalize(sink)?;
            }
        }

        Ok(())
    }

    fn finalize(&mut self, sink: &mut dyn FnMut(TranscriptEvent)) -> Result<()> {
        if self.utterance.is_empty() {
            return Ok(());
        }

        let start = self.utterance_start.unwrap_or(Duration::ZERO);
        let result = self
            .engine
            .transcribe(&self.utterance, TARGET_SAMPLE_RATE)?;

        for mut segment in result.segments {
            segment.id = self.next_id;
            segment.source = self.source_kind;
            segment.status = SegmentStatus::Final;
            segment.start += start;
            segment.end += start;
            self.next_id += 1;
            sink(TranscriptEvent::Segment(segment));
        }

        self.engine.reset();
        self.utterance.clear();
        self.utterance_start = None;
        self.silence_accum = Duration::ZERO;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vad::EnergyVad;
    use wisp_core::engine::TranscriptionResult;
    use wisp_core::testing::{MockAsrEngine, MockAudioSource};
    use wisp_core::transcript::TranscriptSegment;

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
}
