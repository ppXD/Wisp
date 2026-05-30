//! Turns complete utterances into transcript segments via an ASR engine.
//!
//! Owns the (slow) [`AsrEngine`] so it can run on its own thread, draining utterances produced by
//! the [`Segmenter`](crate::segmenter::Segmenter) without ever stalling the capture path.

use wisp_audio::{normalize_for_asr, TARGET_SAMPLE_RATE};
use wisp_core::engine::AsrEngine;
use wisp_core::error::Result;
use wisp_core::transcript::{AudioSourceKind, SegmentStatus, TranscriptSegment};

use crate::segmenter::Utterance;

/// Transcribes utterances, tagging each emitted segment with a monotonic id and the source kind.
pub struct Transcriber {
    engine: Box<dyn AsrEngine>,
    source_kind: AudioSourceKind,
    next_id: u64,
}

impl Transcriber {
    /// Creates a transcriber that tags segments as coming from `source_kind`.
    pub fn new(engine: Box<dyn AsrEngine>, source_kind: AudioSourceKind) -> Self {
        Self {
            engine,
            source_kind,
            next_id: 0,
        }
    }

    /// Transcribes one complete `utterance` into final segments.
    ///
    /// Empty / punctuation-only recognitions are dropped (ASR hallucinations on near-silence).
    /// Surviving segments get a sequential id, the configured source kind, `Final` status, and
    /// their times offset by the utterance's start. The engine is reset afterwards so the next
    /// utterance starts clean.
    pub fn transcribe(&mut self, utterance: &Utterance) -> Result<Vec<TranscriptSegment>> {
        let audio = normalize_for_asr(&utterance.audio, TARGET_SAMPLE_RATE);
        let result = self.engine.transcribe(&audio, TARGET_SAMPLE_RATE)?;

        let mut segments = Vec::new();
        for mut segment in result.segments {
            if !segment.text.chars().any(char::is_alphanumeric) {
                continue;
            }
            segment.id = self.next_id;
            segment.source = self.source_kind;
            segment.status = SegmentStatus::Final;
            segment.start += utterance.start;
            segment.end += utterance.start;
            self.next_id += 1;
            segments.push(segment);
        }

        self.engine.reset();
        Ok(segments)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;
    use wisp_core::engine::TranscriptionResult;
    use wisp_core::testing::MockAsrEngine;

    fn utterance(start_ms: u64) -> Utterance {
        Utterance {
            audio: vec![0.1; 1_600],
            start: Duration::from_millis(start_ms),
        }
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

    #[test]
    fn offsets_times_and_overrides_id_source_status() {
        let engine = Box::new(MockAsrEngine::new(vec![canned("hello")]));
        let mut transcriber = Transcriber::new(engine, AudioSourceKind::Microphone);

        let segments = transcriber.transcribe(&utterance(500)).unwrap();

        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0].id, 0);
        assert_eq!(segments[0].text, "hello");
        assert_eq!(segments[0].source, AudioSourceKind::Microphone);
        assert_eq!(segments[0].status, SegmentStatus::Final);
        assert_eq!(segments[0].start, Duration::from_millis(500));
        assert_eq!(segments[0].end, Duration::from_millis(550));
    }

    #[test]
    fn ids_are_sequential_across_utterances() {
        let engine = Box::new(MockAsrEngine::new(vec![canned("one"), canned("two")]));
        let mut transcriber = Transcriber::new(engine, AudioSourceKind::System);

        let first = transcriber.transcribe(&utterance(0)).unwrap();
        let second = transcriber.transcribe(&utterance(100)).unwrap();

        assert_eq!(first[0].id, 0);
        assert_eq!(second[0].id, 1);
        assert_eq!(second[0].source, AudioSourceKind::System);
    }

    #[test]
    fn drops_punctuation_only_recognition() {
        let engine = Box::new(MockAsrEngine::new(vec![canned("？")]));
        let mut transcriber = Transcriber::new(engine, AudioSourceKind::Microphone);

        let segments = transcriber.transcribe(&utterance(0)).unwrap();
        assert!(segments.is_empty());
    }
}
