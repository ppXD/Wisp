//! Turns complete utterances into transcript segments via an ASR engine.
//!
//! Owns the (slow) [`AsrEngine`] so it can run on its own thread, draining utterances produced by
//! the [`Segmenter`](crate::segmenter::Segmenter) without ever stalling the capture path.

use std::time::Duration;

use wisp_audio::{normalize_for_asr, TARGET_SAMPLE_RATE};
use wisp_core::diarize::Diarizer;
use wisp_core::engine::AsrEngine;
use wisp_core::error::Result;
use wisp_core::transcript::{AudioSourceKind, SegmentStatus, TranscriptSegment};

use crate::segmenter::Utterance;

/// Transcribes utterances into transcript segments, tagging each with the source kind.
///
/// Two entry points: the File pipeline uses [`transcribe`](Self::transcribe) (one segment per
/// engine sentence, sequential ids — granular for SRT/VTT export); the live session uses
/// [`transcribe_utterance`](Self::transcribe_utterance) (one merged segment with a caller-supplied
/// id and status, so a provisional `Partial` and its `Final` share an id and update in place).
pub struct Transcriber {
    engine: Box<dyn AsrEngine>,
    source_kind: AudioSourceKind,
    next_id: u64,
    diarizer: Option<Box<dyn Diarizer>>,
}

impl Transcriber {
    /// Creates a transcriber that tags segments as coming from `source_kind`.
    pub fn new(engine: Box<dyn AsrEngine>, source_kind: AudioSourceKind) -> Self {
        Self {
            engine,
            source_kind,
            next_id: 0,
            diarizer: None,
        }
    }

    /// Attaches a live diarizer that labels each utterance's segments with their speaker.
    pub fn with_diarizer(mut self, diarizer: Box<dyn Diarizer>) -> Self {
        self.diarizer = Some(diarizer);
        self
    }

    /// Transcribes one complete `utterance` into final segments (one per engine sentence).
    ///
    /// Empty / punctuation-only recognitions are dropped (ASR hallucinations on near-silence).
    /// Surviving segments get a sequential id, the configured source kind, `Final` status, and
    /// their times offset by the utterance's start. The engine is reset afterwards. Used by the
    /// File pipeline, where per-sentence granularity feeds SRT/VTT export.
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

        // Live diarization: label every segment of this utterance with its speaker. Best-effort —
        // a failure leaves the speaker unset rather than dropping the segment.
        if !segments.is_empty() {
            if let Some(diarizer) = &mut self.diarizer {
                if let Ok(speaker) = diarizer.identify(&utterance.audio, TARGET_SAMPLE_RATE) {
                    for segment in &mut segments {
                        segment.speaker = Some(speaker);
                    }
                }
            }
        }

        self.engine.reset();
        Ok(segments)
    }

    /// Transcribes `utterance` into a single segment carrying the caller's `id` and `status`.
    ///
    /// All of the engine's sub-segments are merged into one line (text concatenated, span = first
    /// start .. last end) so a multi-sentence utterance stays one updatable row that a `Partial`
    /// can refine and a `Final` can replace in place. Empty / punctuation-only recognitions return
    /// `None` (ASR hallucinations on near-silence). Diarization runs **only for `Final`** — a
    /// partial stays unlabelled, both for speed and because the speaker can still change before the
    /// utterance closes. The engine is reset afterwards so the next decode starts clean.
    pub fn transcribe_utterance(
        &mut self,
        id: u64,
        utterance: &Utterance,
        status: SegmentStatus,
    ) -> Result<Option<TranscriptSegment>> {
        let audio = normalize_for_asr(&utterance.audio, TARGET_SAMPLE_RATE);
        let result = self.engine.transcribe(&audio, TARGET_SAMPLE_RATE)?;
        self.engine.reset();

        let Some(mut segment) = merge_segments(result.segments) else {
            return Ok(None);
        };

        segment.id = id;
        segment.source = self.source_kind;
        segment.status = status;
        segment.start += utterance.start;
        segment.end += utterance.start;

        if status == SegmentStatus::Final {
            if let Some(diarizer) = &mut self.diarizer {
                if let Ok(speaker) = diarizer.identify(&utterance.audio, TARGET_SAMPLE_RATE) {
                    segment.speaker = Some(speaker);
                }
            }
        }

        Ok(Some(segment))
    }
}

/// Collapses an engine's (possibly multi-sentence) segments into one updatable line, or `None` if
/// there's no alphanumeric content. Text and word timings are concatenated in order; the span
/// covers the earliest start to the latest end. The first segment carries over any other fields.
fn merge_segments(segments: Vec<TranscriptSegment>) -> Option<TranscriptSegment> {
    let mut text = String::new();
    let mut words = Vec::new();
    let mut span: Option<(Duration, Duration)> = None;
    for s in &segments {
        text.push_str(&s.text);
        words.extend(s.words.iter().cloned());
        span = Some(match span {
            None => (s.start, s.end),
            Some((start, end)) => (start.min(s.start), end.max(s.end)),
        });
    }

    if !text.chars().any(char::is_alphanumeric) {
        return None;
    }

    let (start, end) = span.unwrap_or((Duration::ZERO, Duration::ZERO));
    let mut merged = segments.into_iter().next()?;
    merged.text = text;
    merged.words = words;
    merged.start = start;
    merged.end = end;
    Some(merged)
}

#[cfg(test)]
mod tests {
    use super::*;
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
    fn overrides_id_source_status_and_offsets_times() {
        let engine = Box::new(MockAsrEngine::new(vec![canned("hello")]));
        let mut transcriber = Transcriber::new(engine, AudioSourceKind::Microphone);

        let segment = transcriber
            .transcribe_utterance(7, &utterance(500), SegmentStatus::Final)
            .unwrap()
            .expect("a non-empty recognition yields a segment");

        assert_eq!(segment.id, 7);
        assert_eq!(segment.text, "hello");
        assert_eq!(segment.source, AudioSourceKind::Microphone);
        assert_eq!(segment.status, SegmentStatus::Final);
        assert_eq!(segment.start, Duration::from_millis(500));
        assert_eq!(segment.end, Duration::from_millis(550));
    }

    #[test]
    fn shares_one_id_across_a_partial_then_its_final() {
        // The capture thread reuses an id so a provisional partial and its final update one row.
        let engine = Box::new(MockAsrEngine::new(vec![
            canned("hi the"),
            canned("hi there"),
        ]));
        let mut transcriber = Transcriber::new(engine, AudioSourceKind::Microphone);

        let partial = transcriber
            .transcribe_utterance(3, &utterance(0), SegmentStatus::Partial)
            .unwrap()
            .unwrap();
        let committed = transcriber
            .transcribe_utterance(3, &utterance(0), SegmentStatus::Final)
            .unwrap()
            .unwrap();

        assert_eq!((partial.id, partial.status), (3, SegmentStatus::Partial));
        assert_eq!((committed.id, committed.status), (3, SegmentStatus::Final));
        assert_eq!(committed.text, "hi there");
    }

    #[test]
    fn drops_punctuation_only_recognition() {
        let engine = Box::new(MockAsrEngine::new(vec![canned("？")]));
        let mut transcriber = Transcriber::new(engine, AudioSourceKind::Microphone);

        let segment = transcriber
            .transcribe_utterance(0, &utterance(0), SegmentStatus::Final)
            .unwrap();
        assert!(segment.is_none());
    }

    #[test]
    fn merges_a_multi_sentence_utterance_into_one_line() {
        let engine = Box::new(MockAsrEngine::new(vec![TranscriptionResult {
            segments: vec![
                TranscriptSegment::new(
                    1,
                    "Hello world.",
                    Duration::ZERO..Duration::from_millis(500),
                    AudioSourceKind::File,
                ),
                TranscriptSegment::new(
                    2,
                    " How are you?",
                    Duration::from_millis(500)..Duration::from_millis(1_200),
                    AudioSourceKind::File,
                ),
            ],
        }]));
        let mut transcriber = Transcriber::new(engine, AudioSourceKind::Microphone);

        let segment = transcriber
            .transcribe_utterance(0, &utterance(100), SegmentStatus::Final)
            .unwrap()
            .unwrap();

        assert_eq!(segment.text, "Hello world. How are you?");
        // Span = earliest start .. latest end, each offset by the utterance start (100 ms).
        assert_eq!(segment.start, Duration::from_millis(100));
        assert_eq!(segment.end, Duration::from_millis(1_300));
    }

    #[test]
    fn final_labels_the_segment_with_its_speaker() {
        use wisp_core::diarize::Diarizer;
        use wisp_core::transcript::SpeakerId;

        struct FixedSpeaker(u32);
        impl Diarizer for FixedSpeaker {
            fn identify(&mut self, _audio: &[f32], _sample_rate: u32) -> Result<SpeakerId> {
                Ok(SpeakerId(self.0))
            }
        }

        let engine = Box::new(MockAsrEngine::new(vec![canned("hi")]));
        let mut transcriber = Transcriber::new(engine, AudioSourceKind::Microphone)
            .with_diarizer(Box::new(FixedSpeaker(3)));

        let segment = transcriber
            .transcribe_utterance(0, &utterance(0), SegmentStatus::Final)
            .unwrap()
            .unwrap();
        assert_eq!(segment.speaker, Some(SpeakerId(3)));
    }

    #[test]
    fn partial_skips_diarization() {
        use wisp_core::diarize::Diarizer;
        use wisp_core::transcript::SpeakerId;

        struct FixedSpeaker;
        impl Diarizer for FixedSpeaker {
            fn identify(&mut self, _audio: &[f32], _sample_rate: u32) -> Result<SpeakerId> {
                Ok(SpeakerId(9))
            }
        }

        let engine = Box::new(MockAsrEngine::new(vec![canned("hi")]));
        let mut transcriber = Transcriber::new(engine, AudioSourceKind::Microphone)
            .with_diarizer(Box::new(FixedSpeaker));

        let segment = transcriber
            .transcribe_utterance(0, &utterance(0), SegmentStatus::Partial)
            .unwrap()
            .unwrap();
        assert!(
            segment.speaker.is_none(),
            "a partial stays unlabelled — the speaker can still change before it closes"
        );
    }

    #[test]
    fn transcribe_assigns_sequential_ids_and_offsets_times() {
        let engine = Box::new(MockAsrEngine::new(vec![canned("one"), canned("two")]));
        let mut transcriber = Transcriber::new(engine, AudioSourceKind::System);

        let first = transcriber.transcribe(&utterance(0)).unwrap();
        let second = transcriber.transcribe(&utterance(100)).unwrap();

        assert_eq!((first[0].id, first[0].status), (0, SegmentStatus::Final));
        assert_eq!(first[0].source, AudioSourceKind::System);
        assert_eq!(second[0].id, 1);
        assert_eq!(second[0].start, Duration::from_millis(100));
    }

    #[test]
    fn transcribe_drops_punctuation_only_recognition() {
        let engine = Box::new(MockAsrEngine::new(vec![canned("？")]));
        let mut transcriber = Transcriber::new(engine, AudioSourceKind::Microphone);

        assert!(transcriber.transcribe(&utterance(0)).unwrap().is_empty());
    }
}
