//! Speaker diarization: traits for attributing audio to speakers, plus the logic that maps
//! diarized spans onto transcript segments.
//!
//! Two shapes, because the two real implementations are genuinely different operations (ISP):
//! [`ClipDiarizer`] runs offline over a whole clip (most accurate — sees the full recording),
//! while [`Diarizer`] is a forward-only hook for a future live/incremental implementation. Only
//! the offline path is implemented today; the pure [`assign_speakers`] mapping it feeds is here in
//! the core so it is fully unit-testable.

use std::collections::BTreeMap;
use std::time::Duration;

use crate::error::Result;
use crate::transcript::{AudioSourceKind, SegmentStatus, SpeakerId, TranscriptSegment, Word};

/// Assigns a [`SpeakerId`] to a span of 16 kHz mono audio.
///
/// A forward-looking hook for live/incremental diarization (assign each utterance as it arrives).
/// Not yet implemented; offline file diarization uses [`ClipDiarizer`] instead.
pub trait Diarizer: Send {
    /// Returns the speaker the supplied audio most likely belongs to.
    fn identify(&mut self, audio: &[f32], sample_rate: u32) -> Result<SpeakerId>;
}

/// A contiguous span of audio attributed to a single speaker by diarization.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpeakerSpan {
    /// The speaker this span belongs to.
    pub speaker: SpeakerId,
    /// Start offset from the clip start.
    pub start: Duration,
    /// End offset from the clip start.
    pub end: Duration,
}

/// Offline speaker diarization over a whole clip.
///
/// Consumes the entire 16 kHz mono signal and returns speaker-labelled time spans. Whole-clip
/// (rather than per-utterance) processing lets the implementation cluster every voice across the
/// full recording, which is far more accurate than deciding speaker-by-speaker as audio streams.
pub trait ClipDiarizer: Send {
    /// Diarizes `audio`, returning the speaker spans it found.
    fn diarize_clip(&mut self, audio: &[f32], sample_rate: u32) -> Result<Vec<SpeakerSpan>>;
}

/// Label each segment with the speaker whose diarized spans overlap it the most.
///
/// Diarization boundaries rarely line up exactly with transcript segment boundaries, so each
/// segment is attributed to whichever speaker shares the most time with it. A segment that
/// overlaps no span keeps its existing speaker (`None` until set).
pub fn assign_speakers(segments: &mut [TranscriptSegment], spans: &[SpeakerSpan]) {
    for segment in segments {
        if let Some(speaker) = majority_speaker(segment.start, segment.end, spans) {
            segment.speaker = Some(speaker);
        }
    }
}

/// Re-attribute segments to speakers at **word** granularity, splitting any segment whose words
/// span a speaker change into one sub-segment per consecutive same-speaker run.
///
/// More precise than [`assign_speakers`], which paints a whole segment with one (majority) speaker:
/// when a single recognized segment actually contains two voices ("…thanks. — sure, go ahead"),
/// word-level timings let us cut it at the exact handover. A segment with no word timings can't be
/// split, so it falls back to whole-segment majority. Word order and text are preserved, and each
/// sub-segment inherits the parent's metadata (`id`, `status`, `source`, `confidence`) — callers
/// that need unique ids should re-assign them afterwards, since a split reuses the parent id.
pub fn attribute_speakers_by_word(
    segments: Vec<TranscriptSegment>,
    spans: &[SpeakerSpan],
) -> Vec<TranscriptSegment> {
    let mut out = Vec::with_capacity(segments.len());
    for segment in segments {
        if segment.words.is_empty() {
            out.push(with_majority_speaker(segment, spans));
        } else {
            split_segment_by_speaker(segment, spans, &mut out);
        }
    }
    out
}

/// Sets a segment's speaker to its whole-span majority, leaving it untouched if nothing overlaps.
fn with_majority_speaker(
    mut segment: TranscriptSegment,
    spans: &[SpeakerSpan],
) -> TranscriptSegment {
    if let Some(speaker) = majority_speaker(segment.start, segment.end, spans) {
        segment.speaker = Some(speaker);
    }
    segment
}

/// Splits one segment into consecutive same-speaker runs of its words, pushing each as a segment.
fn split_segment_by_speaker(
    segment: TranscriptSegment,
    spans: &[SpeakerSpan],
    out: &mut Vec<TranscriptSegment>,
) {
    let speakers: Vec<Option<SpeakerId>> = segment
        .words
        .iter()
        .map(|w| majority_speaker(w.start, w.end, spans))
        .collect();

    // Every word shares one speaker (the common case, incl. a single word): keep the segment whole
    // so its original text and timing are preserved exactly — only set the speaker.
    let first = speakers[0];
    if speakers.iter().all(|&s| s == first) {
        let mut seg = segment;
        if let Some(speaker) = first {
            seg.speaker = Some(speaker);
        }
        out.push(seg);
        return;
    }

    let TranscriptSegment {
        id,
        status,
        source,
        confidence,
        words,
        ..
    } = segment;

    let mut iter = words.into_iter().zip(speakers).peekable();
    while let Some((word, speaker)) = iter.next() {
        let mut run = vec![word];
        while iter.peek().is_some_and(|(_, next)| *next == speaker) {
            run.push(iter.next().expect("peeked").0);
        }
        out.push(run_segment(run, speaker, id, status, source, confidence));
    }
}

/// Builds a sub-segment from a same-speaker run of words, inheriting the parent's metadata.
fn run_segment(
    words: Vec<Word>,
    speaker: Option<SpeakerId>,
    id: u64,
    status: SegmentStatus,
    source: AudioSourceKind,
    confidence: Option<f32>,
) -> TranscriptSegment {
    let text: String = words.iter().map(|w| w.text.as_str()).collect();
    let start = words.first().map(|w| w.start).unwrap_or_default();
    let end = words.last().map(|w| w.end).unwrap_or_default();
    TranscriptSegment {
        id,
        text: text.trim().to_owned(),
        start,
        end,
        status,
        source,
        speaker,
        confidence,
        words,
    }
}

/// The speaker whose diarized spans overlap `[start, end]` the most, or `None` if none do.
///
/// Ties (equal overlap) resolve deterministically to the higher [`SpeakerId`].
fn majority_speaker(start: Duration, end: Duration, spans: &[SpeakerSpan]) -> Option<SpeakerId> {
    let mut by_speaker: BTreeMap<SpeakerId, Duration> = BTreeMap::new();
    for span in spans {
        let shared = overlap(start, end, span.start, span.end);
        if shared > Duration::ZERO {
            let total = by_speaker.entry(span.speaker).or_default();
            *total = total.saturating_add(shared);
        }
    }
    by_speaker
        .into_iter()
        .max_by_key(|&(_, total)| total)
        .map(|(speaker, _)| speaker)
}

/// Length of the time overlap between `[a_start, a_end]` and `[b_start, b_end]`.
fn overlap(a_start: Duration, a_end: Duration, b_start: Duration, b_end: Duration) -> Duration {
    let start = a_start.max(b_start);
    let end = a_end.min(b_end);
    end.saturating_sub(start)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transcript::AudioSourceKind;

    fn segment(id: u64, start_ms: u64, end_ms: u64) -> TranscriptSegment {
        TranscriptSegment::new(
            id,
            "x",
            Duration::from_millis(start_ms)..Duration::from_millis(end_ms),
            AudioSourceKind::File,
        )
    }

    fn span(speaker: u32, start_ms: u64, end_ms: u64) -> SpeakerSpan {
        SpeakerSpan {
            speaker: SpeakerId(speaker),
            start: Duration::from_millis(start_ms),
            end: Duration::from_millis(end_ms),
        }
    }

    #[test]
    fn assigns_the_only_overlapping_speaker() {
        let mut segments = [segment(0, 0, 1_000)];
        assign_speakers(&mut segments, &[span(7, 0, 1_000)]);
        assert_eq!(segments[0].speaker, Some(SpeakerId(7)));
    }

    #[test]
    fn picks_the_majority_overlap() {
        // Segment 0–1000 ms: speaker 0 owns 300 ms, speaker 1 owns 700 ms → speaker 1 wins.
        let mut segments = [segment(0, 0, 1_000)];
        assign_speakers(&mut segments, &[span(0, 0, 300), span(1, 300, 1_000)]);
        assert_eq!(segments[0].speaker, Some(SpeakerId(1)));
    }

    #[test]
    fn no_overlap_leaves_speaker_unset() {
        let mut segments = [segment(0, 0, 500)];
        assign_speakers(&mut segments, &[span(0, 600, 1_000)]);
        assert_eq!(segments[0].speaker, None);
    }

    #[test]
    fn empty_spans_leave_all_unset() {
        let mut segments = [segment(0, 0, 500), segment(1, 500, 1_000)];
        assign_speakers(&mut segments, &[]);
        assert!(segments.iter().all(|s| s.speaker.is_none()));
    }

    #[test]
    fn labels_each_segment_independently() {
        let mut segments = [segment(0, 0, 500), segment(1, 500, 1_000)];
        assign_speakers(&mut segments, &[span(2, 0, 500), span(5, 500, 1_000)]);
        assert_eq!(segments[0].speaker, Some(SpeakerId(2)));
        assert_eq!(segments[1].speaker, Some(SpeakerId(5)));
    }

    fn word(text: &str, start_ms: u64, end_ms: u64) -> Word {
        Word {
            text: text.to_owned(),
            start: Duration::from_millis(start_ms),
            end: Duration::from_millis(end_ms),
        }
    }

    /// A File segment whose span is derived from its words (start of first → end of last).
    fn seg_with_words(id: u64, words: Vec<Word>) -> TranscriptSegment {
        let start = words.first().map(|w| w.start).unwrap_or_default();
        let end = words.last().map(|w| w.end).unwrap_or_default();
        let text: String = words.iter().map(|w| w.text.as_str()).collect();
        let mut seg = TranscriptSegment::new(id, text.trim(), start..end, AudioSourceKind::File);
        seg.status = SegmentStatus::Final;
        seg.words = words;
        seg
    }

    #[test]
    fn word_attribution_keeps_single_speaker_segment_whole() {
        let seg = seg_with_words(0, vec![word("hello", 0, 500), word(" there", 500, 1_000)]);
        let out = attribute_speakers_by_word(vec![seg], &[span(1, 0, 1_000)]);
        assert_eq!(out.len(), 1, "one speaker must not split the segment");
        assert_eq!(out[0].speaker, Some(SpeakerId(1)));
        assert_eq!(out[0].text, "hello there");
        assert_eq!(out[0].words.len(), 2);
        assert_eq!(out[0].start, Duration::ZERO);
        assert_eq!(out[0].end, Duration::from_millis(1_000));
    }

    #[test]
    fn word_attribution_splits_at_a_speaker_change() {
        let seg = seg_with_words(9, vec![word("hello", 0, 500), word(" world", 500, 1_000)]);
        let out = attribute_speakers_by_word(vec![seg], &[span(2, 0, 500), span(5, 500, 1_000)]);

        assert_eq!(out.len(), 2, "a mid-segment handover must split into two");
        assert_eq!(out[0].speaker, Some(SpeakerId(2)));
        assert_eq!(out[0].text, "hello");
        assert_eq!(out[0].start, Duration::ZERO);
        assert_eq!(out[0].end, Duration::from_millis(500));
        assert_eq!(out[0].words, vec![word("hello", 0, 500)]);

        assert_eq!(out[1].speaker, Some(SpeakerId(5)));
        assert_eq!(
            out[1].text, "world",
            "leading space is trimmed on a new line"
        );
        assert_eq!(out[1].start, Duration::from_millis(500));
        assert_eq!(out[1].end, Duration::from_millis(1_000));
        assert_eq!(out[1].words, vec![word(" world", 500, 1_000)]);
    }

    #[test]
    fn word_attribution_falls_back_to_majority_without_words() {
        let seg = segment(3, 0, 1_000); // no word timings
        let out = attribute_speakers_by_word(vec![seg], &[span(0, 0, 300), span(1, 300, 1_000)]);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].speaker, Some(SpeakerId(1)), "majority overlap wins");
    }

    #[test]
    fn word_attribution_without_spans_is_a_noop() {
        let seg = seg_with_words(4, vec![word("a", 0, 500), word(" b", 500, 1_000)]);
        let out = attribute_speakers_by_word(vec![seg.clone()], &[]);
        assert_eq!(
            out,
            vec![seg],
            "no diarization → identical segment, no speaker"
        );
    }

    #[test]
    fn word_attribution_of_empty_input_is_empty() {
        let out = attribute_speakers_by_word(Vec::new(), &[span(1, 0, 1_000)]);
        assert!(out.is_empty());
    }

    #[test]
    fn word_attribution_isolates_an_unattributed_gap() {
        // Middle word overlaps no span → its own `None` run between two speaker-1 runs.
        let seg = seg_with_words(
            0,
            vec![
                word("a", 0, 300),
                word(" b", 400, 600), // 400–600 falls in the diarization gap
                word(" c", 700, 1_000),
            ],
        );
        let out = attribute_speakers_by_word(vec![seg], &[span(1, 0, 350), span(1, 650, 1_000)]);

        assert_eq!(out.len(), 3);
        assert_eq!(out[0].speaker, Some(SpeakerId(1)));
        assert_eq!(out[1].speaker, None, "the gap word is left unattributed");
        assert_eq!(out[1].text, "b");
        assert_eq!(out[2].speaker, Some(SpeakerId(1)));
    }

    #[test]
    fn word_takes_its_majority_overlap_speaker() {
        // The single word spans both speakers; speaker 1 owns more of it.
        let seg = seg_with_words(0, vec![word("hi", 0, 1_000)]);
        let out = attribute_speakers_by_word(vec![seg], &[span(0, 0, 300), span(1, 300, 1_000)]);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].speaker, Some(SpeakerId(1)));
    }

    #[test]
    fn split_sub_segments_inherit_parent_metadata() {
        let mut seg = seg_with_words(42, vec![word("x", 0, 500), word(" y", 500, 1_000)]);
        seg.confidence = Some(0.9);
        let out = attribute_speakers_by_word(vec![seg], &[span(2, 0, 500), span(5, 500, 1_000)]);

        assert_eq!(out.len(), 2);
        for sub in &out {
            assert_eq!(sub.id, 42, "split reuses the parent id (caller re-ids)");
            assert_eq!(sub.status, SegmentStatus::Final);
            assert_eq!(sub.source, AudioSourceKind::File);
            assert_eq!(sub.confidence, Some(0.9));
        }
    }

    #[test]
    fn word_attribution_splits_each_segment_independently() {
        let a = seg_with_words(0, vec![word("a", 0, 500), word(" b", 500, 1_000)]);
        let b = seg_with_words(1, vec![word("c", 1_000, 1_500)]);
        let out = attribute_speakers_by_word(
            vec![a, b],
            &[span(2, 0, 500), span(5, 500, 1_000), span(5, 1_000, 1_500)],
        );
        // First segment splits 2→ (spk2, spk5); second stays whole (spk5).
        assert_eq!(out.len(), 3);
        assert_eq!(out[0].speaker, Some(SpeakerId(2)));
        assert_eq!(out[1].speaker, Some(SpeakerId(5)));
        assert_eq!(out[2].speaker, Some(SpeakerId(5)));
        assert_eq!(out[2].text, "c");
    }

    #[test]
    fn word_attribution_preserves_word_order_and_text() {
        let words = vec![
            word("one", 0, 300),
            word(" two", 300, 600),
            word(" three", 600, 900),
        ];
        let seg = seg_with_words(0, words.clone());
        // Three different speakers → three single-word segments, words intact and in order.
        let out = attribute_speakers_by_word(
            vec![seg],
            &[span(1, 0, 300), span(2, 300, 600), span(3, 600, 900)],
        );
        let recombined: Vec<Word> = out.iter().flat_map(|s| s.words.clone()).collect();
        assert_eq!(recombined, words, "every word is preserved, in order");
    }
}
