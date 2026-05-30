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
use crate::transcript::{SpeakerId, TranscriptSegment};

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
        let mut by_speaker: BTreeMap<SpeakerId, Duration> = BTreeMap::new();
        for span in spans {
            let shared = overlap(segment.start, segment.end, span.start, span.end);
            if shared > Duration::ZERO {
                let total = by_speaker.entry(span.speaker).or_default();
                *total = total.saturating_add(shared);
            }
        }
        if let Some((&speaker, _)) = by_speaker.iter().max_by_key(|&(_, &total)| total) {
            segment.speaker = Some(speaker);
        }
    }
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
}
