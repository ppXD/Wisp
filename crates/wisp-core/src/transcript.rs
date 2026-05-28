//! Transcript domain types: segments, speaker labels, and the event stream.

use std::ops::Range;
use std::time::Duration;

/// Where a piece of audio came from.
///
/// `#[non_exhaustive]` so new kinds can be added without breaking downstream matches.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum AudioSourceKind {
    /// Local microphone input.
    Microphone,
    /// System / loopback audio (e.g. the far end of a meeting).
    System,
    /// Audio decoded from a file.
    File,
}

/// Opaque identifier for a distinct speaker within a session.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SpeakerId(pub u32);

/// Whether a segment is still being revised or has been committed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SegmentStatus {
    /// The text may still change as more audio arrives.
    Partial,
    /// The text is committed and will not change.
    Final,
}

/// A span of recognized speech.
#[derive(Debug, Clone, PartialEq)]
pub struct TranscriptSegment {
    /// Stable id for this segment within a session (lets the UI update it in place).
    pub id: u64,
    /// Recognized text.
    pub text: String,
    /// Start offset from the session start.
    pub start: Duration,
    /// End offset from the session start.
    pub end: Duration,
    /// Whether the text is partial or final.
    pub status: SegmentStatus,
    /// Which audio source produced it.
    pub source: AudioSourceKind,
    /// Assigned speaker, once diarization has run. `None` until then.
    pub speaker: Option<SpeakerId>,
    /// Engine confidence in `[0.0, 1.0]`, when available.
    pub confidence: Option<f32>,
}

impl TranscriptSegment {
    /// Creates a `Partial` segment with no speaker or confidence set.
    ///
    /// Callers set [`status`](Self::status), [`speaker`](Self::speaker), and
    /// [`confidence`](Self::confidence) afterwards as those become known.
    pub fn new(
        id: u64,
        text: impl Into<String>,
        span: Range<Duration>,
        source: AudioSourceKind,
    ) -> Self {
        Self {
            id,
            text: text.into(),
            start: span.start,
            end: span.end,
            status: SegmentStatus::Partial,
            source,
            speaker: None,
            confidence: None,
        }
    }

    /// Duration spanned by the segment (saturating at zero if `end < start`).
    pub fn duration(&self) -> Duration {
        self.end.saturating_sub(self.start)
    }
}

/// An event emitted by the transcription pipeline.
///
/// `#[non_exhaustive]` so new event kinds (errors, stream lifecycle, speaker relabels) can be
/// added without breaking downstream `match` statements.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum TranscriptEvent {
    /// A new or updated transcript segment.
    Segment(TranscriptSegment),
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seg() -> TranscriptSegment {
        TranscriptSegment::new(
            1,
            "hello",
            Duration::from_millis(100)..Duration::from_millis(600),
            AudioSourceKind::Microphone,
        )
    }

    #[test]
    fn new_segment_is_partial_with_no_speaker_or_confidence() {
        let s = seg();
        assert_eq!(s.status, SegmentStatus::Partial);
        assert!(s.speaker.is_none());
        assert!(s.confidence.is_none());
        assert_eq!(s.text, "hello");
    }

    #[test]
    fn duration_is_difference() {
        assert_eq!(seg().duration(), Duration::from_millis(500));
    }

    #[test]
    fn duration_saturates_when_reversed() {
        let mut s = seg();
        s.start = Duration::from_secs(2);
        s.end = Duration::from_secs(1);
        assert_eq!(s.duration(), Duration::ZERO);
    }

    #[test]
    fn event_carries_segment() {
        let ev = TranscriptEvent::Segment(seg());
        match ev {
            TranscriptEvent::Segment(s) => assert_eq!(s.id, 1),
        }
    }
}
