//! Cross-stream echo dedup — a safety net layered on top of acoustic echo cancellation.
//!
//! Even with AEC, a loud speaker can leave residual echo: the microphone ("me") stream occasionally
//! re-transcribes a phrase that the system ("meeting") stream already produced. Because the echo
//! only ever flows system → mic (the meeting capture is clean digital audio), this filter is
//! one-directional: it remembers recent **meeting** segments and drops a **mic** segment whose text
//! closely matches one of them within a short time window.
//!
//! Three guards keep it from dropping genuine speech: a minimum length (don't dedup "ok"/"yeah"), a
//! high similarity threshold, and a time window (only segments that overlap in time can match).
//!
//! Caveat: it can only match a mic segment against meeting segments already seen, so a residual echo
//! that finalizes *before* its meeting counterpart still slips through. AEC removes the bulk; this
//! nets the common residual case without delaying any output.

use std::collections::VecDeque;
use std::time::Duration;

use crate::transcript::{AudioSourceKind, SegmentStatus, TranscriptSegment};

/// How far back a meeting segment can be and still count as the echo source for a mic segment.
const DEFAULT_WINDOW: Duration = Duration::from_secs(5);
/// Segments shorter than this (normalized character count) are never treated as echo — short
/// utterances ("ok", "yeah", "好的") are too easy to match by chance, and the echoes worth dropping
/// are full phrases.
const DEFAULT_MIN_CHARS: usize = 6;
/// Minimum text similarity (0..=1) for a mic segment to be considered an echo of a meeting segment.
const DEFAULT_SIMILARITY: f32 = 0.8;

/// Drops microphone segments that echo a recent system/meeting segment.
///
/// Feed every emitted [`TranscriptSegment`] through [`admit`](Self::admit): meeting segments are
/// always admitted (and remembered); mic segments are admitted unless they echo a remembered one.
pub struct CrossStreamEchoFilter {
    window: Duration,
    min_chars: usize,
    similarity: f32,
    recent_meeting: VecDeque<(Duration, Vec<char>)>,
    now: Duration,
}

impl Default for CrossStreamEchoFilter {
    fn default() -> Self {
        Self::with_params(DEFAULT_WINDOW, DEFAULT_MIN_CHARS, DEFAULT_SIMILARITY)
    }
}

impl CrossStreamEchoFilter {
    /// Creates a filter with the production defaults.
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a filter with explicit parameters (used in tests).
    pub fn with_params(window: Duration, min_chars: usize, similarity: f32) -> Self {
        Self {
            window,
            min_chars,
            similarity,
            recent_meeting: VecDeque::new(),
            now: Duration::ZERO,
        }
    }

    /// Records `segment` and returns whether it should be emitted.
    ///
    /// System segments are always admitted; a committed meeting **final** is remembered as an echo
    /// candidate (a provisional partial is incomplete and would cause false drops). A microphone
    /// segment — **partial or final** — is dropped (returns `false`) when it closely matches a
    /// remembered meeting line, so a residual echo never even flickers on screen as a partial. Mic
    /// text is never remembered, so checking partials can't prime a mic line against itself.
    pub fn admit(&mut self, segment: &TranscriptSegment) -> bool {
        self.now = self.now.max(segment.end);
        self.evict_expired();

        let normalized = normalize(&segment.text);

        match segment.source {
            AudioSourceKind::System => {
                if segment.status == SegmentStatus::Final && normalized.len() >= self.min_chars {
                    self.recent_meeting.push_back((segment.end, normalized));
                }
                true
            }
            AudioSourceKind::Microphone => {
                if normalized.len() < self.min_chars {
                    return true; // too short to safely treat as echo
                }
                let is_echo = self
                    .recent_meeting
                    .iter()
                    .any(|(_, meeting)| similarity(&normalized, meeting) >= self.similarity);
                !is_echo
            }
            _ => true,
        }
    }

    /// Drops remembered meeting segments older than the window relative to the latest time seen.
    fn evict_expired(&mut self) {
        while let Some((end, _)) = self.recent_meeting.front() {
            if *end + self.window < self.now {
                self.recent_meeting.pop_front();
            } else {
                break;
            }
        }
    }
}

/// Lowercases and strips everything but letters/digits (keeps CJK, drops spaces/punctuation).
fn normalize(text: &str) -> Vec<char> {
    text.chars()
        .filter(|c| c.is_alphanumeric())
        .flat_map(char::to_lowercase)
        .collect()
}

/// Similarity in `0.0..=1.0` from the character-level Levenshtein distance.
fn similarity(a: &[char], b: &[char]) -> f32 {
    let max = a.len().max(b.len());
    if max == 0 {
        return 1.0;
    }
    1.0 - levenshtein(a, b) as f32 / max as f32
}

/// Levenshtein edit distance over characters (two-row DP).
fn levenshtein(a: &[char], b: &[char]) -> usize {
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    let mut curr = vec![0usize; b.len() + 1];

    for (i, ca) in a.iter().enumerate() {
        curr[0] = i + 1;
        for (j, cb) in b.iter().enumerate() {
            let cost = usize::from(ca != cb);
            curr[j + 1] = (prev[j + 1] + 1).min(curr[j] + 1).min(prev[j] + cost);
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    prev[b.len()]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transcript::SegmentStatus;

    fn seg(id: u64, text: &str, source: AudioSourceKind, end_ms: u64) -> TranscriptSegment {
        let mut s = TranscriptSegment::new(
            id,
            text,
            Duration::from_millis(end_ms.saturating_sub(500))..Duration::from_millis(end_ms),
            source,
        );
        s.status = SegmentStatus::Final;
        s
    }

    #[test]
    fn drops_identical_mic_echo_of_recent_meeting() {
        let mut f = CrossStreamEchoFilter::new();
        assert!(f.admit(&seg(
            1,
            "做了一份AD，你要不要先看一下？",
            AudioSourceKind::System,
            1000
        )));
        // Same phrase re-heard by the mic shortly after → echo, dropped.
        assert!(!f.admit(&seg(
            2,
            "做了一份AD，你要不要先看一下？",
            AudioSourceKind::Microphone,
            1100
        )));
    }

    #[test]
    fn drops_near_identical_mic_echo() {
        let mut f = CrossStreamEchoFilter::new();
        f.admit(&seg(1, "我共享一下稍等。", AudioSourceKind::System, 1000));
        // One extra leading character → still ~0.86 similar → dropped.
        assert!(!f.admit(&seg(2, "共享一下稍等。", AudioSourceKind::Microphone, 1050)));
    }

    #[test]
    fn keeps_distinct_mic_speech() {
        let mut f = CrossStreamEchoFilter::new();
        f.admit(&seg(1, "今天的天气很好啊", AudioSourceKind::System, 1000));
        assert!(f.admit(&seg(
            2,
            "我们开始开会吧各位",
            AudioSourceKind::Microphone,
            1100
        )));
    }

    #[test]
    fn keeps_short_mic_segments_even_if_matching() {
        let mut f = CrossStreamEchoFilter::new();
        f.admit(&seg(1, "yeah", AudioSourceKind::System, 1000));
        // Below the min-length guard → never treated as echo.
        assert!(f.admit(&seg(2, "yeah", AudioSourceKind::Microphone, 1050)));
    }

    #[test]
    fn keeps_mic_echo_outside_the_time_window() {
        let mut f = CrossStreamEchoFilter::with_params(Duration::from_secs(2), 4, 0.8);
        f.admit(&seg(
            1,
            "这是一段很长的会议内容",
            AudioSourceKind::System,
            1000,
        ));
        // 4 s later, well past the 2 s window → the meeting candidate has been evicted.
        assert!(f.admit(&seg(
            2,
            "这是一段很长的会议内容",
            AudioSourceKind::Microphone,
            5000
        )));
    }

    #[test]
    fn always_keeps_meeting_segments() {
        let mut f = CrossStreamEchoFilter::new();
        assert!(f.admit(&seg(1, "重复的会议内容啊", AudioSourceKind::System, 1000)));
        assert!(f.admit(&seg(2, "重复的会议内容啊", AudioSourceKind::System, 1500)));
    }

    #[test]
    fn mic_echo_finalizing_before_meeting_slips_through() {
        // Documents the ordering caveat: the meeting counterpart is not yet recorded.
        let mut f = CrossStreamEchoFilter::new();
        assert!(f.admit(&seg(
            1,
            "先出现的麦克风回音内容",
            AudioSourceKind::Microphone,
            1000
        )));
        assert!(f.admit(&seg(
            2,
            "先出现的麦克风回音内容",
            AudioSourceKind::System,
            1100
        )));
    }

    fn partial(id: u64, text: &str, source: AudioSourceKind, end_ms: u64) -> TranscriptSegment {
        let mut s = seg(id, text, source, end_ms);
        s.status = SegmentStatus::Partial;
        s
    }

    #[test]
    fn suppresses_a_mic_partial_echo() {
        let mut f = CrossStreamEchoFilter::new();
        f.admit(&seg(
            1,
            "做了一份AD，你要不要先看一下？",
            AudioSourceKind::System,
            1000,
        ));
        // The same phrase re-heard by the mic while still provisional is an echo too — drop it so it
        // never flickers on screen ahead of (or instead of) its final.
        assert!(!f.admit(&partial(
            2,
            "做了一份AD，你要不要先看一下？",
            AudioSourceKind::Microphone,
            1100
        )));
    }

    #[test]
    fn a_meeting_partial_is_not_remembered_as_an_echo_source() {
        let mut f = CrossStreamEchoFilter::new();
        // A provisional meeting partial must not become an echo source — it can be garbled or
        // incomplete, which would wrongly drop genuine mic speech that resembles it.
        f.admit(&partial(
            1,
            "这是一段会议内容啊",
            AudioSourceKind::System,
            1000,
        ));
        assert!(f.admit(&seg(
            2,
            "这是一段会议内容啊",
            AudioSourceKind::Microphone,
            1100
        )));
    }
}
