//! Voice-activity gating for long-form file transcription.
//!
//! Whisper hallucinates words in long stretches of non-speech (silence, music, applause). To remove
//! that without sacrificing the cross-window context that makes whole-clip decoding accurate, we
//! concatenate a clip's detected speech into one gap-free signal, transcribe *that*, then map the
//! engine's (compressed) timestamps back onto the original timeline.
//!
//! This is gating, not chunking: the speech is fed as one continuous clip, so the engine still
//! carries context across the whole conversation — only the non-speech gaps are dropped.

use std::time::Duration;

use wisp_core::transcript::TranscriptSegment;

use crate::segmenter::Utterance;

/// Sample rate the pipeline runs at (16 kHz mono).
const SAMPLE_RATE: u32 = 16_000;

/// One run of speech, located in both the original and the compressed (gap-free) timelines.
#[derive(Debug, Clone, PartialEq)]
struct GateSpan {
    /// Where this run starts in the original clip.
    orig_start: Duration,
    /// Where this run starts in the concatenated speech.
    compressed_start: Duration,
    /// How long the run lasts (identical in both timelines).
    duration: Duration,
}

/// A clip's speech concatenated gap-free, plus the map from compressed time back to original time.
#[derive(Debug, Clone, PartialEq)]
pub struct GatedClip {
    /// 16 kHz mono speech with the non-speech gaps removed — feed this to the engine.
    pub audio: Vec<f32>,
    /// Speech runs ordered by time; empty when no speech was detected.
    spans: Vec<GateSpan>,
}

impl GatedClip {
    /// Concatenates speech `utterances` (in capture order) into one gap-free clip with a timeline.
    pub fn from_utterances(utterances: Vec<Utterance>) -> Self {
        let mut audio = Vec::new();
        let mut spans = Vec::new();
        for utterance in utterances {
            if utterance.audio.is_empty() {
                continue;
            }
            spans.push(GateSpan {
                orig_start: utterance.start,
                compressed_start: samples_to_duration(audio.len()),
                duration: samples_to_duration(utterance.audio.len()),
            });
            audio.extend_from_slice(&utterance.audio);
        }
        Self { audio, spans }
    }

    /// Maps a compressed-timeline timestamp back to the original clip timeline.
    ///
    /// `at_end` chooses the convention at a run seam: an *end* timestamp that lands exactly on a
    /// boundary belongs to the run that just finished, while a *start* timestamp there belongs to
    /// the run about to begin. With no gating information (no speech detected) this is the identity.
    fn to_original(&self, compressed: Duration, at_end: bool) -> Duration {
        if self.spans.is_empty() {
            return compressed;
        }
        // The runs tile the compressed timeline contiguously from zero, so the run containing
        // `compressed` is the last one starting at or before it (strictly before, for an end).
        let count = if at_end {
            self.spans
                .partition_point(|s| s.compressed_start < compressed)
        } else {
            self.spans
                .partition_point(|s| s.compressed_start <= compressed)
        };
        let span = &self.spans[count.saturating_sub(1)];
        let offset = compressed
            .saturating_sub(span.compressed_start)
            .min(span.duration);
        span.orig_start + offset
    }
}

/// Rewrites segment and word timestamps from the compressed timeline back onto the original clip.
pub fn remap_to_original(segments: &mut [TranscriptSegment], gate: &GatedClip) {
    for segment in segments {
        segment.start = gate.to_original(segment.start, false);
        segment.end = gate.to_original(segment.end, true);
        for word in &mut segment.words {
            word.start = gate.to_original(word.start, false);
            word.end = gate.to_original(word.end, true);
        }
    }
}

/// Duration of `samples` mono samples at 16 kHz.
fn samples_to_duration(samples: usize) -> Duration {
    Duration::from_secs_f64(samples as f64 / SAMPLE_RATE as f64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use wisp_core::transcript::{AudioSourceKind, Word};

    /// `ms` milliseconds of (silent) audio is `ms * 16` samples at 16 kHz.
    fn utt(start_ms: u64, len_ms: u64) -> Utterance {
        Utterance {
            audio: vec![0.0; (len_ms * 16) as usize],
            start: Duration::from_millis(start_ms),
        }
    }

    fn ms(t: u64) -> Duration {
        Duration::from_millis(t)
    }

    fn seg_with_words(start: Duration, end: Duration, words: Vec<Word>) -> TranscriptSegment {
        let mut s = TranscriptSegment::new(0, "x", start..end, AudioSourceKind::File);
        s.words = words;
        s
    }

    fn word(start: Duration, end: Duration) -> Word {
        Word {
            text: "w".to_owned(),
            start,
            end,
        }
    }

    #[test]
    fn empty_clip_has_no_audio_and_is_identity() {
        let clip = GatedClip::from_utterances(Vec::new());
        assert!(clip.audio.is_empty());

        let mut segs = [seg_with_words(
            ms(300),
            ms(900),
            vec![word(ms(300), ms(900))],
        )];
        remap_to_original(&mut segs, &clip);
        // No speech runs → timestamps pass through unchanged.
        assert_eq!(segs[0].start, ms(300));
        assert_eq!(segs[0].end, ms(900));
        assert_eq!(segs[0].words[0].start, ms(300));
    }

    #[test]
    fn empty_utterances_are_skipped() {
        let clip = GatedClip::from_utterances(vec![utt(1_000, 0), utt(2_000, 500)]);
        // Only the 500 ms run survives; it owns the whole compressed timeline from zero.
        assert_eq!(clip.audio.len(), 500 * 16);
        assert_eq!(clip.to_original(ms(0), false), ms(2_000));
    }

    #[test]
    fn single_run_offsets_by_its_original_start() {
        // 1 s of speech that originally began at 2 s.
        let clip = GatedClip::from_utterances(vec![utt(2_000, 1_000)]);
        assert_eq!(clip.to_original(ms(0), false), ms(2_000));
        assert_eq!(clip.to_original(ms(500), false), ms(2_500));
        assert_eq!(clip.to_original(ms(1_000), true), ms(3_000), "run end");
        assert_eq!(
            clip.to_original(ms(9_000), true),
            ms(3_000),
            "past the end clamps to the run end"
        );
    }

    #[test]
    fn the_gap_between_runs_is_removed() {
        // u0: original 1.0–2.0 s → compressed 0.0–1.0 s. u1: original 5.0–6.0 s → compressed 1.0–2.0
        // s. The 3 s of silence between them is gone from the compressed timeline.
        let clip = GatedClip::from_utterances(vec![utt(1_000, 1_000), utt(5_000, 1_000)]);

        // A word wholly inside u0.
        assert_eq!(clip.to_original(ms(0), false), ms(1_000));
        assert_eq!(
            clip.to_original(ms(1_000), true),
            ms(2_000),
            "u0 end, not u1 start"
        );

        // A word wholly inside u1.
        assert_eq!(
            clip.to_original(ms(1_000), false),
            ms(5_000),
            "u1 start, not u0 end"
        );
        assert_eq!(clip.to_original(ms(2_000), true), ms(6_000));
    }

    #[test]
    fn remap_splits_a_segment_across_the_seam_correctly() {
        let clip = GatedClip::from_utterances(vec![utt(1_000, 1_000), utt(5_000, 1_000)]);
        // Two segments the engine produced in compressed time, each with one word.
        let mut segs = [
            seg_with_words(ms(0), ms(1_000), vec![word(ms(0), ms(1_000))]),
            seg_with_words(ms(1_000), ms(2_000), vec![word(ms(1_000), ms(2_000))]),
        ];
        remap_to_original(&mut segs, &clip);

        assert_eq!((segs[0].start, segs[0].end), (ms(1_000), ms(2_000)));
        assert_eq!(
            (segs[0].words[0].start, segs[0].words[0].end),
            (ms(1_000), ms(2_000))
        );
        assert_eq!((segs[1].start, segs[1].end), (ms(5_000), ms(6_000)));
        assert_eq!(
            (segs[1].words[0].start, segs[1].words[0].end),
            (ms(5_000), ms(6_000))
        );
    }

    #[test]
    fn remap_rewrites_every_word_in_a_segment() {
        let clip = GatedClip::from_utterances(vec![utt(10_000, 2_000)]);
        let mut segs = [seg_with_words(
            ms(0),
            ms(2_000),
            vec![word(ms(0), ms(500)), word(ms(500), ms(2_000))],
        )];
        remap_to_original(&mut segs, &clip);
        assert_eq!(
            (segs[0].words[0].start, segs[0].words[0].end),
            (ms(10_000), ms(10_500))
        );
        assert_eq!(
            (segs[0].words[1].start, segs[0].words[1].end),
            (ms(10_500), ms(12_000))
        );
    }
}
