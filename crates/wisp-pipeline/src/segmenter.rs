//! Voice-activity segmentation: turns a frame stream into complete utterances.
//!
//! [`Segmenter`] is a narrow trait so segmentation strategies plug in interchangeably — a naive
//! energy gate ([`EnergySegmenter`]) today, a neural VAD (Silero) tomorrow — without touching the
//! pipeline or session that drive it. Segmentation is engine-free and runs on the real-time capture
//! thread, so it must never block on transcription.

use std::time::Duration;

use crate::vad::Vad;

/// A complete spoken utterance: 16 kHz mono PCM plus the timestamp of its first speech frame.
///
/// Produced by a [`Segmenter`] and consumed by the transcriber — decoupling the (fast) capture path
/// from the (slow) ASR engine.
#[derive(Debug, Clone, PartialEq)]
pub struct Utterance {
    /// 16 kHz mono samples spanning the utterance.
    pub audio: Vec<f32>,
    /// Capture timestamp of the utterance's first speech frame, used to offset segment times.
    pub start: Duration,
}

/// Splits a 16 kHz mono frame stream into complete utterances.
///
/// A frame may complete zero, one, or several utterances (a neural VAD draining a backlog can emit
/// more than one), so [`push`](Self::push) returns a `Vec`. Implementors hold no ASR engine —
/// emitting an [`Utterance`] is buffer book-keeping only, cheap enough for the capture thread.
pub trait Segmenter: Send {
    /// Feeds one 16 kHz mono frame (`mono`) stamped at `timestamp` and lasting `frame_duration`,
    /// returning any utterances this frame completed (usually none, or one).
    fn push(
        &mut self,
        mono: &[f32],
        timestamp: Duration,
        frame_duration: Duration,
    ) -> Vec<Utterance>;

    /// Emits any buffered utterance at end-of-stream — the last one may not have been followed by
    /// enough trailing silence to close on its own.
    fn flush(&mut self) -> Vec<Utterance>;

    /// A snapshot of the audio accumulated so far for the currently-open utterance, for decoding a
    /// *provisional* partial transcript before the utterance closes.
    ///
    /// Returns `None` when no utterance is in progress (silence) or when the segmenter can't expose
    /// in-progress audio. The default is `None`, so a segmenter stays final-only unless it opts in —
    /// a non-breaking extension. The returned audio is never authoritative: the matching final
    /// utterance from [`push`](Self::push)/[`flush`](Self::flush) always supersedes it.
    fn partial(&self) -> Option<Utterance> {
        None
    }
}

/// Energy-gate segmenter: accumulates speech frames into an utterance, splitting on trailing
/// silence. Simple and dependency-free; the cross-platform fallback when no neural VAD is loaded.
pub struct EnergySegmenter {
    vad: Box<dyn Vad>,
    silence_hangover: Duration,
    utterance: Vec<f32>,
    utterance_start: Option<Duration>,
    silence_accum: Duration,
}

impl EnergySegmenter {
    /// Creates a segmenter that ends an utterance after `silence_hangover` of trailing silence.
    pub fn new(vad: Box<dyn Vad>, silence_hangover: Duration) -> Self {
        Self {
            vad,
            silence_hangover,
            utterance: Vec::new(),
            utterance_start: None,
            silence_accum: Duration::ZERO,
        }
    }

    /// Detaches the buffered utterance and resets state, or `None` if nothing is buffered.
    fn take(&mut self) -> Option<Utterance> {
        if self.utterance.is_empty() {
            return None;
        }
        let audio = std::mem::take(&mut self.utterance);
        let start = self.utterance_start.take().unwrap_or(Duration::ZERO);
        self.silence_accum = Duration::ZERO;
        Some(Utterance { audio, start })
    }
}

impl Segmenter for EnergySegmenter {
    fn push(
        &mut self,
        mono: &[f32],
        timestamp: Duration,
        frame_duration: Duration,
    ) -> Vec<Utterance> {
        if self.vad.is_speech(mono) {
            if self.utterance.is_empty() {
                self.utterance_start = Some(timestamp);
            }
            self.utterance.extend_from_slice(mono);
            self.silence_accum = Duration::ZERO;
            return Vec::new();
        }

        if !self.utterance.is_empty() {
            self.silence_accum += frame_duration;
            if self.silence_accum >= self.silence_hangover {
                return self.take().into_iter().collect();
            }
        }

        Vec::new()
    }

    fn flush(&mut self) -> Vec<Utterance> {
        self.take().into_iter().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vad::EnergyVad;

    const HANGOVER: Duration = Duration::from_millis(150);
    const FRAME: Duration = Duration::from_millis(100);

    fn segmenter() -> EnergySegmenter {
        EnergySegmenter::new(Box::new(EnergyVad::new(0.01)), HANGOVER)
    }

    /// 100 ms of speech (loud) at `t_ms`.
    fn speech(t_ms: u64) -> (Vec<f32>, Duration) {
        (vec![0.5; 1_600], Duration::from_millis(t_ms))
    }

    /// 100 ms of silence at `t_ms`.
    fn silence(t_ms: u64) -> (Vec<f32>, Duration) {
        (vec![0.0; 1_600], Duration::from_millis(t_ms))
    }

    #[test]
    fn buffers_speech_until_hangover_then_emits_one_utterance() {
        let mut seg = segmenter();

        let (s1, t1) = speech(0);
        assert!(seg.push(&s1, t1, FRAME).is_empty());
        let (s2, t2) = speech(100);
        assert!(seg.push(&s2, t2, FRAME).is_empty());

        // One silent frame (100 ms) is below the 150 ms hangover — keep buffering.
        let (q1, tq1) = silence(200);
        assert!(seg.push(&q1, tq1, FRAME).is_empty());

        // Second silent frame crosses the hangover — utterance closes.
        let (q2, tq2) = silence(300);
        let utts = seg.push(&q2, tq2, FRAME);
        assert_eq!(utts.len(), 1);
        assert_eq!(utts[0].start, Duration::from_millis(0));
        assert_eq!(utts[0].audio.len(), 3_200); // two 1 600-sample speech frames
    }

    #[test]
    fn start_timestamp_tracks_first_speech_frame() {
        let mut seg = segmenter();

        // Leading silence with no buffered speech yields nothing and sets no start.
        let (q, tq) = silence(0);
        assert!(seg.push(&q, tq, FRAME).is_empty());

        let (s, ts) = speech(500);
        assert!(seg.push(&s, ts, FRAME).is_empty());

        assert!(seg
            .push(&silence(600).0, Duration::from_millis(600), FRAME)
            .is_empty());
        let utts = seg.push(&silence(700).0, Duration::from_millis(700), FRAME);

        assert_eq!(utts.len(), 1);
        assert_eq!(utts[0].start, Duration::from_millis(500));
    }

    #[test]
    fn emits_two_separate_utterances() {
        let mut seg = segmenter();

        seg.push(&speech(0).0, Duration::from_millis(0), FRAME);
        seg.push(&silence(100).0, Duration::from_millis(100), FRAME);
        assert_eq!(
            seg.push(&silence(200).0, Duration::from_millis(200), FRAME)
                .len(),
            1
        );

        seg.push(&speech(300).0, Duration::from_millis(300), FRAME);
        seg.push(&silence(400).0, Duration::from_millis(400), FRAME);
        let utts = seg.push(&silence(500).0, Duration::from_millis(500), FRAME);
        assert_eq!(utts.len(), 1);
        assert_eq!(utts[0].start, Duration::from_millis(300));
    }

    #[test]
    fn flush_emits_buffered_speech_without_trailing_silence() {
        let mut seg = segmenter();
        seg.push(&speech(0).0, Duration::from_millis(0), FRAME);
        seg.push(&speech(100).0, Duration::from_millis(100), FRAME);

        let utts = seg.flush();
        assert_eq!(utts.len(), 1);
        assert_eq!(utts[0].start, Duration::ZERO);
        assert_eq!(utts[0].audio.len(), 3_200);

        // Nothing left after a flush.
        assert!(seg.flush().is_empty());
    }

    #[test]
    fn flush_when_idle_emits_nothing() {
        let mut seg = segmenter();
        assert!(seg.flush().is_empty());
        seg.push(&silence(0).0, Duration::ZERO, FRAME);
        assert!(seg.flush().is_empty());
    }
}
