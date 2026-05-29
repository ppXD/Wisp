//! Voice-activity segmentation: turns a frame stream into complete utterances.
//!
//! Pure, engine-free logic so it can run on the real-time capture thread without ever blocking on
//! transcription. Feed 16 kHz mono frames via [`Segmenter::push`]; when trailing silence reaches
//! the hangover, it hands back the buffered [`Utterance`].

use std::time::Duration;

use crate::vad::Vad;

/// A complete spoken utterance: 16 kHz mono PCM plus the timestamp of its first speech frame.
///
/// Produced by the [`Segmenter`] and consumed by the transcriber — decoupling the (fast) capture
/// path from the (slow) ASR engine.
#[derive(Debug, Clone, PartialEq)]
pub struct Utterance {
    /// 16 kHz mono samples spanning the utterance.
    pub audio: Vec<f32>,
    /// Capture timestamp of the utterance's first speech frame, used to offset segment times.
    pub start: Duration,
}

/// Accumulates speech frames into utterances, splitting on trailing silence.
///
/// Holds no ASR engine — emitting an [`Utterance`] is just buffer book-keeping, so this can be
/// driven from the capture thread at real-time without stalling on the engine.
pub struct Segmenter {
    vad: Box<dyn Vad>,
    silence_hangover: Duration,
    utterance: Vec<f32>,
    utterance_start: Option<Duration>,
    silence_accum: Duration,
}

impl Segmenter {
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

    /// Feeds one 16 kHz mono frame (`mono`) stamped at `timestamp` and lasting `frame_duration`.
    ///
    /// Returns `Some(utterance)` exactly when this frame's trailing silence completes the buffered
    /// utterance; otherwise `None` (still buffering, or idle).
    pub fn push(
        &mut self,
        mono: &[f32],
        timestamp: Duration,
        frame_duration: Duration,
    ) -> Option<Utterance> {
        if self.vad.is_speech(mono) {
            if self.utterance.is_empty() {
                self.utterance_start = Some(timestamp);
            }
            self.utterance.extend_from_slice(mono);
            self.silence_accum = Duration::ZERO;
            return None;
        }

        if !self.utterance.is_empty() {
            self.silence_accum += frame_duration;
            if self.silence_accum >= self.silence_hangover {
                return self.take();
            }
        }

        None
    }

    /// Emits any buffered utterance unconditionally — for end-of-stream, where the last utterance
    /// may not have been followed by enough trailing silence to close on its own.
    pub fn flush(&mut self) -> Option<Utterance> {
        self.take()
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vad::EnergyVad;

    const HANGOVER: Duration = Duration::from_millis(150);
    const FRAME: Duration = Duration::from_millis(100);

    fn segmenter() -> Segmenter {
        Segmenter::new(Box::new(EnergyVad::new(0.01)), HANGOVER)
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
        assert_eq!(seg.push(&s1, t1, FRAME), None);
        let (s2, t2) = speech(100);
        assert_eq!(seg.push(&s2, t2, FRAME), None);

        // One silent frame (100 ms) is below the 150 ms hangover — keep buffering.
        let (q1, tq1) = silence(200);
        assert_eq!(seg.push(&q1, tq1, FRAME), None);

        // Second silent frame crosses the hangover — utterance closes.
        let (q2, tq2) = silence(300);
        let utt = seg.push(&q2, tq2, FRAME).expect("utterance should close");
        assert_eq!(utt.start, Duration::from_millis(0));
        assert_eq!(utt.audio.len(), 3_200); // two 1 600-sample speech frames
    }

    #[test]
    fn start_timestamp_tracks_first_speech_frame() {
        let mut seg = segmenter();

        // Leading silence with no buffered speech yields nothing and sets no start.
        let (q, tq) = silence(0);
        assert_eq!(seg.push(&q, tq, FRAME), None);

        let (s, ts) = speech(500);
        assert_eq!(seg.push(&s, ts, FRAME), None);

        let (q1, _) = silence(600);
        assert_eq!(seg.push(&q1, Duration::from_millis(600), FRAME), None);
        let (q2, _) = silence(700);
        let utt = seg.push(&q2, Duration::from_millis(700), FRAME).unwrap();

        assert_eq!(utt.start, Duration::from_millis(500));
    }

    #[test]
    fn emits_two_separate_utterances() {
        let mut seg = segmenter();

        seg.push(&speech(0).0, Duration::from_millis(0), FRAME);
        seg.push(&silence(100).0, Duration::from_millis(100), FRAME);
        let first = seg.push(&silence(200).0, Duration::from_millis(200), FRAME);
        assert!(first.is_some(), "first utterance closes after hangover");

        seg.push(&speech(300).0, Duration::from_millis(300), FRAME);
        seg.push(&silence(400).0, Duration::from_millis(400), FRAME);
        let second = seg.push(&silence(500).0, Duration::from_millis(500), FRAME);
        let utt = second.expect("second utterance closes");
        assert_eq!(utt.start, Duration::from_millis(300));
    }

    #[test]
    fn flush_emits_buffered_speech_without_trailing_silence() {
        let mut seg = segmenter();
        seg.push(&speech(0).0, Duration::from_millis(0), FRAME);
        seg.push(&speech(100).0, Duration::from_millis(100), FRAME);

        let utt = seg.flush().expect("buffered speech should flush");
        assert_eq!(utt.start, Duration::ZERO);
        assert_eq!(utt.audio.len(), 3_200);

        // Nothing left after a flush.
        assert_eq!(seg.flush(), None);
    }

    #[test]
    fn flush_when_idle_emits_nothing() {
        let mut seg = segmenter();
        assert_eq!(seg.flush(), None);
        seg.push(&silence(0).0, Duration::ZERO, FRAME);
        assert_eq!(seg.flush(), None);
    }
}
