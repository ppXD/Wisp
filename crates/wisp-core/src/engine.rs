//! The [`AsrEngine`] trait and its result types.

use crate::error::Result;
use crate::transcript::TranscriptSegment;

/// Static description of an ASR engine implementation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EngineInfo {
    /// Engine name (e.g. `"sherpa-onnx"`, `"mock"`).
    pub name: String,
    /// Whether the engine supports incremental/streaming decoding.
    pub streaming: bool,
}

/// The output of a single [`AsrEngine::transcribe`] call.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct TranscriptionResult {
    /// Segments recognized from the supplied audio.
    pub segments: Vec<TranscriptSegment>,
}

impl TranscriptionResult {
    /// A result with no segments.
    pub fn empty() -> Self {
        Self::default()
    }
}

/// How a whole-clip transcription should trade accuracy against speed.
///
/// Not `Debug`: it can carry a progress callback (`&dyn Fn`), which has no `Debug` impl.
#[derive(Clone, Copy)]
#[non_exhaustive]
pub struct ClipOptions<'a> {
    /// Emit per-segment timestamps (for subtitle export). Off lets the decoder focus on text.
    pub timestamps: bool,
    /// Use beam search — keeps several candidate sentences and picks the best overall (more
    /// accurate, slower) — instead of greedy decoding (faster, slightly less accurate).
    pub beam: bool,
    /// Optional context primer (names, jargon, domain terms) that biases the decoder toward the
    /// correct spellings. Empty = none.
    pub prompt: &'a str,
    /// Optional sink called with decoding progress as a percentage (0–100), so a caller can show a
    /// progress bar. Engines that can't report progress simply never call it.
    pub progress: Option<&'a dyn Fn(u8)>,
}

impl<'a> ClipOptions<'a> {
    /// Options with the given timestamp, beam-search, and biasing-prompt choices and no progress.
    pub fn new(timestamps: bool, beam: bool, prompt: &'a str) -> Self {
        Self {
            timestamps,
            beam,
            prompt,
            progress: None,
        }
    }

    /// Attaches a progress sink, called with `0..=100` as decoding advances.
    pub fn with_progress(mut self, progress: &'a dyn Fn(u8)) -> Self {
        self.progress = Some(progress);
        self
    }
}

impl Default for ClipOptions<'_> {
    fn default() -> Self {
        Self {
            timestamps: false,
            beam: true,
            prompt: "",
            progress: None,
        }
    }
}

/// A speech-to-text engine.
///
/// Engines consume 16 kHz mono `f32` audio and return recognized [`TranscriptSegment`]s. They
/// are swapped freely behind this trait (Whisper, Parakeet, a mock…).
pub trait AsrEngine: Send {
    /// Describes this engine.
    fn info(&self) -> EngineInfo;

    /// Transcribes a chunk of 16 kHz mono audio.
    fn transcribe(&mut self, audio: &[f32], sample_rate: u32) -> Result<TranscriptionResult>;

    /// Transcribes a whole clip (e.g. an entire file) at once, rather than a short streamed chunk.
    ///
    /// Engines with a native long-form path (full cross-window context) should override this for
    /// much better accuracy on files, honouring [`ClipOptions`] (beam vs greedy, timestamps). The
    /// default delegates to [`transcribe`](Self::transcribe).
    fn transcribe_clip(
        &mut self,
        audio: &[f32],
        sample_rate: u32,
        options: ClipOptions<'_>,
    ) -> Result<TranscriptionResult> {
        let _ = options;
        self.transcribe(audio, sample_rate)
    }

    /// Clears any accumulated streaming state (e.g. between utterances).
    ///
    /// The default is a no-op, which is correct for stateless engines.
    fn reset(&mut self) {}
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transcript::{AudioSourceKind, TranscriptSegment};
    use std::time::Duration;

    struct OneSegEngine;
    impl AsrEngine for OneSegEngine {
        fn info(&self) -> EngineInfo {
            EngineInfo {
                name: "one".to_owned(),
                streaming: false,
            }
        }
        fn transcribe(&mut self, _audio: &[f32], _sample_rate: u32) -> Result<TranscriptionResult> {
            Ok(TranscriptionResult {
                segments: vec![TranscriptSegment::new(
                    0,
                    "hi",
                    Duration::ZERO..Duration::from_millis(10),
                    AudioSourceKind::File,
                )],
            })
        }
    }

    #[test]
    fn empty_result_has_no_segments() {
        assert!(TranscriptionResult::empty().segments.is_empty());
    }

    #[test]
    fn clip_options_carry_prompt() {
        assert_eq!(ClipOptions::default().prompt, "", "default has no biasing");

        let opts = ClipOptions::new(true, false, "Acme, Inc.; kubectl");
        assert!(opts.timestamps);
        assert!(!opts.beam);
        assert_eq!(opts.prompt, "Acme, Inc.; kubectl");
    }

    #[test]
    fn clip_options_carry_a_progress_sink() {
        use std::cell::Cell;

        assert!(ClipOptions::default().progress.is_none());

        let seen = Cell::new(0u8);
        let sink = |p: u8| seen.set(p);
        let opts = ClipOptions::new(false, true, "").with_progress(&sink);
        (opts.progress.expect("sink set"))(42);
        assert_eq!(seen.get(), 42, "the attached sink receives progress");
    }

    #[test]
    fn transcribe_clip_defaults_to_transcribe() {
        let mut engine = OneSegEngine;
        let result = engine
            .transcribe_clip(&[0.0; 16], 16_000, ClipOptions::default())
            .unwrap();
        assert_eq!(result.segments.len(), 1);
        assert_eq!(result.segments[0].text, "hi");
    }
}
