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
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct ClipOptions {
    /// Emit per-segment timestamps (for subtitle export). Off lets the decoder focus on text.
    pub timestamps: bool,
    /// Use beam search — keeps several candidate sentences and picks the best overall (more
    /// accurate, slower) — instead of greedy decoding (faster, slightly less accurate).
    pub beam: bool,
}

impl ClipOptions {
    /// Options with the given timestamp and beam-search choices.
    pub fn new(timestamps: bool, beam: bool) -> Self {
        Self { timestamps, beam }
    }
}

impl Default for ClipOptions {
    fn default() -> Self {
        Self {
            timestamps: false,
            beam: true,
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
        options: ClipOptions,
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
    fn transcribe_clip_defaults_to_transcribe() {
        let mut engine = OneSegEngine;
        let result = engine
            .transcribe_clip(&[0.0; 16], 16_000, ClipOptions::default())
            .unwrap();
        assert_eq!(result.segments.len(), 1);
        assert_eq!(result.segments[0].text, "hi");
    }
}
