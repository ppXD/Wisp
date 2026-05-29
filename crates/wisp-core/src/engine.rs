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
    /// much better accuracy on files. `with_timestamps` requests per-segment timings for subtitle
    /// export, at a small accuracy cost — callers that only need text pass `false` to let the
    /// engine spend everything on words. The default delegates to [`transcribe`](Self::transcribe).
    fn transcribe_clip(
        &mut self,
        audio: &[f32],
        sample_rate: u32,
        with_timestamps: bool,
    ) -> Result<TranscriptionResult> {
        let _ = with_timestamps;
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

    #[test]
    fn empty_result_has_no_segments() {
        assert!(TranscriptionResult::empty().segments.is_empty());
    }
}
