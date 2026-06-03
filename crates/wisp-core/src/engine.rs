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

    /// Whether [`transcribe_clip`](Self::transcribe_clip) drives the [`ClipOptions::progress`] sink
    /// as it decodes. Engines with a native long-form path (whisper.cpp) report; the default
    /// one-shot path can't, so a caller that wants a progress bar must window the clip itself.
    fn reports_clip_progress(&self) -> bool {
        false
    }

    /// Configures the streaming [`transcribe`](Self::transcribe) path: a biasing `prompt` (names,
    /// jargon) and whether to use beam search (more accurate, slower) instead of greedy decoding.
    ///
    /// The default is a no-op, correct for engines that don't expose these knobs.
    fn configure_streaming(&mut self, prompt: &str, beam: bool) {
        let _ = (prompt, beam);
    }

    /// Supplies recent finalized transcript text to bias the next [`transcribe`](Self::transcribe)
    /// toward consistent spellings, names, and terminology across a live session.
    ///
    /// This is *text* context (folded into the biasing prompt), not carried decoder state, so it
    /// improves cross-utterance consistency without the hallucination/repetition drift that full
    /// context-carrying causes. The default is a no-op, correct for engines that take no prompt.
    fn set_context(&mut self, recent_text: &str) {
        let _ = recent_text;
    }

    /// Clears any accumulated streaming state (e.g. between utterances).
    ///
    /// The default is a no-op, which is correct for stateless engines.
    fn reset(&mut self) {}
}

/// One incremental result from a [`StreamingAsrEngine`].
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct StreamingResult {
    /// The current hypothesis for the in-progress utterance (grows as more audio arrives).
    pub text: String,
    /// True when an utterance boundary was just reached: treat `text` as final. The engine resets
    /// internally, so the next call begins a fresh utterance.
    pub is_endpoint: bool,
    /// An optional parallel rendering of the same utterance from a secondary pass — e.g. a cloud
    /// model session's instruction-steered output (a translation) running alongside the verbatim
    /// transcript in `text`. `None` for engines that produce a single stream.
    pub aux: Option<String>,
}

impl StreamingResult {
    /// A single-stream result (no `aux`) — the common case for engines with one output.
    pub fn new(text: impl Into<String>, is_endpoint: bool) -> Self {
        Self {
            text: text.into(),
            is_endpoint,
            aux: None,
        }
    }
}

/// A *streaming* speech-to-text engine.
///
/// Unlike [`AsrEngine`] — which transcribes a whole (VAD-segmented) utterance at once — a streaming
/// engine consumes audio incrementally and emits a growing hypothesis within a couple hundred
/// milliseconds, detecting utterance endpoints itself. That removes the segmentation latency, so the
/// pipeline drives it directly, chunk by chunk, instead of waiting for the speaker to pause.
pub trait StreamingAsrEngine: Send {
    /// Feeds a chunk of mono `f32` audio at [`input_sample_rate`](Self::input_sample_rate) and returns
    /// the updated hypothesis, with `is_endpoint` set when an utterance boundary was reached (the
    /// engine then resets for the next utterance).
    fn accept_waveform(&mut self, sample_rate: u32, samples: &[f32]) -> StreamingResult;

    /// The mono sample rate this engine wants its audio at. The default is 16 kHz — the on-device ASR
    /// rate; the pipeline resamples each frame to this before calling
    /// [`accept_waveform`](Self::accept_waveform). A cloud engine that accepts richer audio overrides
    /// it (e.g. 24 kHz) so the pipeline feeds native high-rate audio instead of 16 kHz-upsampled.
    fn input_sample_rate(&self) -> u32 {
        16_000
    }

    /// Drops any in-progress hypothesis and starts a fresh utterance (e.g. on session restart).
    fn reset(&mut self);

    /// Injects an authoritative text item into the session's context — e.g. the app's diarized,
    /// speaker-attributed transcript final — for engines that can be steered with text alongside audio.
    /// Default: no-op (most engines only consume audio).
    fn inject_text(&mut self, _text: &str) {}

    /// Requests the engine produce a response now, for instruction-steered sessions whose replies are
    /// triggered explicitly (a controlled-cadence assist) rather than automatically per turn. Default:
    /// no-op.
    fn request_response(&mut self) {}
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

    #[test]
    fn configure_streaming_defaults_to_a_harmless_noop() {
        // Engines that don't support streaming knobs keep working after the call.
        let mut engine = OneSegEngine;
        engine.configure_streaming("Acme, Inc.", true);
        let result = engine.transcribe(&[0.0; 16], 16_000).unwrap();
        assert_eq!(result.segments.len(), 1);
    }

    #[test]
    fn streaming_result_new_has_no_aux_stream() {
        let r = StreamingResult::new("hello", true);
        assert_eq!(r.text, "hello");
        assert!(r.is_endpoint);
        assert!(r.aux.is_none(), "single-stream result carries no aux");
        assert_eq!(StreamingResult::default(), StreamingResult::new("", false));
    }

    #[test]
    fn set_context_defaults_to_a_harmless_noop() {
        // Engines that take no biasing prompt ignore rolling context and keep working.
        let mut engine = OneSegEngine;
        engine.set_context("we discussed Kubernetes and the Acme account");
        let result = engine.transcribe(&[0.0; 16], 16_000).unwrap();
        assert_eq!(result.segments.len(), 1);
    }
}
