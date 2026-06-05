//! sherpa-onnx ASR engines for Wisp.
//!
//! Wraps sherpa-onnx offline recognizers behind [`wisp_core::AsrEngine`] so any of them drops into
//! the pipeline interchangeably:
//! - [`SenseVoiceEngine`] — single-file multilingual model (Chinese, English, Japanese, Korean,
//!   Cantonese), fast and light.
//! - [`WhisperEngine`] — OpenAI Whisper (encoder + decoder), ~99 languages, most accurate but
//!   larger and slower.
//!
//! Both transcribe a whole utterance at once; the pipeline supplies VAD-segmented utterances and
//! offsets the returned segment's timestamps.
//!
//! [`SileroSegmenter`] adapts sherpa-onnx's Silero neural VAD to the pipeline's segmentation trait.

mod denoise;
mod diarize;
mod live_diarize;
mod silero;
mod streaming;
pub use denoise::GtcrnDenoiser;
pub use diarize::SherpaDiarizer;
pub use live_diarize::SherpaLiveDiarizer;
pub use silero::SileroSegmenter;
pub use streaming::StreamingTransducerEngine;

use std::path::Path;
use std::time::Duration;

use sherpa_rs::paraformer::{ParaformerConfig, ParaformerRecognizer};
use sherpa_rs::sense_voice::{SenseVoiceConfig, SenseVoiceRecognizer};
use sherpa_rs::transducer::{TransducerConfig, TransducerRecognizer};
use sherpa_rs::whisper::{WhisperConfig, WhisperRecognizer};
use wisp_core::align::words_from_token_timestamps;
use wisp_core::engine::{AsrEngine, EngineInfo, TranscriptionResult};
use wisp_core::error::{Result, WispError};
use wisp_core::transcript::{AudioSourceKind, TranscriptSegment};

/// CPU worker threads for the sherpa engines — both SenseVoice and Whisper run on the CPU, and
/// sherpa-rs otherwise defaults to 1 thread, which is far from real-time. Scales to the machine's
/// physical cores via the shared policy (overridable with `WISP_ASR_THREADS`).
fn asr_threads() -> i32 {
    let n = wisp_core::perf::asr_threads(
        std::env::var(wisp_core::perf::ASR_THREADS_ENV).ok(),
        num_cpus::get_physical(),
    );
    i32::try_from(n).unwrap_or(i32::MAX)
}

/// Env var selecting the ONNX Runtime execution provider for every sherpa engine — e.g. `cpu`,
/// `cuda`, `directml`, `coreml`. Unset or empty leaves sherpa-rs on its built-in default (CPU). The
/// app sets it once at startup from the detected host accelerator; an operator can override it to
/// force (or disable) a GPU provider without a rebuild.
pub const EXECUTION_PROVIDER_ENV: &str = "WISP_ONNX_PROVIDER";

/// The configured execution provider for the sherpa ONNX engines, or `None` to leave sherpa-rs on its
/// default. Read per config build, so a process can be pointed at a different provider via the
/// environment without recompiling.
fn execution_provider() -> Option<String> {
    parse_provider(std::env::var(EXECUTION_PROVIDER_ENV).ok())
}

/// Normalizes a raw [`EXECUTION_PROVIDER_ENV`] value: trims surrounding space and treats an
/// empty/whitespace value as unset (`None`). Pure, so it's tested without touching the process env.
fn parse_provider(raw: Option<String>) -> Option<String> {
    let trimmed = raw?.trim().to_owned();
    (!trimmed.is_empty()).then_some(trimmed)
}

/// Whisper emits non-speech annotations like `[BLANK_AUDIO]` or `(speaking in foreign language)` on
/// silence/noise. Treat any segment that is purely such an annotation as empty so it isn't shown.
fn strip_whisper_annotation(raw: &str) -> &str {
    let trimmed = raw.trim();
    if trimmed.starts_with('[') || trimmed.starts_with('(') {
        ""
    } else {
        trimmed
    }
}

/// Wraps recognized `text` as a single whole-clip segment spanning `clip_secs`. When the recognizer
/// also exposes per-token `tokens` + `timestamps`, they're grouped into the segment's
/// [`words`](TranscriptSegment::words) for word-level speaker attribution and subtitle timing; a
/// recognizer that returns text only (empty `tokens`) yields a word-less segment. Empty text → empty.
fn to_result(
    text: &str,
    tokens: &[String],
    timestamps: &[f32],
    clip_secs: f32,
) -> TranscriptionResult {
    let text = text.trim();
    if text.is_empty() {
        return TranscriptionResult::empty();
    }

    let mut segment = TranscriptSegment::new(
        0,
        text,
        Duration::ZERO..Duration::from_secs_f32(clip_secs.max(0.0)),
        AudioSourceKind::Microphone,
    );
    segment.words = words_from_token_timestamps(tokens, timestamps, clip_secs);

    TranscriptionResult {
        segments: vec![segment],
    }
}

/// An [`AsrEngine`] backed by sherpa-onnx's SenseVoice offline recognizer.
pub struct SenseVoiceEngine {
    recognizer: SenseVoiceRecognizer,
}

impl SenseVoiceEngine {
    /// Loads a SenseVoice model (`model.onnx` or `model.int8.onnx`) and its `tokens.txt`.
    ///
    /// `language` is a SenseVoice code (`zh`/`en`/`ja`/`ko`/`yue`); an empty string means `auto`.
    pub fn new(model: &Path, tokens: &Path, language: &str) -> Result<Self> {
        let language = if language.is_empty() {
            "auto"
        } else {
            language
        };
        let config = SenseVoiceConfig {
            model: model.to_string_lossy().into_owned(),
            tokens: tokens.to_string_lossy().into_owned(),
            language: language.to_owned(),
            use_itn: true,
            // Without this sherpa-rs runs SenseVoice single-threaded — the biggest cross-platform
            // speed loss, since SenseVoice is the default engine off macOS.
            num_threads: Some(asr_threads()),
            provider: execution_provider(),
            ..Default::default()
        };

        let recognizer = SenseVoiceRecognizer::new(config)
            .map_err(|e| WispError::Engine(format!("sense-voice init: {e}")))?;

        Ok(Self { recognizer })
    }
}

impl AsrEngine for SenseVoiceEngine {
    fn info(&self) -> EngineInfo {
        EngineInfo {
            name: "sherpa-sense-voice".to_owned(),
            streaming: false,
        }
    }

    fn transcribe(&mut self, audio: &[f32], sample_rate: u32) -> Result<TranscriptionResult> {
        let rate = if sample_rate == 0 {
            16_000
        } else {
            sample_rate
        };
        let result = self.recognizer.transcribe(rate, audio);
        let secs = audio.len() as f32 / rate as f32;
        Ok(to_result(
            &result.text,
            &result.tokens,
            &result.timestamps,
            secs,
        ))
    }
}

/// An [`AsrEngine`] backed by sherpa-onnx's Whisper offline recognizer.
pub struct WhisperEngine {
    recognizer: WhisperRecognizer,
}

impl WhisperEngine {
    /// Loads a Whisper model from its `encoder`/`decoder` ONNX files and `tokens.txt`.
    ///
    /// `language` is a Whisper language code (e.g. `"yue"` for Cantonese); an empty string lets
    /// Whisper auto-detect.
    pub fn new(encoder: &Path, decoder: &Path, tokens: &Path, language: &str) -> Result<Self> {
        let config = WhisperConfig {
            encoder: encoder.to_string_lossy().into_owned(),
            decoder: decoder.to_string_lossy().into_owned(),
            tokens: tokens.to_string_lossy().into_owned(),
            language: language.to_owned(),
            num_threads: Some(asr_threads()),
            provider: execution_provider(),
            ..Default::default()
        };

        let recognizer = WhisperRecognizer::new(config)
            .map_err(|e| WispError::Engine(format!("whisper init: {e}")))?;

        Ok(Self { recognizer })
    }
}

impl AsrEngine for WhisperEngine {
    fn info(&self) -> EngineInfo {
        EngineInfo {
            name: "sherpa-whisper".to_owned(),
            streaming: false,
        }
    }

    fn transcribe(&mut self, audio: &[f32], sample_rate: u32) -> Result<TranscriptionResult> {
        let rate = if sample_rate == 0 {
            16_000
        } else {
            sample_rate
        };
        let result = self.recognizer.transcribe(rate, audio);
        let secs = audio.len() as f32 / rate as f32;
        Ok(to_result(
            strip_whisper_annotation(&result.text),
            &result.tokens,
            &result.timestamps,
            secs,
        ))
    }
}

/// An [`AsrEngine`] backed by sherpa-onnx's Paraformer offline recognizer (FunASR, zh + en) — a
/// non-autoregressive model that's fast on CPU. Single `model.onnx` + `tokens.txt`, like SenseVoice.
pub struct ParaformerEngine {
    recognizer: ParaformerRecognizer,
}

impl ParaformerEngine {
    /// Loads a Paraformer model (`model.onnx` or `model.int8.onnx`) and its `tokens.txt`.
    pub fn new(model: &Path, tokens: &Path) -> Result<Self> {
        let config = ParaformerConfig {
            model: model.to_string_lossy().into_owned(),
            tokens: tokens.to_string_lossy().into_owned(),
            num_threads: Some(asr_threads()),
            provider: execution_provider(),
            ..Default::default()
        };

        let recognizer = ParaformerRecognizer::new(config)
            .map_err(|e| WispError::Engine(format!("paraformer init: {e}")))?;

        Ok(Self { recognizer })
    }
}

impl AsrEngine for ParaformerEngine {
    fn info(&self) -> EngineInfo {
        EngineInfo {
            name: "sherpa-paraformer".to_owned(),
            streaming: false,
        }
    }

    fn transcribe(&mut self, audio: &[f32], sample_rate: u32) -> Result<TranscriptionResult> {
        let rate = if sample_rate == 0 {
            16_000
        } else {
            sample_rate
        };
        let result = self.recognizer.transcribe(rate, audio);
        let secs = audio.len() as f32 / rate as f32;
        Ok(to_result(
            &result.text,
            &result.tokens,
            &result.timestamps,
            secs,
        ))
    }
}

/// An [`AsrEngine`] backed by sherpa-onnx's offline transducer — NVIDIA NeMo Parakeet
/// (encoder/decoder/joiner), English, state-of-the-art accuracy.
pub struct ParakeetEngine {
    recognizer: TransducerRecognizer,
}

impl ParakeetEngine {
    /// Loads a NeMo Parakeet transducer from its `encoder`/`decoder`/`joiner` ONNX files + `tokens.txt`.
    pub fn new(encoder: &Path, decoder: &Path, joiner: &Path, tokens: &Path) -> Result<Self> {
        let config = TransducerConfig {
            encoder: encoder.to_string_lossy().into_owned(),
            decoder: decoder.to_string_lossy().into_owned(),
            joiner: joiner.to_string_lossy().into_owned(),
            tokens: tokens.to_string_lossy().into_owned(),
            // A NeMo-exported transducer decoded greedily — the standard sherpa-onnx setup for Parakeet.
            model_type: "nemo_transducer".to_owned(),
            decoding_method: "greedy_search".to_owned(),
            num_threads: asr_threads(),
            provider: execution_provider(),
            ..Default::default()
        };

        let recognizer = TransducerRecognizer::new(config)
            .map_err(|e| WispError::Engine(format!("parakeet init: {e}")))?;

        Ok(Self { recognizer })
    }
}

impl AsrEngine for ParakeetEngine {
    fn info(&self) -> EngineInfo {
        EngineInfo {
            name: "sherpa-parakeet".to_owned(),
            streaming: false,
        }
    }

    fn transcribe(&mut self, audio: &[f32], sample_rate: u32) -> Result<TranscriptionResult> {
        let rate = if sample_rate == 0 {
            16_000
        } else {
            sample_rate
        };
        let text = self.recognizer.transcribe(rate, audio);
        let secs = audio.len() as f32 / rate as f32;
        // The transducer wrapper returns text only (no per-token timings), so the segment is word-less.
        Ok(to_result(&text, &[], &[], secs))
    }
}

#[cfg(test)]
mod tests {
    use super::{parse_provider, strip_whisper_annotation, to_result, EXECUTION_PROVIDER_ENV};

    #[test]
    fn drops_whisper_non_speech_annotations() {
        assert_eq!(strip_whisper_annotation("[BLANK_AUDIO]"), "");
        assert_eq!(
            strip_whisper_annotation("  (speaking in foreign language) "),
            ""
        );
        assert_eq!(strip_whisper_annotation("[BLANK"), "");
        assert_eq!(strip_whisper_annotation("聽不聽到"), "聽不聽到");
        assert_eq!(strip_whisper_annotation("  Hello, "), "Hello,");
    }

    #[test]
    fn to_result_attaches_word_timings_from_tokens() {
        // A recognizer that exposes per-token timings (SenseVoice/Whisper/Paraformer) yields a
        // segment whose words carry real spans for diarization and subtitle timing.
        let tokens = vec!["\u{2581}hello".to_owned(), "\u{2581}world".to_owned()];
        let timestamps = vec![0.0_f32, 0.5];

        let result = to_result("hello world", &tokens, &timestamps, 1.0);

        assert_eq!(result.segments.len(), 1);
        let segment = &result.segments[0];
        assert_eq!(segment.text, "hello world");
        assert_eq!(segment.words.len(), 2);
        assert_eq!(segment.words[0].text.trim(), "hello");
        assert_eq!(segment.words[1].text.trim(), "world");
    }

    #[test]
    fn to_result_without_token_timings_is_word_less() {
        // The transducer wrapper returns text only; the segment carries text but no words, exactly
        // as before this change.
        let result = to_result("hello", &[], &[], 1.0);

        assert_eq!(result.segments.len(), 1);
        assert!(result.segments[0].words.is_empty());
    }

    #[test]
    fn to_result_empty_text_is_empty() {
        assert!(to_result("   ", &[], &[], 1.0).segments.is_empty());
    }

    /// Renaming this env var silently breaks any operator who pinned a GPU execution provider.
    #[test]
    fn execution_provider_env_name_is_pinned() {
        assert_eq!(EXECUTION_PROVIDER_ENV, "WISP_ONNX_PROVIDER");
    }

    #[test]
    fn parse_provider_trims_and_treats_blank_as_unset() {
        assert_eq!(
            parse_provider(Some("directml".to_owned())),
            Some("directml".to_owned())
        );
        assert_eq!(
            parse_provider(Some("  cuda  ".to_owned())),
            Some("cuda".to_owned())
        );
        assert_eq!(parse_provider(Some("   ".to_owned())), None);
        assert_eq!(parse_provider(Some(String::new())), None);
        assert_eq!(parse_provider(None), None);
    }
}
