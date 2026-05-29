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

use std::path::Path;
use std::time::Duration;

use sherpa_rs::sense_voice::{SenseVoiceConfig, SenseVoiceRecognizer};
use sherpa_rs::whisper::{WhisperConfig, WhisperRecognizer};
use wisp_core::engine::{AsrEngine, EngineInfo, TranscriptionResult};
use wisp_core::error::{Result, WispError};
use wisp_core::transcript::{AudioSourceKind, TranscriptSegment};

/// Wraps recognized `text` (spanning `audio` at `rate`) as a single-segment result; empty → empty.
fn to_result(text: &str, audio: &[f32], rate: u32) -> TranscriptionResult {
    let text = text.trim();
    if text.is_empty() {
        return TranscriptionResult::empty();
    }

    let secs = audio.len() as f32 / rate as f32;
    let segment = TranscriptSegment::new(
        0,
        text,
        Duration::ZERO..Duration::from_secs_f32(secs),
        AudioSourceKind::Microphone,
    );
    TranscriptionResult { segments: vec![segment] }
}

/// An [`AsrEngine`] backed by sherpa-onnx's SenseVoice offline recognizer.
pub struct SenseVoiceEngine {
    recognizer: SenseVoiceRecognizer,
}

impl SenseVoiceEngine {
    /// Loads a SenseVoice model (`model.onnx` or `model.int8.onnx`) and its `tokens.txt`.
    pub fn new(model: &Path, tokens: &Path) -> Result<Self> {
        let config = SenseVoiceConfig {
            model: model.to_string_lossy().into_owned(),
            tokens: tokens.to_string_lossy().into_owned(),
            language: "auto".to_owned(),
            use_itn: true,
            ..Default::default()
        };

        let recognizer = SenseVoiceRecognizer::new(config)
            .map_err(|e| WispError::Engine(format!("sense-voice init: {e}")))?;

        Ok(Self { recognizer })
    }
}

impl AsrEngine for SenseVoiceEngine {
    fn info(&self) -> EngineInfo {
        EngineInfo { name: "sherpa-sense-voice".to_owned(), streaming: false }
    }

    fn transcribe(&mut self, audio: &[f32], sample_rate: u32) -> Result<TranscriptionResult> {
        let rate = if sample_rate == 0 { 16_000 } else { sample_rate };
        let result = self.recognizer.transcribe(rate, audio);
        Ok(to_result(&result.text, audio, rate))
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
            ..Default::default()
        };

        let recognizer = WhisperRecognizer::new(config)
            .map_err(|e| WispError::Engine(format!("whisper init: {e}")))?;

        Ok(Self { recognizer })
    }
}

impl AsrEngine for WhisperEngine {
    fn info(&self) -> EngineInfo {
        EngineInfo { name: "sherpa-whisper".to_owned(), streaming: false }
    }

    fn transcribe(&mut self, audio: &[f32], sample_rate: u32) -> Result<TranscriptionResult> {
        let rate = if sample_rate == 0 { 16_000 } else { sample_rate };
        let result = self.recognizer.transcribe(rate, audio);
        Ok(to_result(&result.text, audio, rate))
    }
}
