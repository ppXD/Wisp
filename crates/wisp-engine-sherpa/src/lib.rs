//! sherpa-onnx ASR engine for Wisp.
//!
//! Wraps sherpa-onnx's **SenseVoice** offline recognizer — a single-file multilingual model
//! (Chinese, English, Japanese, Korean, Cantonese) — behind [`wisp_core::AsrEngine`], so it
//! drops into the pipeline in place of any other engine.

use std::path::Path;
use std::time::Duration;

use sherpa_rs::sense_voice::{SenseVoiceConfig, SenseVoiceRecognizer};
use wisp_core::engine::{AsrEngine, EngineInfo, TranscriptionResult};
use wisp_core::error::{Result, WispError};
use wisp_core::transcript::{AudioSourceKind, TranscriptSegment};

/// An [`AsrEngine`] backed by sherpa-onnx's SenseVoice offline recognizer.
///
/// SenseVoice transcribes a whole utterance at once; the pipeline supplies VAD-segmented
/// utterances and offsets the returned segment's timestamps.
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

        let text = result.text.trim().to_owned();
        if text.is_empty() {
            return Ok(TranscriptionResult::empty());
        }

        let secs = audio.len() as f32 / rate as f32;
        let segment = TranscriptSegment::new(
            0,
            text,
            Duration::ZERO..Duration::from_secs_f32(secs),
            AudioSourceKind::Microphone,
        );
        Ok(TranscriptionResult { segments: vec![segment] })
    }
}
