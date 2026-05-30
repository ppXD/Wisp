//! Incremental (live) speaker diarization via sherpa-onnx speaker embeddings.
//!
//! Implements [`wisp_core::diarize::Diarizer`]: for each utterance it computes a speaker embedding
//! and matches it against the speakers seen so far (sherpa's embedding manager does the nearest-
//! match search), reusing the closest speaker above a similarity threshold or registering a brand-
//! new one. Reuses the `embedding.onnx` from the downloadable diarization pack — no extra model.

use std::path::Path;

use sherpa_rs::embedding_manager::EmbeddingManager;
use sherpa_rs::speaker_id::{EmbeddingExtractor, ExtractorConfig, DEFAULT_SIMILARITY_THRESHOLD};
use wisp_core::diarize::Diarizer;
use wisp_core::error::{Result, WispError};
use wisp_core::transcript::SpeakerId;

/// Online speaker labelling for the live pipeline: one stable [`SpeakerId`] per recurring voice.
pub struct SherpaLiveDiarizer {
    extractor: EmbeddingExtractor,
    manager: EmbeddingManager,
    threshold: f32,
    next_speaker: u32,
}

impl SherpaLiveDiarizer {
    /// Builds a live diarizer from a speaker-embedding model (`embedding.onnx` from the diarization
    /// pack).
    pub fn new(embedding_model: &Path) -> Result<Self> {
        let extractor = EmbeddingExtractor::new(ExtractorConfig {
            model: embedding_model.to_string_lossy().into_owned(),
            provider: None,
            num_threads: None,
            debug: false,
        })
        .map_err(|e| WispError::Engine(format!("live diarizer: {e}")))?;
        let manager = EmbeddingManager::new(extractor.embedding_size as i32);
        Ok(Self {
            extractor,
            manager,
            threshold: DEFAULT_SIMILARITY_THRESHOLD,
            next_speaker: 0,
        })
    }
}

impl Diarizer for SherpaLiveDiarizer {
    fn identify(&mut self, audio: &[f32], sample_rate: u32) -> Result<SpeakerId> {
        let mut embedding = self
            .extractor
            .compute_speaker_embedding(audio.to_vec(), sample_rate)
            .map_err(|e| WispError::Engine(format!("speaker embedding: {e}")))?;

        // Close enough to a known speaker → reuse that id (speakers are registered under their
        // numeric id); otherwise this is a new voice.
        if let Some(id) = self
            .manager
            .search(&embedding, self.threshold)
            .and_then(|name| name.parse::<u32>().ok())
        {
            return Ok(SpeakerId(id));
        }

        let id = self.next_speaker;
        self.next_speaker += 1;
        let _ = self.manager.add(id.to_string(), &mut embedding);
        Ok(SpeakerId(id))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    /// Real sherpa embedding inference. Skip-guarded behind an env var so `cargo test` is clean
    /// where the model isn't available:
    ///   WISP_LIVE_DIARIZE_TEST_EMBEDDING=/path/to/embedding.onnx
    #[test]
    fn identical_audio_maps_to_the_same_speaker() {
        let Some(model) = std::env::var_os("WISP_LIVE_DIARIZE_TEST_EMBEDDING") else {
            return;
        };
        let model = PathBuf::from(model);
        if !model.exists() {
            return;
        }

        let mut diarizer = SherpaLiveDiarizer::new(&model).expect("load embedding model");
        let audio = vec![0.05f32; 16_000]; // 1 s at 16 kHz
        let first = diarizer.identify(&audio, 16_000).expect("identify");
        let again = diarizer.identify(&audio, 16_000).expect("identify");

        assert_eq!(first, again, "identical audio must map to the same speaker");
    }
}
