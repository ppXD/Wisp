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
    min_new_speaker_secs: f32,
    next_speaker: u32,
}

/// Default seconds of speech an utterance must carry before it may register a *new* speaker. A
/// speaker-embedding model needs a couple of seconds to be stable; shorter clips (back-channels like
/// "系") embed unreliably and would otherwise each spawn a spurious speaker.
const DEFAULT_MIN_NEW_SPEAKER_SECS: f32 = 1.5;

/// Similarity floor that always matches the nearest existing speaker (cosine ∈ [-1, 1]): used to
/// attach a too-short utterance to the closest known voice instead of minting a new one.
const MATCH_NEAREST: f32 = -1.0;

impl SherpaLiveDiarizer {
    /// Env override (seconds) for the minimum utterance length that may register a new speaker —
    /// raise it if a session over-splits one voice into many, lower it to pick up brief speakers
    /// sooner. Defaults to [`DEFAULT_MIN_NEW_SPEAKER_SECS`].
    pub const MIN_NEW_SPEAKER_SECS_ENV: &'static str = "WISP_LIVE_DIARIZE_MIN_NEW_SPEAKER_SECS";

    /// Env override for the cosine-similarity threshold above which an utterance reuses a known
    /// speaker. Lower merges similar voices (fewer speakers); higher splits more readily. Defaults to
    /// sherpa's [`DEFAULT_SIMILARITY_THRESHOLD`] (0.5).
    pub const SIMILARITY_THRESHOLD_ENV: &'static str = "WISP_LIVE_DIARIZE_SIMILARITY";

    /// Builds a live diarizer from a speaker-embedding model (`embedding.onnx` from the diarization
    /// pack).
    pub fn new(embedding_model: &Path) -> Result<Self> {
        let extractor = EmbeddingExtractor::new(ExtractorConfig {
            model: embedding_model.to_string_lossy().into_owned(),
            provider: None,
            // Scale the embedding pass to the machine's physical cores (shared ASR thread policy).
            // Left single-threaded it serialized every live utterance onto one core, and that extra
            // per-utterance latency stalled real-time transcription whenever speaker labels were on.
            num_threads: Some(crate::asr_threads() as usize),
            debug: false,
        })
        .map_err(|e| WispError::Engine(format!("live diarizer: {e}")))?;

        let manager = EmbeddingManager::new(extractor.embedding_size as i32);

        Ok(Self {
            extractor,
            manager,
            threshold: env_f32(Self::SIMILARITY_THRESHOLD_ENV, DEFAULT_SIMILARITY_THRESHOLD),
            min_new_speaker_secs: env_f32(
                Self::MIN_NEW_SPEAKER_SECS_ENV,
                DEFAULT_MIN_NEW_SPEAKER_SECS,
            )
            .max(0.0),
            next_speaker: 0,
        })
    }
}

/// Reads a finite `f32` from environment variable `var`, falling back to `default` when it is unset
/// or unparseable.
fn env_f32(var: &str, default: f32) -> f32 {
    std::env::var(var)
        .ok()
        .and_then(|v| v.parse::<f32>().ok())
        .filter(|v| v.is_finite())
        .unwrap_or(default)
}

impl Diarizer for SherpaLiveDiarizer {
    fn identify(&mut self, audio: &[f32], sample_rate: u32) -> Result<SpeakerId> {
        let mut embedding = self
            .extractor
            .compute_speaker_embedding(audio.to_vec(), sample_rate)
            .map_err(|e| WispError::Engine(format!("speaker embedding: {e}")))?;

        // Close enough to a known speaker → reuse that id (speakers are registered under their
        // numeric id).
        if let Some(id) = self
            .manager
            .search(&embedding, self.threshold)
            .and_then(|name| name.parse::<u32>().ok())
        {
            return Ok(SpeakerId(id));
        }

        // No confident match. Minting a new speaker from a *short* utterance is the main source of
        // spurious speakers — a back-channel like "系" is too brief for a stable embedding — so unless
        // this clip is long enough, attach it to the nearest known voice instead of a new id. (The
        // first voice has nothing to fall back to and always registers.)
        let min_samples = (self.min_new_speaker_secs * sample_rate as f32) as usize;
        if audio.len() < min_samples {
            if let Some(id) = self
                .manager
                .search(&embedding, MATCH_NEAREST)
                .and_then(|name| name.parse::<u32>().ok())
            {
                return Ok(SpeakerId(id));
            }
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

    /// Renaming these env vars silently breaks any operator who pinned diarization sensitivity.
    #[test]
    fn env_var_names_are_pinned() {
        assert_eq!(
            SherpaLiveDiarizer::MIN_NEW_SPEAKER_SECS_ENV,
            "WISP_LIVE_DIARIZE_MIN_NEW_SPEAKER_SECS"
        );
        assert_eq!(
            SherpaLiveDiarizer::SIMILARITY_THRESHOLD_ENV,
            "WISP_LIVE_DIARIZE_SIMILARITY"
        );
    }

    /// The diarization model, or `None` to skip — these tests run real sherpa inference, gated so
    /// `cargo test` is clean where the model isn't available:
    ///   WISP_LIVE_DIARIZE_TEST_EMBEDDING=/path/to/embedding.onnx
    fn test_model() -> Option<PathBuf> {
        let model = PathBuf::from(std::env::var_os("WISP_LIVE_DIARIZE_TEST_EMBEDDING")?);
        model.exists().then_some(model)
    }

    #[test]
    fn identical_audio_maps_to_the_same_speaker() {
        let Some(model) = test_model() else { return };

        let mut diarizer = SherpaLiveDiarizer::new(&model).expect("load embedding model");
        let audio = vec![0.05f32; 16_000]; // 1 s at 16 kHz
        let first = diarizer.identify(&audio, 16_000).expect("identify");
        let again = diarizer.identify(&audio, 16_000).expect("identify");

        assert_eq!(first, again, "identical audio must map to the same speaker");
    }

    #[test]
    fn short_utterance_does_not_mint_a_new_speaker() {
        let Some(model) = test_model() else { return };

        let mut diarizer = SherpaLiveDiarizer::new(&model).expect("load embedding model");
        // A long first utterance registers speaker 0.
        let long = vec![0.05f32; 16_000 * 3]; // 3 s
        let first = diarizer.identify(&long, 16_000).expect("identify");
        // A clip well under the 1.5 s gate must reuse an existing speaker, not create a new id.
        let short = vec![0.2f32; 4_000]; // 0.25 s
        let labelled = diarizer.identify(&short, 16_000).expect("identify");

        assert_eq!(
            labelled, first,
            "a short clip attaches to the nearest speaker, never a fresh id"
        );
    }
}
