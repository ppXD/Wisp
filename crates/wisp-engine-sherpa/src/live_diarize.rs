//! Incremental (live) speaker diarization via sherpa-onnx speaker embeddings.
//!
//! Implements [`wisp_core::diarize::Diarizer`]: for each utterance it computes a speaker embedding
//! and hands it to [`OnlineClusters`], which assigns one stable [`SpeakerId`] per recurring voice by
//! nearest running centroid. A short utterance can't mint a new speaker (its embedding is unreliable)
//! — it attaches to the nearest known voice instead. Reuses the `embedding.onnx` from the
//! downloadable diarization pack — no extra model.

use std::path::Path;

use sherpa_rs::speaker_id::{EmbeddingExtractor, ExtractorConfig, DEFAULT_SIMILARITY_THRESHOLD};
use wisp_core::diarize::{Diarizer, OnlineClusters};
use wisp_core::error::{Result, WispError};
use wisp_core::transcript::SpeakerId;

/// Default seconds of speech an utterance must carry before it may register a *new* speaker. A
/// speaker-embedding model needs a couple of seconds to be stable; shorter clips (back-channels like
/// "系") embed unreliably and would otherwise each spawn a spurious speaker.
const DEFAULT_MIN_NEW_SPEAKER_SECS: f32 = 1.5;

/// Online speaker labelling for the live pipeline: one stable [`SpeakerId`] per recurring voice.
pub struct SherpaLiveDiarizer {
    extractor: EmbeddingExtractor,
    clusters: OnlineClusters,
    min_new_speaker_secs: f32,
}

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

        let threshold = env_f32(Self::SIMILARITY_THRESHOLD_ENV, DEFAULT_SIMILARITY_THRESHOLD);
        let min_new_speaker_secs =
            env_f32(Self::MIN_NEW_SPEAKER_SECS_ENV, DEFAULT_MIN_NEW_SPEAKER_SECS).max(0.0);

        Ok(Self {
            extractor,
            clusters: OnlineClusters::new(threshold),
            min_new_speaker_secs,
        })
    }
}

impl Diarizer for SherpaLiveDiarizer {
    fn identify(&mut self, audio: &[f32], sample_rate: u32) -> Result<SpeakerId> {
        let embedding = self
            .extractor
            .compute_speaker_embedding(audio.to_vec(), sample_rate)
            .map_err(|e| WispError::Engine(format!("speaker embedding: {e}")))?;

        // A clip is "reliable" enough to mint or refine a speaker only when it carries at least
        // `min_new_speaker_secs` of speech — shorter clips embed too noisily, so they attach to the
        // nearest known voice without disturbing it (see [`OnlineClusters::assign`]).
        let min_samples = (self.min_new_speaker_secs * sample_rate as f32) as usize;
        let reliable = audio.len() >= min_samples;

        Ok(SpeakerId(self.clusters.assign(&embedding, reliable)))
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
