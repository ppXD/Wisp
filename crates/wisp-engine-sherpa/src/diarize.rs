//! sherpa-onnx offline speaker diarization — labels who speaks when over a whole clip.
//!
//! Wraps sherpa-onnx's `OfflineSpeakerDiarization` (pyannote segmentation + speaker embeddings +
//! clustering) behind [`wisp_core::ClipDiarizer`]. It consumes the entire recording at once, which
//! is what lets it cluster every voice consistently; the resulting [`SpeakerSpan`]s are mapped onto
//! transcript segments by [`wisp_core::assign_speakers`].

use std::path::Path;
use std::time::Duration;

use sherpa_rs::diarize::{Diarize, DiarizeConfig};
use wisp_core::diarize::{ClipDiarizer, SpeakerSpan};
use wisp_core::error::{Result, WispError};
use wisp_core::transcript::SpeakerId;

/// An offline [`ClipDiarizer`] backed by sherpa-onnx.
pub struct SherpaDiarizer {
    diarize: Diarize,
}

impl SherpaDiarizer {
    /// Loads the pyannote `segmentation` model and a speaker `embedding` model.
    ///
    /// The speaker count is auto-detected by clustering voices on similarity, so a meeting with an
    /// unknown number of participants needs no configuration.
    pub fn new(segmentation: &Path, embedding: &Path) -> Result<Self> {
        let config = DiarizeConfig {
            num_clusters: Some(-1), // < 0 → auto-detect the speaker count via `threshold`
            threshold: Some(0.5),   // clustering distance; lower splits into more speakers
            min_duration_on: Some(0.3),
            min_duration_off: Some(0.5),
            provider: None,
            debug: false,
        };

        let diarize = Diarize::new(segmentation, embedding, config)
            .map_err(|e| WispError::Engine(format!("diarization init: {e}")))?;

        Ok(Self { diarize })
    }
}

impl ClipDiarizer for SherpaDiarizer {
    fn diarize_clip(&mut self, audio: &[f32], _sample_rate: u32) -> Result<Vec<SpeakerSpan>> {
        if audio.is_empty() {
            return Ok(Vec::new());
        }

        let segments = self
            .diarize
            .compute(audio.to_vec(), None)
            .map_err(|e| WispError::Engine(format!("diarization failed: {e}")))?;

        Ok(segments.into_iter().map(to_span).collect())
    }
}

/// Convert a sherpa diarization segment (seconds, `i32` speaker) to a core [`SpeakerSpan`].
fn to_span(s: sherpa_rs::diarize::Segment) -> SpeakerSpan {
    SpeakerSpan {
        speaker: SpeakerId(s.speaker.max(0) as u32),
        start: Duration::from_secs_f32(s.start.max(0.0)),
        end: Duration::from_secs_f32(s.end.max(0.0)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    /// End-to-end (real ONNX inference): diarizes a fixture WAV with the real models. Skip-guarded
    /// behind three env vars so `cargo test` is clean where the assets aren't available:
    ///   WISP_DIARIZE_TEST_SEG=/path/to/segmentation.onnx
    ///   WISP_DIARIZE_TEST_EMB=/path/to/embedding.onnx
    ///   WISP_DIARIZE_TEST_WAV=/path/to/16khz-mono-multi-speaker.wav
    #[test]
    fn diarizes_fixture_wav_when_assets_present() {
        let (Some(seg), Some(emb), Some(wav)) = (
            std::env::var_os("WISP_DIARIZE_TEST_SEG"),
            std::env::var_os("WISP_DIARIZE_TEST_EMB"),
            std::env::var_os("WISP_DIARIZE_TEST_WAV"),
        ) else {
            return;
        };
        let (seg, emb, wav) = (PathBuf::from(seg), PathBuf::from(emb), PathBuf::from(wav));
        if !seg.exists() || !emb.exists() || !wav.exists() {
            return;
        }

        let mut reader = hound::WavReader::open(&wav).expect("open fixture wav");
        let audio: Vec<f32> = reader
            .samples::<i16>()
            .map(|s| s.expect("read sample") as f32 / 32768.0)
            .collect();

        let mut diarizer = SherpaDiarizer::new(&seg, &emb).expect("load diarization models");
        let spans = diarizer.diarize_clip(&audio, 16_000).expect("diarize");

        assert!(!spans.is_empty(), "expected at least one speaker span");
        assert!(
            spans.iter().all(|s| s.end >= s.start),
            "spans must be well-formed"
        );
    }
}
