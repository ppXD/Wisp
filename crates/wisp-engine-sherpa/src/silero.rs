//! Silero neural-VAD segmenter.
//!
//! Drives sherpa-onnx's bundled Silero voice-activity detector as a [`Segmenter`], emitting one
//! [`Utterance`] per detected speech segment. Far better speech/non-speech discrimination than the
//! energy gate: it rejects non-speech noise (so the ASR engine isn't fed junk that it hallucinates
//! words from) and yields clean utterance boundaries (no clipped onsets, no mid-word cuts).

use std::path::Path;
use std::time::Duration;

use sherpa_rs::silero_vad::{SileroVad, SileroVadConfig};
use wisp_core::error::{Result, WispError};
use wisp_pipeline::{Segmenter, Utterance};

/// Rate the VAD — and the rest of the pipeline — operates at.
const SAMPLE_RATE: u32 = 16_000;

/// Internal ring-buffer the detector retains, in seconds. Generous so a long utterance is never
/// dropped before it closes.
const BUFFER_SECONDS: f32 = 30.0;

/// A [`Segmenter`] backed by the Silero neural VAD.
pub struct SileroSegmenter {
    vad: SileroVad,
}

impl SileroSegmenter {
    /// Loads the Silero model at `model` (the bundled `silero_vad.onnx`).
    pub fn new(model: &Path) -> Result<Self> {
        let config = SileroVadConfig {
            model: model.to_string_lossy().into_owned(),
            // End an utterance after 0.5 s of trailing silence; ignore speech blips < 0.25 s
            // (noise) and force a split if someone talks past 20 s without a pause.
            min_silence_duration: 0.5,
            min_speech_duration: 0.25,
            max_speech_duration: 20.0,
            threshold: 0.5,
            sample_rate: SAMPLE_RATE,
            window_size: 512,
            provider: None,
            num_threads: Some(1),
            debug: false,
        };

        let vad = SileroVad::new(config, BUFFER_SECONDS)
            .map_err(|e| WispError::Engine(format!("silero vad init: {e}")))?;

        Ok(Self { vad })
    }

    /// Drains every completed speech segment the detector currently holds into utterances.
    fn drain(&mut self) -> Vec<Utterance> {
        let mut utterances = Vec::new();
        while !self.vad.is_empty() {
            let segment = self.vad.front();
            self.vad.pop();
            // `segment.start` is the sample offset from the start of the fed stream → elapsed time.
            let start = Duration::from_secs_f64(segment.start as f64 / SAMPLE_RATE as f64);
            utterances.push(Utterance {
                audio: segment.samples,
                start,
            });
        }
        utterances
    }
}

impl Segmenter for SileroSegmenter {
    fn push(
        &mut self,
        mono: &[f32],
        _timestamp: Duration,
        _frame_duration: Duration,
    ) -> Vec<Utterance> {
        self.vad.accept_waveform(mono.to_vec());
        self.drain()
    }

    fn flush(&mut self) -> Vec<Utterance> {
        self.vad.flush();
        self.drain()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The bundled VAD model, relative to this crate. Absent in some checkouts (it's a build
    /// resource of the app), so tests using it are skip-guarded.
    fn model_path() -> std::path::PathBuf {
        std::path::PathBuf::from(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../app/src-tauri/resources/silero_vad.onnx"
        ))
    }

    /// Integration (real Silero model): pure silence must never produce an utterance. Skip-guarded
    /// so `cargo test` is clean where the bundled model isn't present.
    #[test]
    fn silence_yields_no_utterances() {
        let path = model_path();
        if !path.exists() {
            return;
        }

        let mut seg = SileroSegmenter::new(&path).expect("load bundled silero model");

        for i in 0..10 {
            let out = seg.push(
                &vec![0.0; 1_600],
                Duration::from_millis(i * 100),
                Duration::from_millis(100),
            );
            assert!(out.is_empty(), "silence must not produce utterances");
        }
        assert!(seg.flush().is_empty());
    }
}
