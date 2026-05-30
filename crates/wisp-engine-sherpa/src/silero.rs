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

/// Trailing silence that closes an utterance, in seconds. 0.4 s sits at a natural clause boundary —
/// snappier to finalize than the old 0.5 s without splitting mid-sentence (and partials already
/// show provisional text while the speaker talks). Overridable via
/// [`SileroSegmenter::MIN_SILENCE_SECS_ENV`].
const DEFAULT_MIN_SILENCE_SECS: f32 = 0.4;

/// Look-back retained to seed a partial at speech onset, so the first provisional decode isn't
/// clipped by the VAD's onset latency (it confirms speech ~`min_speech_duration` in).
const PREROLL_SAMPLES: usize = (SAMPLE_RATE as usize * 3) / 10; // 0.3 s

/// Don't expose a partial until the open utterance holds at least this much audio — decoding a
/// shorter sliver just invites the engine to hallucinate.
const MIN_PARTIAL_SAMPLES: usize = (SAMPLE_RATE as usize * 2) / 5; // 0.4 s

/// A [`Segmenter`] backed by the Silero neural VAD.
pub struct SileroSegmenter {
    vad: SileroVad,
    /// Audio of the currently-open utterance, mirrored from the fed stream while the VAD reports
    /// speech — the source for provisional [`partial`](Segmenter::partial) decodes.
    in_progress: Vec<f32>,
    /// Capture-elapsed start of `in_progress`, for offsetting the partial's segment times.
    in_progress_start: Duration,
    /// Short look-back ring of recent samples, used to seed `in_progress` at onset.
    preroll: Vec<f32>,
    /// Total samples fed so far, for deriving `in_progress_start`.
    samples_fed: u64,
}

impl SileroSegmenter {
    /// Env var overriding the trailing-silence-to-finalize duration, in seconds. Lower = snappier
    /// finals but more aggressive splitting; higher = safer merges but laggier. Clamped to a sane
    /// range; an unset or unparseable value uses [`DEFAULT_MIN_SILENCE_SECS`].
    pub const MIN_SILENCE_SECS_ENV: &'static str = "WISP_VAD_MIN_SILENCE_SECS";

    /// Loads the Silero model at `model` (the bundled `silero_vad.onnx`).
    pub fn new(model: &Path) -> Result<Self> {
        let config = SileroVadConfig {
            model: model.to_string_lossy().into_owned(),
            // End an utterance after a short trailing silence (tunable); ignore speech blips
            // < 0.25 s (noise) and force a split if someone talks past 20 s without a pause.
            min_silence_duration: min_silence_secs(),
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

        Ok(Self {
            vad,
            in_progress: Vec::new(),
            in_progress_start: Duration::ZERO,
            preroll: Vec::new(),
            samples_fed: 0,
        })
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

        // Keep a short look-back of recent audio so a partial can include the onset the VAD only
        // confirms a fraction of a second in.
        self.preroll.extend_from_slice(mono);
        let overflow = self.preroll.len().saturating_sub(PREROLL_SAMPLES);
        if overflow > 0 {
            self.preroll.drain(..overflow);
        }
        self.samples_fed += mono.len() as u64;

        // Mirror the open utterance while the VAD reports speech; seed from the look-back at onset.
        let in_speech = self.vad.is_speech();
        if in_speech {
            if self.in_progress.is_empty() {
                let start_sample = self.samples_fed.saturating_sub(self.preroll.len() as u64);
                self.in_progress_start =
                    Duration::from_secs_f64(start_sample as f64 / SAMPLE_RATE as f64);
                self.in_progress = self.preroll.clone();
            } else {
                self.in_progress.extend_from_slice(mono);
            }
        }

        let finals = self.drain();
        // The open utterance is void once a final is produced (superseded) or speech stops without
        // one (a sub-threshold blip) — clear the mirror either way.
        if !finals.is_empty() || !in_speech {
            self.in_progress.clear();
        }
        finals
    }

    fn flush(&mut self) -> Vec<Utterance> {
        self.vad.flush();
        self.in_progress.clear();
        self.drain()
    }

    fn partial(&self) -> Option<Utterance> {
        if self.in_progress.len() < MIN_PARTIAL_SAMPLES {
            return None;
        }
        Some(Utterance {
            audio: self.in_progress.clone(),
            start: self.in_progress_start,
        })
    }
}

/// Reads the finalize-silence override, clamped to a sane `[0.2, 2.0]` s; falls back to the default
/// when unset or unparseable so a stray value can never break segmentation.
fn min_silence_secs() -> f32 {
    std::env::var(SileroSegmenter::MIN_SILENCE_SECS_ENV)
        .ok()
        .and_then(|v| v.parse::<f32>().ok())
        .filter(|v| v.is_finite())
        .map(|v| v.clamp(0.2, 2.0))
        .unwrap_or(DEFAULT_MIN_SILENCE_SECS)
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
            assert!(
                seg.partial().is_none(),
                "silence must not expose an in-progress partial"
            );
        }
        assert!(seg.flush().is_empty());
    }

    /// Renaming this constant silently breaks anyone who pinned a finalize-silence override via env.
    #[test]
    fn min_silence_env_var_name_is_pinned() {
        assert_eq!(
            SileroSegmenter::MIN_SILENCE_SECS_ENV,
            "WISP_VAD_MIN_SILENCE_SECS"
        );
    }

    /// The whole override lifecycle in one test — the env var is process-global, so keeping it
    /// serial here avoids racing sibling tests.
    #[test]
    fn min_silence_override_parses_clamps_and_defaults() {
        let env = SileroSegmenter::MIN_SILENCE_SECS_ENV;

        std::env::remove_var(env);
        assert_eq!(
            min_silence_secs(),
            DEFAULT_MIN_SILENCE_SECS,
            "unset → default"
        );

        std::env::set_var(env, "0.3");
        assert_eq!(min_silence_secs(), 0.3, "a valid value is honoured");

        std::env::set_var(env, "0.05");
        assert_eq!(min_silence_secs(), 0.2, "below the floor clamps up");

        std::env::set_var(env, "9.0");
        assert_eq!(min_silence_secs(), 2.0, "above the ceiling clamps down");

        std::env::set_var(env, "nonsense");
        assert_eq!(
            min_silence_secs(),
            DEFAULT_MIN_SILENCE_SECS,
            "an unparseable value falls back to the default"
        );

        std::env::remove_var(env);
    }
}
