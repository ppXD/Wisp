//! The light, built-in speech denoiser: a pure-Rust port of Xiph's RNNoise.

use nnnoiseless::DenoiseState;
use wisp_core::denoise::Denoiser;

use crate::dsp::resample_linear;

/// RNNoise runs at 48 kHz on 10 ms (480-sample) frames.
const RNNOISE_RATE: u32 = 48_000;
const RNNOISE_FRAME: usize = 480;
/// RNNoise expects samples in the i16 range, not normalized to `[-1.0, 1.0]`.
const I16_SCALE: f32 = 32_768.0;

/// The light, no-download denoiser.
///
/// Good on steady background noise (fans, hum, hiss); weaker on non-stationary noise like babble or
/// music. Needs no model download. Our pipeline is 16 kHz, so audio is resampled to RNNoise's native
/// 48 kHz around the denoise and back, then trimmed to the original length so timestamps still line
/// up.
pub struct RnnoiseDenoiser {
    state: Box<DenoiseState<'static>>,
}

impl RnnoiseDenoiser {
    /// Builds a denoiser using RNNoise's built-in model.
    pub fn new() -> Self {
        Self {
            state: DenoiseState::new(),
        }
    }
}

impl Default for RnnoiseDenoiser {
    fn default() -> Self {
        Self::new()
    }
}

impl Denoiser for RnnoiseDenoiser {
    fn denoise(&mut self, audio: &[f32], sample_rate: u32) -> Vec<f32> {
        if audio.is_empty() {
            return Vec::new();
        }

        // To 48 kHz, i16-scaled, padded out to a whole number of frames.
        let upsampled = resample_linear(audio, sample_rate, RNNOISE_RATE);
        let mut scaled: Vec<f32> = upsampled.iter().map(|s| s * I16_SCALE).collect();
        let pad = (RNNOISE_FRAME - scaled.len() % RNNOISE_FRAME) % RNNOISE_FRAME;
        scaled.resize(scaled.len() + pad, 0.0);

        let mut cleaned = vec![0.0f32; scaled.len()];
        for (input, output) in scaled
            .chunks_exact(RNNOISE_FRAME)
            .zip(cleaned.chunks_exact_mut(RNNOISE_FRAME))
        {
            self.state.process_frame(output, input);
        }

        // Back to the input rate, un-scaled, and trimmed to the exact original length.
        let unscaled: Vec<f32> = cleaned.iter().map(|s| s / I16_SCALE).collect();
        let mut result = resample_linear(&unscaled, RNNOISE_RATE, sample_rate);
        result.resize(audio.len(), 0.0);
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_size_matches_rnnoise() {
        // Drift detector: if RNNoise's frame size ever changes upstream, our framing breaks.
        assert_eq!(RNNOISE_FRAME, DenoiseState::FRAME_SIZE);
    }

    #[test]
    fn empty_input_is_empty() {
        assert!(RnnoiseDenoiser::new().denoise(&[], 16_000).is_empty());
    }

    #[test]
    fn preserves_length_for_any_input() {
        let mut denoiser = RnnoiseDenoiser::new();
        // Sub-frame, exact-frame, and multi-frame lengths all come back the same length.
        for n in [1usize, 100, 480, 1_600, 5_000] {
            let out = denoiser.denoise(&vec![0.1; n], 16_000);
            assert_eq!(out.len(), n, "length preserved for n = {n}");
        }
    }

    #[test]
    fn silence_stays_silent() {
        let out = RnnoiseDenoiser::new().denoise(&vec![0.0; 8_000], 16_000);
        assert!(
            out.iter().all(|s| s.abs() < 1e-3),
            "denoised silence should remain ~silent"
        );
    }

    #[test]
    fn output_is_finite_for_noisy_input() {
        let noisy: Vec<f32> = (0..8_000)
            .map(|i| (i as f32 * 0.3).sin() * 0.2 + if i % 7 == 0 { 0.1 } else { -0.05 })
            .collect();
        let out = RnnoiseDenoiser::new().denoise(&noisy, 16_000);
        assert!(
            out.iter().all(|s| s.is_finite()),
            "no NaN/inf in the output"
        );
        assert_eq!(out.len(), noisy.len());
    }
}
