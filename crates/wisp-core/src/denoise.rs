//! Speech enhancement (noise suppression) applied before ASR.

/// Attenuates background noise in mono audio so the engine sees cleaner speech.
///
/// Opt-in and pluggable, like [`AsrEngine`](crate::engine::AsrEngine): a light pure-Rust denoiser
/// today, heavier downloadable neural models (GTCRN, DeepFilterNet) behind the same trait tomorrow
/// — none of them load-bearing, so a clip transcribes fine with no denoiser at all.
pub trait Denoiser: Send {
    /// Returns a denoised copy of `audio` (mono, sampled at `sample_rate`). The result has the same
    /// length as the input, so downstream timestamps still line up.
    fn denoise(&mut self, audio: &[f32], sample_rate: u32) -> Vec<f32>;
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A trivial denoiser used only to exercise the trait contract.
    struct HalfGain;
    impl Denoiser for HalfGain {
        fn denoise(&mut self, audio: &[f32], _sample_rate: u32) -> Vec<f32> {
            audio.iter().map(|s| s * 0.5).collect()
        }
    }

    #[test]
    fn denoiser_is_object_safe_and_keeps_length() {
        let mut denoiser: Box<dyn Denoiser> = Box::new(HalfGain);
        let out = denoiser.denoise(&[1.0, -1.0, 0.5], 16_000);
        assert_eq!(out, vec![0.5, -0.5, 0.25]);
        assert_eq!(out.len(), 3, "length is preserved");
    }
}
