//! Audio DSP: channel downmixing and sample-rate conversion to the engine's input format.

use std::time::Duration;

use wisp_core::audio::AudioFrame;

/// Sample rate (Hz) expected by the ASR engines: 16 kHz, mono.
pub const TARGET_SAMPLE_RATE: u32 = 16_000;

/// Default frame length, in milliseconds, that file-backed sources emit.
pub const FRAME_CHUNK_MS: u64 = 100;

/// Downmix interleaved multi-channel samples to mono by averaging each channel group.
pub fn downmix_to_mono(samples: &[f32], channels: u16) -> Vec<f32> {
    let channels = channels.max(1) as usize;
    if channels == 1 {
        return samples.to_vec();
    }

    samples
        .chunks(channels)
        .map(|group| group.iter().sum::<f32>() / group.len() as f32)
        .collect()
}

/// Resample a mono signal from `from_rate` to `to_rate` with linear interpolation.
///
/// Linear interpolation is modest quality but dependency-free and adequate for a first pass; it
/// can be swapped for a higher-quality resampler behind this same function without breaking
/// callers.
pub fn resample_linear(input: &[f32], from_rate: u32, to_rate: u32) -> Vec<f32> {
    if from_rate == 0 || to_rate == 0 || input.is_empty() {
        return Vec::new();
    }
    if from_rate == to_rate {
        return input.to_vec();
    }

    let ratio = f64::from(to_rate) / f64::from(from_rate);
    let out_len = (input.len() as f64 * ratio).round() as usize;
    let last = input.len() - 1;

    (0..out_len)
        .map(|i| {
            let src_pos = i as f64 / ratio;
            let idx = src_pos.floor() as usize;
            let frac = src_pos - idx as f64;
            let a = f64::from(input[idx.min(last)]);
            let b = f64::from(input[(idx + 1).min(last)]);
            (a + (b - a) * frac) as f32
        })
        .collect()
}

/// Convert an [`AudioFrame`] of any rate/channel count to [`TARGET_SAMPLE_RATE`] mono f32.
pub fn to_mono_16k(frame: &AudioFrame) -> Vec<f32> {
    let mono = downmix_to_mono(&frame.samples, frame.channels);
    resample_linear(&mono, frame.sample_rate, TARGET_SAMPLE_RATE)
}

/// Split decoded interleaved `samples` into `chunk_ms`-long [`AudioFrame`]s, each stamped with its
/// offset from the start. Shared by the file-backed sources (WAV and media decoders).
pub fn chunk_into_frames(
    samples: &[f32],
    sample_rate: u32,
    channels: u16,
    chunk_ms: u64,
) -> Vec<AudioFrame> {
    let channels = channels.max(1) as usize;
    let instants_per_chunk = (u64::from(sample_rate) * chunk_ms / 1000).max(1) as usize;
    let per_chunk = instants_per_chunk * channels;

    let mut frames = Vec::new();
    let mut instants_emitted: u64 = 0;

    for chunk in samples.chunks(per_chunk) {
        let timestamp = if sample_rate == 0 {
            Duration::ZERO
        } else {
            Duration::from_secs_f64(instants_emitted as f64 / f64::from(sample_rate))
        };
        frames.push(AudioFrame::new(
            chunk.to_vec(),
            sample_rate,
            channels as u16,
            timestamp,
        ));
        instants_emitted += (chunk.len() / channels) as u64;
    }

    frames
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn downmix_averages_stereo_pairs() {
        // Interleaved L,R: (1,3) -> 2, (2,4) -> 3.
        assert_eq!(downmix_to_mono(&[1.0, 3.0, 2.0, 4.0], 2), vec![2.0, 3.0]);
    }

    #[test]
    fn downmix_mono_is_identity() {
        assert_eq!(downmix_to_mono(&[1.0, 2.0, 3.0], 1), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn resample_same_rate_is_identity() {
        let x = vec![0.1, 0.2, 0.3];
        assert_eq!(resample_linear(&x, 16_000, 16_000), x);
    }

    #[test]
    fn resample_constant_signal_stays_constant() {
        let x = vec![0.5f32; 100];
        let y = resample_linear(&x, 48_000, 16_000);
        assert!(!y.is_empty());
        assert!(y.iter().all(|v| (v - 0.5).abs() < 1e-6));
    }

    #[test]
    fn resample_downsamples_length_by_ratio() {
        let x = vec![0.0f32; 300];
        assert_eq!(resample_linear(&x, 48_000, 16_000).len(), 100);
    }

    #[test]
    fn resample_empty_or_zero_rate_is_empty() {
        assert!(resample_linear(&[], 48_000, 16_000).is_empty());
        assert!(resample_linear(&[1.0], 0, 16_000).is_empty());
        assert!(resample_linear(&[1.0], 48_000, 0).is_empty());
    }

    #[test]
    fn to_mono_16k_downmixes_then_resamples() {
        // 48 kHz stereo, 300 interleaved (150 instants) -> mono 150 -> 16 kHz -> 50.
        let frame = AudioFrame::new(vec![0.25f32; 300], 48_000, 2, Duration::ZERO);
        let out = to_mono_16k(&frame);
        assert_eq!(out.len(), 50);
        assert!(out.iter().all(|v| (v - 0.25).abs() < 1e-6));
    }
}
