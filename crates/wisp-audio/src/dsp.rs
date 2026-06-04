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
///
/// Stateless and dependency-free, fine for whole-clip (file) conversion. For a *live* stream prefer
/// [`Resampler`], which anti-aliases and joins frames seamlessly.
pub fn to_mono_16k(frame: &AudioFrame) -> Vec<f32> {
    let mono = downmix_to_mono(&frame.samples, frame.channels);
    resample_linear(&mono, frame.sample_rate, TARGET_SAMPLE_RATE)
}

/// A streaming, anti-aliased sample-rate converter to a target mono rate.
///
/// Plain linear resampling ([`to_mono_16k`]) folds any content above the output Nyquist back into the
/// speech band as aliasing noise, which the ASR engine then has to fight. This applies a stateful
/// windowed-sinc low-pass *before* downsampling, so nothing above ~0.45·`to_rate` aliases and
/// successive frames of one stream join seamlessly. Hold one per stream and feed it frames in order;
/// it lazily designs its filter once the source rate is known and rebuilds it if the rate changes.
///
/// The target is usually [`TARGET_SAMPLE_RATE`] (the on-device ASR rate); a cloud engine that accepts
/// richer audio can target a higher rate (e.g. 24 kHz) to keep the fricative band a 16 kHz pass drops.
pub struct Resampler {
    to_rate: u32,
    from_rate: u32,
    coeffs: Vec<f32>,
    history: Vec<f32>,
}

impl Resampler {
    /// A fresh converter to `to_rate` mono; the anti-alias filter is built on the first downsampling
    /// frame.
    pub fn new(to_rate: u32) -> Self {
        Self {
            to_rate,
            from_rate: 0,
            coeffs: Vec::new(),
            history: Vec::new(),
        }
    }

    /// Downmixes `frame` to mono and converts it to the target rate, anti-aliasing when downsampling.
    pub fn process(&mut self, frame: &AudioFrame) -> Vec<f32> {
        let mono = downmix_to_mono(&frame.samples, frame.channels);

        // Up-sampling / same-rate can't alias, so skip the filter (and its latency) entirely.
        if frame.sample_rate <= self.to_rate {
            return resample_linear(&mono, frame.sample_rate, self.to_rate);
        }

        if self.from_rate != frame.sample_rate {
            self.from_rate = frame.sample_rate;
            self.coeffs = design_lowpass(frame.sample_rate, self.to_rate);
            self.history = vec![0.0; self.coeffs.len().saturating_sub(1)];
        }

        let filtered = self.low_pass(&mono);
        resample_linear(&filtered, frame.sample_rate, self.to_rate)
    }

    /// Stateful FIR convolution: convolve `[saved history ++ input]`, then keep the new tail as the
    /// next call's history so the stream stays continuous across frame boundaries.
    fn low_pass(&mut self, input: &[f32]) -> Vec<f32> {
        let taps = self.coeffs.len();
        if taps == 0 || input.is_empty() {
            return input.to_vec();
        }
        let mut buf = std::mem::take(&mut self.history);
        buf.extend_from_slice(input);

        let out: Vec<f32> = (0..input.len())
            .map(|i| self.coeffs.iter().zip(&buf[i..]).map(|(c, x)| c * x).sum())
            .collect();

        self.history = buf[buf.len() - (taps - 1)..].to_vec();
        out
    }
}

/// A unity-DC-gain windowed-sinc low-pass FIR, cutoff just under `to_rate`/2, sampled at `from_rate`.
fn design_lowpass(from_rate: u32, to_rate: u32) -> Vec<f32> {
    use std::f32::consts::PI;
    const TAPS: usize = 63; // odd → symmetric, constant (integer) group delay

    // Cut off at 0.45·to_rate, a touch below the 0.5·to_rate output Nyquist for a transition band.
    let fc = 0.45 * to_rate as f32 / from_rate as f32; // cycles per input sample
    let mid = (TAPS - 1) as f32 / 2.0;

    let mut h: Vec<f32> = (0..TAPS)
        .map(|i| {
            let x = i as f32 - mid;
            let sinc = if x == 0.0 {
                2.0 * fc
            } else {
                (2.0 * PI * fc * x).sin() / (PI * x)
            };
            let hann = 0.5 - 0.5 * (2.0 * PI * i as f32 / (TAPS - 1) as f32).cos();
            sinc * hann
        })
        .collect();

    let sum: f32 = h.iter().sum();
    if sum != 0.0 {
        for c in &mut h {
            *c /= sum;
        }
    }
    h
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

    #[test]
    fn resampler16k_passes_low_frequencies() {
        use std::f32::consts::PI;
        // A 1 kHz tone at 48 kHz is well inside the 8 kHz passband — it must survive conversion.
        let tone: Vec<f32> = (0..4_800)
            .map(|i| (2.0 * PI * 1_000.0 * i as f32 / 48_000.0).sin())
            .collect();
        let frame = AudioFrame::new(tone, 48_000, 1, Duration::ZERO);
        let out = Resampler::new(TARGET_SAMPLE_RATE).process(&frame);
        let rms = (out.iter().map(|v| v * v).sum::<f32>() / out.len() as f32).sqrt();
        assert!(rms > 0.5, "a 1 kHz tone must pass through (rms {rms})");
    }

    #[test]
    fn resampler16k_suppresses_aliasing_that_linear_lets_through() {
        use std::f32::consts::PI;
        // A 12 kHz tone at 48 kHz is above the 8 kHz output Nyquist: naively downsampled to 16 kHz it
        // aliases down to 4 kHz. Linear resampling passes that alias through; the anti-aliased
        // resampler removes the tone first, so far less energy survives.
        let tone: Vec<f32> = (0..9_600)
            .map(|i| (2.0 * PI * 12_000.0 * i as f32 / 48_000.0).sin())
            .collect();

        let linear = resample_linear(&tone, 48_000, 16_000);
        let frame = AudioFrame::new(tone, 48_000, 1, Duration::ZERO);
        let antialiased = Resampler::new(TARGET_SAMPLE_RATE).process(&frame);

        let rms = |s: &[f32]| (s.iter().map(|v| v * v).sum::<f32>() / s.len() as f32).sqrt();
        let (rms_lin, rms_aa) = (rms(&linear), rms(&antialiased));
        assert!(
            rms_aa < rms_lin * 0.25,
            "anti-aliasing should cut the alias well below linear (linear {rms_lin}, aa {rms_aa})"
        );
    }

    #[test]
    fn resampler_targeting_24k_keeps_a_band_a_16k_target_drops() {
        use std::f32::consts::PI;
        // A 9 kHz tone at 48 kHz sits above a 16 kHz target's 8 kHz Nyquist (filtered out) but inside a
        // 24 kHz target's 12 kHz Nyquist — so the richer target preserves it. This is the fidelity the
        // cloud path gains by taking native 24 kHz instead of 16 kHz-upsampled audio.
        let tone: Vec<f32> = (0..9_600)
            .map(|i| (2.0 * PI * 9_000.0 * i as f32 / 48_000.0).sin())
            .collect();
        let frame = AudioFrame::new(tone, 48_000, 1, Duration::ZERO);

        let rms = |s: &[f32]| (s.iter().map(|v| v * v).sum::<f32>() / s.len() as f32).sqrt();
        let to_16k = rms(&Resampler::new(16_000).process(&frame));
        let to_24k = rms(&Resampler::new(24_000).process(&frame));

        assert!(
            to_24k > 0.4,
            "the 24 kHz target keeps the 9 kHz tone (rms {to_24k})"
        );
        assert!(
            to_16k < to_24k * 0.5,
            "the 16 kHz target filters it out (16k {to_16k}, 24k {to_24k})"
        );
    }
}
