//! Audio preprocessing for ASR: rumble/DC removal and level normalization.
//!
//! Real-world meeting audio arrives at wildly different levels — a quiet laptop mic, a hot
//! interface, a recording with DC bias or low-frequency rumble. ASR models decode such audio more
//! reliably when it is brought to a consistent, healthy level first.
//!
//! The chain is deliberately **gentle**: a one-pole high-pass (removes DC and sub-speech rumble)
//! followed by speech-gated level normalization to a target RMS, capped by a peak ceiling so the
//! result never clips. We intentionally do **not** apply spectral denoise — Whisper-class models
//! are trained on noisy audio and are robust to it, while aggressive denoise introduces artifacts
//! that *reduce* accuracy. Cleanup, not surgery.

/// High-pass cutoff (Hz): below this is DC offset and rumble (HVAC, handling), not speech. Male
/// fundamentals start ~85 Hz, so 70 Hz is a safe floor.
const HPF_CUTOFF_HZ: f32 = 70.0;

/// Target RMS level, linear (≈ -20 dBFS). A healthy speech level that leaves headroom for peaks.
const TARGET_RMS: f32 = 0.1;

/// Peak ceiling, linear (≈ -1 dBFS). After gain, the signal is scaled so its peak never exceeds
/// this — no clipping, no distortion (a single linear scale, not a limiter).
const PEAK_CEILING: f32 = 0.891;

/// Maximum boost (linear, ≈ +20 dB). Caps how far a quiet clip is amplified so we never blow up
/// the noise floor of barely-audible material.
const MAX_GAIN: f32 = 10.0;

/// Frames quieter than this (mean-square, ≈ -60 dBFS RMS) are treated as digital silence when
/// measuring the speech level.
const SILENCE_FLOOR_MS: f32 = 1.0e-6;

/// A frame counts toward the speech level only if its energy is within this ratio of the loudest
/// frame (≈ -25 dB). This gates out silence and room tone so the level targets speech, not gaps.
const GATE_REL_MS: f32 = 3.16e-3;

/// Normalize `samples` (mono `f32`) for ASR: remove DC/rumble, then bring the speech level to a
/// consistent target without clipping. Returns the cleaned signal; silent or empty input is
/// returned essentially unchanged. Allocates a copy — for a whole-clip buffer the caller owns and can
/// mutate, prefer [`normalize_for_asr_in_place`], which skips the extra clip-sized allocation.
pub fn normalize_for_asr(samples: &[f32], sample_rate: u32) -> Vec<f32> {
    let mut out = samples.to_vec();
    normalize_for_asr_in_place(&mut out, sample_rate);
    out
}

/// In-place [`normalize_for_asr`]: removes DC/rumble and normalizes the speech level **without**
/// allocating a second full-length buffer — the high-pass and the gain are written straight into
/// `samples`. Decoding a long file to one big `Vec<f32>` and normalizing it in place avoids holding a
/// redundant clip-sized copy through the (long) transcription and diarization that follow. Empty or
/// zero-rate input is left untouched.
pub fn normalize_for_asr_in_place(samples: &mut [f32], sample_rate: u32) {
    if samples.is_empty() || sample_rate == 0 {
        return;
    }

    high_pass_in_place(samples, HPF_CUTOFF_HZ, sample_rate);

    let Some(rms) = gated_rms(samples, sample_rate) else {
        return; // no speech-level content — leave the (now de-rumbled) signal alone
    };

    let gain = (TARGET_RMS / rms).min(MAX_GAIN);
    apply_gain_capped_in_place(samples, gain);
}

/// One-pole high-pass filter, in place. Removes DC and frequencies below `cutoff_hz` with a gentle
/// 6 dB/oct slope — enough to clear rumble without touching speech. The recurrence reads each sample's
/// original value, so the previous input is stashed in a scalar before its slot is overwritten.
fn high_pass_in_place(samples: &mut [f32], cutoff_hz: f32, sample_rate: u32) {
    if samples.is_empty() {
        return;
    }

    let dt = 1.0 / sample_rate as f32;
    let rc = 1.0 / (std::f32::consts::TAU * cutoff_hz);
    let alpha = rc / (rc + dt);

    let mut prev_in = samples[0];
    let mut prev_out = 0.0;
    samples[0] = 0.0; // start from rest; the first sample's DC is dropped
    for s in samples.iter_mut().skip(1) {
        let x = *s;
        let y = alpha * (prev_out + x - prev_in);
        *s = y;
        prev_out = y;
        prev_in = x;
    }
}

/// Speech-gated RMS: measures the level over 20 ms frames, counting only frames within
/// [`GATE_REL_MS`] of the loudest, so silence and room tone don't drag the estimate down. Returns
/// `None` when the whole signal is below [`SILENCE_FLOOR_MS`] (nothing to normalize).
fn gated_rms(samples: &[f32], sample_rate: u32) -> Option<f32> {
    let frame = (sample_rate as usize / 50).max(1); // 20 ms
    let frame_ms: Vec<f32> = samples.chunks(frame).map(mean_square).collect();

    let max_ms = frame_ms.iter().copied().fold(0.0_f32, f32::max);
    if max_ms <= SILENCE_FLOOR_MS {
        return None;
    }

    let gate = max_ms * GATE_REL_MS;
    let active: Vec<f32> = frame_ms.into_iter().filter(|&m| m >= gate).collect();
    if active.is_empty() {
        return None;
    }

    let mean = active.iter().sum::<f32>() / active.len() as f32;
    Some(mean.sqrt())
}

/// Apply `gain` in place, but if that would push the peak above [`PEAK_CEILING`], scale the whole
/// signal down so the peak lands exactly on the ceiling instead. Never clips.
fn apply_gain_capped_in_place(samples: &mut [f32], gain: f32) {
    let peak = samples.iter().fold(0.0_f32, |m, &x| m.max(x.abs()));
    let gain = if peak * gain > PEAK_CEILING {
        PEAK_CEILING / peak
    } else {
        gain
    };
    for s in samples.iter_mut() {
        *s *= gain;
    }
}

/// Mean of the squares of a frame (its energy / power).
fn mean_square(frame: &[f32]) -> f32 {
    if frame.is_empty() {
        return 0.0;
    }
    frame.iter().map(|&x| x * x).sum::<f32>() / frame.len() as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    const SR: u32 = 16_000;

    /// One second of a `freq`-Hz sine at `amp`, optionally with a `dc` offset.
    fn tone(freq: f32, amp: f32, dc: f32) -> Vec<f32> {
        (0..SR)
            .map(|n| {
                let t = n as f32 / SR as f32;
                dc + amp * (std::f32::consts::TAU * freq * t).sin()
            })
            .collect()
    }

    fn rms(samples: &[f32]) -> f32 {
        (samples.iter().map(|&x| x * x).sum::<f32>() / samples.len() as f32).sqrt()
    }

    fn peak(samples: &[f32]) -> f32 {
        samples.iter().fold(0.0_f32, |m, &x| m.max(x.abs()))
    }

    #[test]
    fn boosts_quiet_speech_toward_target() {
        // Quiet tone needing < 20 dB of boost (so the gain cap doesn't bind): RMS lands near target.
        let out = normalize_for_asr(&tone(300.0, 0.02, 0.0), SR);
        assert!(
            (rms(&out) - TARGET_RMS).abs() < 0.015,
            "expected RMS ≈ {TARGET_RMS}, got {}",
            rms(&out)
        );
        assert!(peak(&out) <= PEAK_CEILING + 1e-4);
    }

    #[test]
    fn attenuates_hot_audio_without_clipping() {
        // Hot tone (RMS well above target) is pulled down, never up.
        let out = normalize_for_asr(&tone(300.0, 0.9, 0.0), SR);
        assert!(rms(&out) < 0.9, "hot audio should be attenuated");
        assert!((rms(&out) - TARGET_RMS).abs() < 0.02);
        assert!(peak(&out) <= PEAK_CEILING + 1e-4, "must not clip");
    }

    #[test]
    fn high_crest_transient_never_clips() {
        // Quiet tone (→ large gain) plus a lone transient spike: the peak ceiling must hold.
        let mut signal = tone(300.0, 0.02, 0.0);
        signal[8_000] = 0.5;
        let out = normalize_for_asr(&signal, SR);
        assert!(
            peak(&out) <= PEAK_CEILING + 1e-4,
            "peak {} exceeded ceiling",
            peak(&out)
        );
    }

    #[test]
    fn removes_dc_offset() {
        // A tone riding on a large DC bias: output is centered (mean ≈ 0).
        let out = normalize_for_asr(&tone(300.0, 0.05, 0.3), SR);
        let mean = out.iter().sum::<f32>() / out.len() as f32;
        assert!(mean.abs() < 1e-3, "DC not removed: mean {mean}");
    }

    #[test]
    fn silence_is_left_untouched_and_finite() {
        let out = normalize_for_asr(&vec![0.0; SR as usize], SR);
        assert!(out.iter().all(|v| v.abs() < 1e-6 && v.is_finite()));
    }

    #[test]
    fn empty_or_zero_rate_is_passthrough() {
        assert!(normalize_for_asr(&[], SR).is_empty());
        assert_eq!(normalize_for_asr(&[0.1, 0.2], 0), vec![0.1, 0.2]);
    }

    #[test]
    fn output_is_always_finite() {
        let out = normalize_for_asr(&tone(440.0, 0.2, 0.05), SR);
        assert!(out.iter().all(|v| v.is_finite()));
        assert_eq!(out.len(), SR as usize);
    }

    #[test]
    fn in_place_matches_allocating() {
        // The in-place path is the real implementation and the allocating one wraps it, so the two
        // must stay bit-identical across every branch: quiet (boosted), hot (attenuated), DC-biased,
        // pure silence (gate returns None), and a sub-window clip. This pins them together.
        let cases = [
            tone(300.0, 0.02, 0.0),
            tone(300.0, 0.9, 0.0),
            tone(300.0, 0.05, 0.3),
            vec![0.0; SR as usize],
            vec![0.25; 3],
        ];
        for signal in cases {
            let expected = normalize_for_asr(&signal, SR);
            let mut got = signal.clone();
            normalize_for_asr_in_place(&mut got, SR);
            assert_eq!(
                got,
                expected,
                "in-place diverged for a {}-sample clip",
                signal.len()
            );
        }

        // Empty and zero-rate are left untouched, matching the allocating passthrough.
        let mut empty: Vec<f32> = Vec::new();
        normalize_for_asr_in_place(&mut empty, SR);
        assert!(empty.is_empty());

        let mut passthrough = vec![0.1, 0.2];
        normalize_for_asr_in_place(&mut passthrough, 0);
        assert_eq!(passthrough, vec![0.1, 0.2]);
    }
}
