//! Mixing two mono streams (the local mic and the meeting/system audio) into one — a **gain-sharing
//! automatic mixer** (the Dugan algorithm, the broadcast/conferencing standard for decades).
//!
//! Each source's gain is its *share* of the combined level, so the gains always sum to ≤ 1 and the mix
//! is a convex combination of the inputs — it can therefore **never clip** (output ≤ the loudest source
//! ≤ full-scale), with no limiter needed. The active talker takes almost all the gain (so the quiet
//! side's background bleed is suppressed), while two simultaneous talkers share it (so overlapping
//! speech is preserved). An envelope follower with a fast attack and slow release moves the gains
//! smoothly: a new talker grabs gain within a frame, and the gain doesn't pump between words.
//!
//! This beats the two naive options a meeting tool reaches for: a `(a + b).clamp(-1, 1)` sum (flat-tops
//! a loud overlap into clipping distortion) and a single-output recorder's plain ducking (drops the
//! quiet side outright, losing overlap). For a downstream listener — an ASR model or an AI assist —
//! both speakers staying intelligible *and* a clip-free signal are what matter.

/// Below this combined level the scene is silence — split the (near-zero) mix evenly rather than divide
/// noise by noise.
const SILENCE: f32 = 1e-4;

/// Per-frame envelope-follower coefficients: rise fast toward a louder peak (a new talker grabs gain
/// within ~a frame, so onsets aren't ducked) and fall slowly (the gain glides down between words instead
/// of pumping).
const ENV_ATTACK: f32 = 0.7;
const ENV_RELEASE: f32 = 0.1;

/// The peak absolute amplitude of `samples` (0 for an empty slice).
fn peak(samples: &[f32]) -> f32 {
    samples.iter().fold(0.0_f32, |m, &x| m.max(x.abs()))
}

/// Tracks a level toward `target`, rising fast and falling slowly (a standard envelope follower).
fn follow(level: f32, target: f32) -> f32 {
    let coeff = if target > level {
        ENV_ATTACK
    } else {
        ENV_RELEASE
    };
    level + coeff * (target - level)
}

/// The gain-share weights for two source levels: each source's fraction of the total. They sum to 1
/// (an even split when both are silent), so applying them yields a clip-safe convex mix.
fn share_gains(level_a: f32, level_b: f32) -> (f32, f32) {
    let total = level_a + level_b;
    if total < SILENCE {
        return (0.5, 0.5);
    }
    (level_a / total, level_b / total)
}

/// A gain-sharing automatic mixer for two mono streams. Stateful: it carries each source's smoothed
/// level across frames for the gain decision, so feed it consecutive frames.
#[derive(Debug, Default)]
pub struct MeetingMixer {
    level_primary: f32,
    level_secondary: f32,
}

impl MeetingMixer {
    pub fn new() -> Self {
        Self::default()
    }

    /// Mixes `secondary` into `primary` in place. They should be the same length; a shorter `secondary`
    /// leaves the trailing `primary` samples gain-scaled on their own. Each source is level-followed and
    /// gain-shared, so the louder talker dominates, overlap is shared, and the convex mix never clips
    /// (the final clamp only guards rare resampler overshoot of an individual source past full-scale).
    pub fn mix(&mut self, primary: &mut [f32], secondary: &[f32]) {
        self.level_primary = follow(self.level_primary, peak(primary));
        self.level_secondary = follow(self.level_secondary, peak(secondary));

        let (gain_primary, gain_secondary) = share_gains(self.level_primary, self.level_secondary);

        let overlap = primary.len().min(secondary.len());
        for (p, &s) in primary[..overlap].iter_mut().zip(&secondary[..overlap]) {
            *p = (*p * gain_primary + s * gain_secondary).clamp(-1.0, 1.0);
        }
        for p in &mut primary[overlap..] {
            *p = (*p * gain_primary).clamp(-1.0, 1.0);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Drives the mixer for enough frames that the envelopes settle, then returns the gains it would
    /// apply for the given steady-state source peaks.
    fn settled_gains(primary_peak: f32, secondary_peak: f32) -> (f32, f32) {
        let mut mixer = MeetingMixer::new();
        for _ in 0..40 {
            let mut p = vec![primary_peak; 128];
            let s = vec![secondary_peak; 128];
            mixer.mix(&mut p, &s);
        }
        share_gains(mixer.level_primary, mixer.level_secondary)
    }

    #[test]
    fn silence_splits_the_gain_evenly() {
        assert_eq!(share_gains(0.0, 0.0), (0.5, 0.5));
    }

    #[test]
    fn the_gains_are_a_convex_combination() {
        for (a, b) in [(0.8, 0.2), (0.5, 0.5), (0.9, 0.01), (0.3, 0.3)] {
            let (ga, gb) = share_gains(a, b);
            assert!((ga + gb - 1.0).abs() < 1e-5, "{a},{b} -> gains sum to 1");
            assert!(ga >= 0.0 && gb >= 0.0, "non-negative");
        }
    }

    #[test]
    fn a_dominant_talker_takes_almost_all_the_gain() {
        // Loud primary, faint secondary background → primary ~unity, secondary suppressed toward 0.
        let (gp, gs) = settled_gains(0.8, 0.04);
        assert!(gp > 0.9, "the talker dominates the mix (got {gp})");
        assert!(gs < 0.1, "the background bleed is suppressed (got {gs})");
    }

    #[test]
    fn overlap_shares_the_gain_between_both() {
        // Both equally loud (talking over each other) → roughly half each, both audible.
        let (gp, gs) = settled_gains(0.6, 0.6);
        assert!(
            (gp - 0.5).abs() < 0.05 && (gs - 0.5).abs() < 0.05,
            "both ~0.5 (got {gp},{gs})"
        );
    }

    #[test]
    fn a_loud_overlap_never_clips() {
        let mut mixer = MeetingMixer::new();
        // Two full-scale sources for several frames — a naive sum would hit 2.0 and hard-clip.
        for _ in 0..10 {
            let mut a = vec![1.0_f32; 256];
            let b = vec![1.0_f32; 256];
            mixer.mix(&mut a, &b);
            assert!(
                a.iter().all(|&s| s.abs() <= 1.0),
                "convex mix stays within full-scale"
            );
        }
    }

    #[test]
    fn a_new_talker_grabs_gain_within_about_a_frame() {
        let mut mixer = MeetingMixer::new();
        // Secondary silent for a while, then it speaks up suddenly.
        for _ in 0..20 {
            let mut p = vec![0.6_f32; 128];
            let s = vec![0.0_f32; 128];
            mixer.mix(&mut p, &s);
        }
        // One frame of the secondary jumping in loud.
        let mut p = vec![0.6_f32; 128];
        let s = vec![0.6_f32; 128];
        mixer.mix(&mut p, &s);
        let (_, gs) = share_gains(mixer.level_primary, mixer.level_secondary);
        assert!(
            gs > 0.3,
            "the interjection is heard within a frame, not ducked away (got {gs})"
        );
    }

    #[test]
    fn a_shorter_secondary_leaves_the_primary_tail_scaled_but_present() {
        let mut mixer = MeetingMixer::new();
        let mut p = vec![0.5_f32; 10];
        let s = vec![0.5_f32; 4];
        mixer.mix(&mut p, &s);
        // First frame: even split → primary tail scaled by 0.5, still present (not zeroed).
        assert!(
            p[6..].iter().all(|&x| x > 0.2 && x <= 0.5),
            "tail is the primary, gain-scaled"
        );
    }
}
