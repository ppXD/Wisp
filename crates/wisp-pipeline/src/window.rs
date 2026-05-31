//! Splitting a clip into transcription windows that break at quiet points.
//!
//! Engines without a native progress callback (the sherpa offline recognizers) decode a whole clip
//! in one opaque call, so the only way to show a real percentage is to transcribe the clip in pieces
//! and report after each. A naive cut every N seconds can slice a word in half; [`window_bounds`]
//! snaps each cut to the quietest short frame nearby, so windows break in pauses instead of mid-word.

/// 16 kHz mono — the rate the whole pipeline runs at.
const SAMPLE_RATE: usize = 16_000;

/// One analysis frame for the quiet-point search: 20 ms.
const FRAME: usize = SAMPLE_RATE / 50;

/// Split `audio` into consecutive `(start, end)` sample windows of roughly `target` samples each,
/// snapping every interior cut to the lowest-energy [`FRAME`]-sample frame within `radius` samples of
/// the ideal cut so windows break in pauses, not mid-word.
///
/// The windows tile `0..audio.len()` with no gaps and no overlap. A clip at or under one window
/// length (or `target == 0`) is returned whole.
pub fn window_bounds(audio: &[f32], target: usize, radius: usize) -> Vec<(usize, usize)> {
    let n = audio.len();
    if n == 0 {
        return Vec::new();
    }
    if target == 0 || n <= target {
        return vec![(0, n)];
    }

    let mut bounds = Vec::new();
    let mut start = 0;
    while start < n {
        let ideal = start + target;
        // Not enough left for another full window → the remainder is the last window.
        if ideal + radius >= n {
            bounds.push((start, n));
            break;
        }
        // Snap the cut to the quietest nearby frame, but always move forward by at least a frame.
        let cut = quietest_cut(audio, ideal, radius).max(start + FRAME).min(n);
        bounds.push((start, cut));
        start = cut;
    }
    bounds
}

/// Start index of the quietest [`FRAME`]-sample frame within `radius` of `ideal` — the best place to
/// cut. Energy is the frame's sum of squares; ties keep the earliest (closest to the clip start).
fn quietest_cut(audio: &[f32], ideal: usize, radius: usize) -> usize {
    let lo = ideal.saturating_sub(radius);
    let hi = (ideal + radius).min(audio.len().saturating_sub(FRAME));

    let mut best = ideal;
    let mut best_energy = f32::MAX;
    let mut i = lo;
    while i <= hi {
        let energy: f32 = audio[i..i + FRAME].iter().map(|s| s * s).sum();
        if energy < best_energy {
            best_energy = energy;
            best = i;
        }
        i += FRAME / 2; // half-frame steps, so a pause can't fall entirely between samples
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_audio_has_no_windows() {
        assert!(window_bounds(&[], 1_000, 100).is_empty());
    }

    #[test]
    fn a_short_clip_is_one_window() {
        let audio = vec![0.1f32; 100];
        assert_eq!(window_bounds(&audio, 1_000, 100), vec![(0, 100)]);
        // target 0 also means "don't split".
        assert_eq!(window_bounds(&audio, 0, 100), vec![(0, 100)]);
    }

    #[test]
    fn windows_tile_the_clip_contiguously() {
        let audio = vec![0.5f32; 10_000];
        let bounds = window_bounds(&audio, 3_000, FRAME);
        assert!(bounds.len() >= 3, "a 10k clip in ~3k windows is ≥3 pieces");
        assert_eq!(bounds.first().unwrap().0, 0, "starts at 0");
        assert_eq!(bounds.last().unwrap().1, 10_000, "ends at the clip end");
        for pair in bounds.windows(2) {
            assert_eq!(pair[0].1, pair[1].0, "no gaps or overlaps between windows");
        }
    }

    #[test]
    fn the_cut_snaps_into_a_quiet_gap_off_the_ideal_point() {
        // Loud everywhere except a planted silent gap offset from the ideal 3 000-sample cut.
        let mut audio = vec![0.8f32; 6_000];
        let gap_start = 2_600;
        let gap_len = 640; // two frames, so a whole frame lands inside the silence
        for s in &mut audio[gap_start..gap_start + gap_len] {
            *s = 0.0;
        }

        let cut = window_bounds(&audio, 3_000, 500)[0].1;
        assert!(
            (gap_start..gap_start + gap_len).contains(&cut),
            "cut snapped into the silent gap (got {cut}, gap {gap_start}..{})",
            gap_start + gap_len
        );
        assert_ne!(cut, 3_000, "did not cut at the loud ideal point");
    }
}
