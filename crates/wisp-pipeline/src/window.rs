//! Splitting a clip into transcription windows that break at quiet points, and transcribing it
//! window by window.
//!
//! Engines without a native progress callback (the sherpa offline recognizers) decode a whole clip
//! in one opaque call, so the only way to show a real percentage is to transcribe the clip in pieces
//! and report after each. A naive cut every N seconds can slice a word in half; [`window_bounds`]
//! snaps each cut to the quietest short frame nearby, so windows break in pauses instead of mid-word.
//! [`transcribe_in_windows`] drives an engine over those windows and stitches the result back onto
//! the clip timeline.

use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use wisp_core::engine::{AsrEngine, ClipOptions};
use wisp_core::error::Result;
use wisp_core::transcript::TranscriptSegment;

/// 16 kHz mono — the rate the whole pipeline runs at.
const SAMPLE_RATE: usize = 16_000;

/// Target transcription-window length: Whisper's native 30 s window (and SenseVoice's too).
const WINDOW_SECS: usize = 30;

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

/// Transcribes `audio` in windows that break at quiet points (see [`window_bounds`]), offsetting
/// each window's segment and word timestamps onto the clip timeline and reporting progress after
/// each window. For engines that decode a whole clip in one opaque call (the sherpa offline
/// recognizers) and so can't report their own progress; whisper.cpp uses its native long-form path
/// instead. SenseVoice is non-autoregressive (no cross-window context to lose), so windowing barely
/// affects accuracy and yields finer segments than the single block it would otherwise produce.
pub fn transcribe_in_windows(
    engine: &mut dyn AsrEngine,
    audio: &[f32],
    sample_rate: u32,
    clip: ClipOptions,
    cancel: &AtomicBool,
    on_progress: &dyn Fn(u8),
) -> Result<Vec<TranscriptSegment>> {
    let target = sample_rate as usize * WINDOW_SECS;
    let radius = sample_rate as usize / 2; // snap each cut within ±0.5 s of the ideal
    let bounds = window_bounds(audio, target, radius);
    let total = bounds.len().max(1);

    let mut collected = Vec::new();
    for (i, &(start, end)) in bounds.iter().enumerate() {
        // A cancelled File transcription stops at the next window boundary, returning what was
        // decoded so far; the caller owns the flag and discards the partial.
        if cancel.load(Ordering::Relaxed) {
            break;
        }
        let offset = Duration::from_secs_f64(start as f64 / sample_rate as f64);
        let result = engine.transcribe_clip(&audio[start..end], sample_rate, clip)?;
        for mut segment in result.segments {
            segment.start += offset;
            segment.end += offset;
            for word in &mut segment.words {
                word.start += offset;
                word.end += offset;
            }
            collected.push(segment);
        }
        on_progress(((i + 1) * 100 / total) as u8);
    }
    Ok(collected)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::{Cell, RefCell};
    use wisp_core::engine::{EngineInfo, TranscriptionResult};
    use wisp_core::transcript::{AudioSourceKind, Word};

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

    /// A mock engine returning one segment per call spanning the window it was handed — in
    /// window-local time (`0..len`), with one word over the same span, tagged by call index — so a
    /// test can check each window's offset reaches both the segment and its words.
    struct PerWindowEngine {
        calls: Cell<u32>,
    }

    impl AsrEngine for PerWindowEngine {
        fn info(&self) -> EngineInfo {
            EngineInfo {
                name: "mock".to_owned(),
                streaming: false,
            }
        }

        fn transcribe(&mut self, audio: &[f32], sample_rate: u32) -> Result<TranscriptionResult> {
            let n = self.calls.get();
            self.calls.set(n + 1);
            let span =
                Duration::ZERO..Duration::from_secs_f32(audio.len() as f32 / sample_rate as f32);
            let mut segment =
                TranscriptSegment::new(0, format!("w{n}"), span.clone(), AudioSourceKind::File);
            segment.words = vec![Word {
                text: format!("w{n}"),
                start: span.start,
                end: span.end,
            }];
            Ok(TranscriptionResult {
                segments: vec![segment],
            })
        }
    }

    #[test]
    fn windowed_transcription_offsets_each_window_onto_the_clip_timeline() {
        // sample_rate 10 → a 30 s window is 300 samples, so a 1 000-sample clip spans several
        // windows. The engine returns window-local timestamps; transcribe_in_windows must offset
        // each onto the clip timeline so segments tile forward without overlap.
        let mut engine = PerWindowEngine {
            calls: Cell::new(0),
        };
        let audio = vec![0.3f32; 1_000];
        let pcts = RefCell::new(Vec::new());

        let cancel = AtomicBool::new(false);
        let segs = transcribe_in_windows(
            &mut engine,
            &audio,
            10,
            ClipOptions::new(true, false, ""),
            &cancel,
            &|p| pcts.borrow_mut().push(p),
        )
        .unwrap();

        assert!(
            segs.len() >= 3,
            "1 000 samples in ~300-sample windows → ≥3 segments"
        );
        assert_eq!(
            segs[0].start,
            Duration::ZERO,
            "the first window starts at 0"
        );
        for pair in segs.windows(2) {
            assert!(
                pair[1].start >= pair[0].end,
                "windows are offset in capture order with no overlap"
            );
        }
        // The word timestamps are offset alongside their segment.
        assert_eq!(segs[0].words[0].start, segs[0].start);
        let last = segs.last().unwrap();
        assert_eq!(last.words[0].end, last.end);
        assert_eq!(*pcts.borrow().last().unwrap(), 100, "progress ends at 100%");
    }

    #[test]
    fn windowed_transcription_of_a_sub_window_clip_is_one_tick_at_100() {
        let mut engine = PerWindowEngine {
            calls: Cell::new(0),
        };
        let audio = vec![0.3f32; 50]; // smaller than one 300-sample window
        let pcts = RefCell::new(Vec::new());

        let cancel = AtomicBool::new(false);
        let segs = transcribe_in_windows(
            &mut engine,
            &audio,
            10,
            ClipOptions::new(false, false, ""),
            &cancel,
            &|p| pcts.borrow_mut().push(p),
        )
        .unwrap();

        assert_eq!(segs.len(), 1);
        assert_eq!(segs[0].start, Duration::ZERO);
        assert_eq!(*pcts.borrow(), vec![100]);
    }

    #[test]
    fn a_pre_cancelled_run_decodes_nothing() {
        let mut engine = PerWindowEngine {
            calls: Cell::new(0),
        };
        let audio = vec![0.3f32; 1_000]; // several 300-sample windows
        let cancel = AtomicBool::new(true); // already cancelled before the first window
        let segs = transcribe_in_windows(
            &mut engine,
            &audio,
            10,
            ClipOptions::new(false, false, ""),
            &cancel,
            &|_| {},
        )
        .unwrap();
        assert!(
            segs.is_empty(),
            "nothing is decoded when cancelled up front"
        );
        assert_eq!(engine.calls.get(), 0, "the engine was never called");
    }

    #[test]
    fn cancelling_mid_run_stops_at_the_next_window_boundary() {
        let mut engine = PerWindowEngine {
            calls: Cell::new(0),
        };
        let audio = vec![0.3f32; 1_000];
        // The progress callback fires after each window; flip the flag once the first one completes.
        let cancel = AtomicBool::new(false);
        let segs = transcribe_in_windows(
            &mut engine,
            &audio,
            10,
            ClipOptions::new(false, false, ""),
            &cancel,
            &|_| {
                cancel.store(true, Ordering::Relaxed);
            },
        )
        .unwrap();
        assert_eq!(
            segs.len(),
            1,
            "decoded one window, then stopped at the boundary"
        );
        assert_eq!(
            engine.calls.get(),
            1,
            "the engine decoded exactly one window"
        );
    }
}
