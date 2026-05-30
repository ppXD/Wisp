//! Acoustic echo cancellation.

/// Removes far-end echo from a near-end (microphone) signal.
///
/// On speaker playback the microphone re-hears whatever is playing (the "far-end"), so the mic
/// stream carries the speaker's voice *plus* an echo of the system audio. An `EchoCanceller` is fed
/// the far-end **reference** (the system/loopback signal) via [`push_reference`](Self::push_reference),
/// then cleans each near-end frame via [`process_capture`](Self::process_capture), leaving the
/// near-end voice.
///
/// All audio is **16 kHz mono** `f32` in `[-1.0, 1.0]`. Implementations may buffer internally to a
/// fixed processing frame size, so `process_capture` returns the samples that are ready — which can
/// lag the input by less than one internal frame.
pub trait EchoCanceller: Send {
    /// Feeds far-end reference audio — the signal the microphone will re-hear from the speakers.
    fn push_reference(&mut self, samples: &[f32]);

    /// Removes echo of previously-pushed reference audio from near-end `samples`, returning the
    /// cleaned samples that are ready.
    fn process_capture(&mut self, samples: &[f32]) -> Vec<f32>;
}

/// An [`EchoCanceller`] that does nothing: it returns the near-end audio untouched and discards the
/// reference. The portable fallback where no platform AEC is available (Windows/Linux today, or a
/// Mac where WebRTC failed to init) — the cross-stream dedup still drops residual echo, and the
/// pipeline stays uniform across platforms instead of branching on whether an AEC exists.
#[derive(Debug, Default, Clone, Copy)]
pub struct PassthroughEchoCanceller;

impl EchoCanceller for PassthroughEchoCanceller {
    fn push_reference(&mut self, _samples: &[f32]) {}

    fn process_capture(&mut self, samples: &[f32]) -> Vec<f32> {
        samples.to_vec()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn passthrough_returns_capture_unchanged_and_ignores_reference() {
        let mut aec = PassthroughEchoCanceller;
        aec.push_reference(&[0.5, -0.5]); // ignored
        assert_eq!(aec.process_capture(&[0.1, -0.2, 0.3]), vec![0.1, -0.2, 0.3]);
    }
}
