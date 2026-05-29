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
