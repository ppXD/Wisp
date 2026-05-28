//! The [`Diarizer`] trait — assigns speaker identities to audio segments.
//!
//! Diarization is a post-MVP capability; the trait exists now so the pipeline can carry a
//! diarizer hook, letting a real implementation land later without breaking changes.

use crate::error::Result;
use crate::transcript::SpeakerId;

/// Assigns a [`SpeakerId`] to a span of 16 kHz mono audio.
pub trait Diarizer: Send {
    /// Returns the speaker the supplied audio most likely belongs to.
    fn identify(&mut self, audio: &[f32], sample_rate: u32) -> Result<SpeakerId>;
}
