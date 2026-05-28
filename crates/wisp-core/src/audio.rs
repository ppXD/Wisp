//! Audio frames and the [`AudioSource`] trait.
//!
//! An [`AudioSource`] is any producer of PCM audio — a microphone, a system/loopback capture,
//! or a file reader. New sources plug in by implementing this trait; nothing else changes.

use std::time::Duration;

use crate::error::Result;
use crate::transcript::AudioSourceKind;

/// A chunk of PCM audio: interleaved `f32` samples in the range `[-1.0, 1.0]`.
#[derive(Debug, Clone, PartialEq)]
pub struct AudioFrame {
    /// Interleaved samples (one value per channel at each sample instant).
    pub samples: Vec<f32>,
    /// Samples per second per channel (e.g. 16_000, 48_000).
    pub sample_rate: u32,
    /// Number of interleaved channels (1 = mono, 2 = stereo).
    pub channels: u16,
    /// Offset of this frame from the start of the stream.
    pub timestamp: Duration,
}

impl AudioFrame {
    /// Creates a new frame.
    pub fn new(samples: Vec<f32>, sample_rate: u32, channels: u16, timestamp: Duration) -> Self {
        Self {
            samples,
            sample_rate,
            channels,
            timestamp,
        }
    }

    /// Number of sample instants — i.e. samples per channel.
    pub fn frame_count(&self) -> usize {
        let channels = self.channels.max(1) as usize;
        self.samples.len() / channels
    }

    /// Duration of audio contained in this frame.
    pub fn duration(&self) -> Duration {
        if self.sample_rate == 0 {
            return Duration::ZERO;
        }
        Duration::from_secs_f64(self.frame_count() as f64 / self.sample_rate as f64)
    }

    /// True when the frame carries no samples.
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }
}

/// Describes an [`AudioSource`] for display and routing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AudioSourceInfo {
    /// Which kind of source this is (microphone, system audio, file…).
    pub kind: AudioSourceKind,
    /// Human-readable name (e.g. the device name).
    pub name: String,
}

/// A producer of [`AudioFrame`]s.
///
/// Implementations run on a dedicated capture thread; [`next_frame`](AudioSource::next_frame)
/// blocks until audio is available and returns `Ok(None)` when the stream ends.
pub trait AudioSource: Send {
    /// Metadata describing this source.
    fn info(&self) -> AudioSourceInfo;

    /// Blocks for and returns the next frame, or `Ok(None)` at end of stream.
    fn next_frame(&mut self) -> Result<Option<AudioFrame>>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_count_divides_by_channels() {
        let f = AudioFrame::new(vec![0.0; 8], 16_000, 2, Duration::ZERO);
        assert_eq!(f.frame_count(), 4);
    }

    #[test]
    fn duration_matches_sample_rate() {
        let f = AudioFrame::new(vec![0.0; 16_000], 16_000, 1, Duration::ZERO);
        assert_eq!(f.duration(), Duration::from_secs(1));
    }

    #[test]
    fn zero_sample_rate_is_zero_duration() {
        let f = AudioFrame::new(vec![0.0; 10], 0, 1, Duration::ZERO);
        assert_eq!(f.duration(), Duration::ZERO);
    }

    #[test]
    fn empty_frame_detected() {
        let f = AudioFrame::new(vec![], 16_000, 1, Duration::ZERO);
        assert!(f.is_empty());
        assert_eq!(f.frame_count(), 0);
    }
}
