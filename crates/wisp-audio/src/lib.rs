//! Audio capture sources and DSP for Wisp.
//!
//! Provides conversion to the engine's 16 kHz mono input ([`dsp`]) and a file-backed
//! [`WavSource`]. Live capture (microphone, system/loopback) plugs in behind
//! [`wisp_core::AudioSource`] and is added with the application, where it can be verified on
//! real hardware.

pub mod dsp;
pub mod wav;

pub use dsp::{resample_linear, to_mono_16k, TARGET_SAMPLE_RATE};
pub use wav::WavSource;
