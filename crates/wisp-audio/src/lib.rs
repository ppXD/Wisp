//! Audio capture sources and DSP for Wisp.
//!
//! Provides conversion to the engine's 16 kHz mono input ([`dsp`]) and a file-backed
//! [`WavSource`]. Live capture (microphone, system/loopback) plugs in behind
//! [`wisp_core::AudioSource`] and is added with the application, where it can be verified on
//! real hardware.

pub mod denoise;
pub mod dsp;
pub mod echo;
#[cfg(feature = "file")]
pub mod media;
#[cfg(feature = "mic")]
pub mod mic;
pub mod mixer;
pub mod preprocess;
pub mod stream;
pub mod wav;

pub use denoise::RnnoiseDenoiser;
pub use dsp::{
    chunk_into_frames, resample_linear, to_mono_16k, Resampler, FRAME_CHUNK_MS, TARGET_SAMPLE_RATE,
};
pub use echo::EchoCancellingSource;
#[cfg(feature = "file")]
pub use media::MediaSource;
#[cfg(feature = "mic")]
pub use mic::{list_input_devices, MicSource};
pub use mixer::MeetingMixer;
pub use preprocess::normalize_for_asr;
pub use stream::{tee, ChannelSource, Tee};
pub use wav::WavSource;
