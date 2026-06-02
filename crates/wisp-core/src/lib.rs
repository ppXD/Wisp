//! Wisp core — generic, engine-agnostic primitives for local, real-time transcription.
//!
//! This crate holds Wisp's pluggable backbone (audio sources, ASR engines, diarizers, model
//! stores) behind narrow traits, with **no native dependencies**, so the logic is fully
//! unit-testable and new capabilities slot in without breaking changes.
//!
//! The architecture lands incrementally via small PRs — see `CLAUDE.md` at the repo root.

pub mod aec;
pub mod align;
pub mod audio;
pub mod channel;
pub mod cloud;
pub mod config;
pub mod dedup;
pub mod denoise;
pub mod diarize;
pub mod engine;
pub mod error;
pub mod export;
pub mod model;
pub mod params;
pub mod perf;
pub mod transcript;

#[cfg(any(test, feature = "testing"))]
pub mod testing;

pub use aec::EchoCanceller;
pub use align::{merge_tokens_into_words, TokenTiming};
pub use audio::{AudioFrame, AudioSource, AudioSourceInfo};
pub use channel::{frame_channel, FrameReceiver, FrameSender};
pub use config::Config;
pub use dedup::CrossStreamEchoFilter;
pub use denoise::Denoiser;
pub use diarize::{assign_speakers, ClipDiarizer, Diarizer, SpeakerSpan};
pub use engine::{
    AsrEngine, ClipOptions, EngineInfo, StreamingAsrEngine, StreamingResult, TranscriptionResult,
};
pub use error::{Result, WispError};
pub use export::{format_transcript, group_paragraphs, ExportFormat, Paragraph};
pub use model::{ModelDescriptor, ModelFamily, ModelFile, ModelId, ModelStore, Quant};
pub use params::{EnumOption, ParamKind, ParamSpec, ParamValue, ParamValues};
pub use transcript::{
    AudioSourceKind, SegmentStatus, SpeakerId, TranscriptEvent, TranscriptSegment, Word,
};

/// The `wisp-core` crate version, surfaced to the app and UI.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_is_populated() {
        assert!(!VERSION.is_empty(), "crate version must be set by Cargo");
    }
}
