//! Streaming transcription pipeline for Wisp.
//!
//! Pulls audio frames from a [`wisp_core::AudioSource`], converts each to 16 kHz mono, segments
//! speech with a [`Vad`], transcribes each utterance through a [`wisp_core::AsrEngine`], and
//! emits [`wisp_core::TranscriptEvent`]s.
//!
//! Every collaborator is a trait object, so the real engine, a richer VAD, or a different audio
//! source drop in without touching this orchestration.

pub mod pipeline;
pub mod segmenter;
pub mod session;
pub mod transcriber;
pub mod vad;

pub use pipeline::{Pipeline, DEFAULT_SILENCE_HANGOVER};
pub use segmenter::{EnergySegmenter, Segmenter, Utterance};
pub use session::{EventSink, Session};
pub use transcriber::Transcriber;
pub use vad::{EnergyVad, Vad};
