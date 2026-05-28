//! Wisp core — generic, engine-agnostic primitives for local, real-time transcription.
//!
//! This crate holds Wisp's pluggable backbone (audio sources, ASR engines, diarizers, model
//! stores) behind narrow traits, with **no native dependencies**, so the logic is fully
//! unit-testable and new capabilities slot in without breaking changes.
//!
//! The architecture lands incrementally via small PRs — see `CLAUDE.md` at the repo root.

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
