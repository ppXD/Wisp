//! SQLite-backed meeting knowledge base for Wisp.
//!
//! Persists each finished meeting (metadata + finalized transcript segments) and indexes it for
//! full-text search. SQLite is the source of truth; Markdown is rendered on demand from the same
//! [`wisp_core`] domain types (via [`wisp_core::export::format_markdown`]). The store is engine- and
//! shell-agnostic — the desktop app is one caller, a CLI or sync agent could be another.
//!
//! This is the persistence + full-text foundation; semantic search and RAG build on it later.

mod record;
mod store;

pub use record::{Note, NoteSummary, SearchHit, Segment};
pub use store::Library;

/// An error from the meeting library.
#[derive(Debug, thiserror::Error)]
pub enum LibraryError {
    /// The underlying SQLite database returned an error.
    #[error("library database error: {0}")]
    Db(#[from] rusqlite::Error),
}

/// Result of a library operation.
pub type Result<T> = std::result::Result<T, LibraryError>;
