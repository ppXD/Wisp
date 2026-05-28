//! Model catalog, download, verification, and on-disk storage for Wisp.
//!
//! Implements [`wisp_core::ModelStore`] over the local filesystem: it downloads a model's files
//! through a pluggable [`FileDownloader`], verifies each against its SHA-256, stores them
//! atomically, and writes a completion marker so a repeated `ensure` is a no-op.
//!
//! The catalog *data* (which models exist, their URLs and checksums) is injected into
//! [`FsModelStore`]; sourcing a built-in catalog is handled separately.

pub mod checksum;
pub mod download;
pub mod store;

pub use download::FileDownloader;
#[cfg(feature = "http")]
pub use download::HttpDownloader;
pub use store::FsModelStore;
