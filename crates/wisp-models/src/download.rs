//! Pluggable file downloading.

use std::path::Path;

use wisp_core::error::Result;

/// Downloads a single file from a URL to a local path.
///
/// Abstracted so the store is driven by a real HTTP client in production and by a fake in tests,
/// keeping the test suite hermetic (no network access).
pub trait FileDownloader: Send + Sync {
    /// Downloads `url`, writing its bytes to `dest` (created or overwritten).
    fn download(&self, url: &str, dest: &Path) -> Result<()>;
}

/// A [`FileDownloader`] backed by `ureq` over HTTPS.
#[cfg(feature = "http")]
#[derive(Debug, Default)]
pub struct HttpDownloader;

#[cfg(feature = "http")]
impl FileDownloader for HttpDownloader {
    fn download(&self, url: &str, dest: &Path) -> Result<()> {
        use std::fs::File;
        use wisp_core::error::WispError;

        let response = ureq::get(url)
            .call()
            .map_err(|e| WispError::Model(format!("download {url}: {e}")))?;

        let mut reader = response.into_reader();
        let mut file = File::create(dest)?;
        std::io::copy(&mut reader, &mut file)?;
        Ok(())
    }
}
