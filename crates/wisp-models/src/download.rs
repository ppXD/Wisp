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

    /// Like [`download`](Self::download) but reports the cumulative bytes written so far via
    /// `on_bytes` as the transfer proceeds. The default ignores progress and delegates to
    /// [`download`](Self::download); real downloaders override it to stream.
    fn download_with_progress(
        &self,
        url: &str,
        dest: &Path,
        on_bytes: &mut dyn FnMut(u64),
    ) -> Result<()> {
        let _ = on_bytes;
        self.download(url, dest)
    }
}

/// A [`FileDownloader`] backed by `ureq` over HTTPS.
#[cfg(feature = "http")]
#[derive(Debug, Default)]
pub struct HttpDownloader;

#[cfg(feature = "http")]
impl FileDownloader for HttpDownloader {
    fn download(&self, url: &str, dest: &Path) -> Result<()> {
        self.download_with_progress(url, dest, &mut |_| {})
    }

    fn download_with_progress(
        &self,
        url: &str,
        dest: &Path,
        on_bytes: &mut dyn FnMut(u64),
    ) -> Result<()> {
        use std::fs::File;
        use std::io::{Read, Write};
        use wisp_core::error::WispError;

        let response = ureq::get(url)
            .call()
            .map_err(|e| WispError::Model(format!("download {url}: {e}")))?;

        let mut reader = response.into_reader();
        let mut file = File::create(dest)?;
        let mut buf = vec![0u8; 64 * 1024];
        let mut written = 0u64;

        loop {
            let n = reader.read(&mut buf)?;
            if n == 0 {
                break;
            }
            file.write_all(&buf[..n])?;
            written += n as u64;
            on_bytes(written);
        }
        Ok(())
    }
}
