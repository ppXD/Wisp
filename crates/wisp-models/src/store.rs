//! Filesystem-backed [`ModelStore`].

use std::fs;
use std::path::{Path, PathBuf};

use wisp_core::error::{Result, WispError};
use wisp_core::model::{ModelDescriptor, ModelFile, ModelId, ModelStore};

use crate::checksum::verify_file;
use crate::download::FileDownloader;

/// Marker file written into a model's directory once every file is downloaded and verified.
const COMPLETE_MARKER: &str = ".wisp-ok";

/// A [`ModelStore`] that keeps each model in `root/<model-id>/`, downloading missing files
/// through an injected [`FileDownloader`] and verifying every file's SHA-256.
///
/// Downloads are atomic: each file lands in a `*.part` temporary, is verified, and only then
/// renamed into place. An interrupted run therefore never leaves a corrupt file that looks valid.
pub struct FsModelStore {
    root: PathBuf,
    catalog: Vec<ModelDescriptor>,
    downloader: Box<dyn FileDownloader>,
}

impl FsModelStore {
    /// Creates a store rooted at `root`, offering `catalog`, fetching through `downloader`.
    pub fn new(
        root: impl Into<PathBuf>,
        catalog: Vec<ModelDescriptor>,
        downloader: Box<dyn FileDownloader>,
    ) -> Self {
        Self {
            root: root.into(),
            catalog,
            downloader,
        }
    }

    fn descriptor(&self, id: &ModelId) -> Result<&ModelDescriptor> {
        self.catalog
            .iter()
            .find(|d| d.id == *id)
            .ok_or_else(|| WispError::Model(format!("unknown model {}", id.as_str())))
    }

    fn model_dir(&self, id: &ModelId) -> PathBuf {
        self.root.join(id.as_str())
    }

    fn is_complete(&self, id: &ModelId) -> bool {
        self.model_dir(id).join(COMPLETE_MARKER).is_file()
    }

    fn fetch_file(&self, dir: &Path, file: &ModelFile) -> Result<()> {
        let dest = dir.join(&file.name);

        if dest.is_file() && verify_file(&dest, &file.sha256).is_ok() {
            return Ok(());
        }

        let part = dir.join(format!("{}.part", file.name));
        self.downloader.download(&file.url, &part)?;

        if let Err(e) = verify_file(&part, &file.sha256) {
            let _ = fs::remove_file(&part);
            return Err(e);
        }

        fs::rename(&part, &dest)?;
        Ok(())
    }
}

impl ModelStore for FsModelStore {
    fn available(&self) -> Vec<ModelDescriptor> {
        self.catalog.clone()
    }

    fn installed(&self) -> Result<Vec<ModelId>> {
        Ok(self
            .catalog
            .iter()
            .map(|d| d.id.clone())
            .filter(|id| self.is_complete(id))
            .collect())
    }

    fn ensure(&self, id: &ModelId) -> Result<PathBuf> {
        let descriptor = self.descriptor(id)?;
        let dir = self.model_dir(id);

        if self.is_complete(id) {
            return Ok(dir);
        }

        fs::create_dir_all(&dir)?;

        for file in &descriptor.files {
            self.fetch_file(&dir, file)?;
        }

        fs::write(dir.join(COMPLETE_MARKER), id.as_str())?;
        Ok(dir)
    }

    fn remove(&self, id: &ModelId) -> Result<()> {
        let dir = self.model_dir(id);
        if dir.exists() {
            fs::remove_dir_all(&dir)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::checksum::sha256_hex;
    use std::collections::HashMap;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;
    use wisp_core::model::{ModelFamily, Quant};

    /// A [`FileDownloader`] that serves canned bytes per URL and counts its calls — keeps the
    /// store tests hermetic (no network) while exercising the real filesystem logic.
    struct FakeDownloader {
        files: HashMap<String, Vec<u8>>,
        calls: Arc<AtomicUsize>,
    }

    impl FakeDownloader {
        fn new(files: HashMap<String, Vec<u8>>) -> (Self, Arc<AtomicUsize>) {
            let calls = Arc::new(AtomicUsize::new(0));
            (
                Self {
                    files,
                    calls: Arc::clone(&calls),
                },
                calls,
            )
        }
    }

    impl FileDownloader for FakeDownloader {
        fn download(&self, url: &str, dest: &Path) -> Result<()> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            let bytes = self
                .files
                .get(url)
                .ok_or_else(|| WispError::Model(format!("no fake for {url}")))?;
            std::fs::write(dest, bytes)?;
            Ok(())
        }
    }

    fn single_file_descriptor(id: &str, url: &str, name: &str, bytes: &[u8]) -> ModelDescriptor {
        ModelDescriptor {
            id: ModelId(id.into()),
            family: ModelFamily::Whisper,
            quant: Quant::Q5,
            display_name: id.into(),
            files: vec![ModelFile {
                name: name.into(),
                url: url.into(),
                sha256: sha256_hex(bytes),
                size_bytes: bytes.len() as u64,
            }],
            languages: vec![],
        }
    }

    fn fake_for(url: &str, bytes: &[u8]) -> (FakeDownloader, Arc<AtomicUsize>) {
        let mut files = HashMap::new();
        files.insert(url.to_string(), bytes.to_vec());
        FakeDownloader::new(files)
    }

    #[test]
    fn ensure_downloads_verifies_marks_complete_and_is_idempotent() {
        let url = "https://example/m1.bin";
        let bytes = b"hello wisp".to_vec();
        let desc = single_file_descriptor("m1", url, "m1.bin", &bytes);
        let (downloader, calls) = fake_for(url, &bytes);

        let root = tempfile::tempdir().unwrap();
        let store = FsModelStore::new(root.path(), vec![desc], Box::new(downloader));

        assert!(store.installed().unwrap().is_empty());

        let dir = store.ensure(&ModelId("m1".into())).unwrap();
        assert!(dir.join("m1.bin").is_file());
        assert!(dir.join(COMPLETE_MARKER).is_file());
        assert_eq!(std::fs::read(dir.join("m1.bin")).unwrap(), bytes);
        assert_eq!(calls.load(Ordering::SeqCst), 1);

        // Second ensure short-circuits on the completion marker — no extra download.
        store.ensure(&ModelId("m1".into())).unwrap();
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert_eq!(store.installed().unwrap(), vec![ModelId("m1".into())]);
    }

    #[test]
    fn ensure_reuses_valid_existing_file_without_redownload() {
        // A valid file is already on disk but the completion marker is missing (e.g. a prior run
        // was interrupted just before marking). ensure should reuse it, not re-download.
        let url = "https://example/m1.bin";
        let bytes = b"hello wisp".to_vec();
        let desc = single_file_descriptor("m1", url, "m1.bin", &bytes);
        let (downloader, calls) = fake_for(url, &bytes);

        let root = tempfile::tempdir().unwrap();
        let dir = root.path().join("m1");
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("m1.bin"), &bytes).unwrap();

        let store = FsModelStore::new(root.path(), vec![desc], Box::new(downloader));
        store.ensure(&ModelId("m1".into())).unwrap();

        assert_eq!(calls.load(Ordering::SeqCst), 0);
        assert!(dir.join(COMPLETE_MARKER).is_file());
    }

    #[test]
    fn ensure_rejects_checksum_mismatch_and_leaves_no_marker_or_file() {
        let url = "https://example/bad.bin";
        // Descriptor expects the hash of b"expected"...
        let desc = single_file_descriptor("bad", url, "bad.bin", b"expected");
        // ...but the downloader serves different bytes.
        let (downloader, _calls) = fake_for(url, b"corrupted");

        let root = tempfile::tempdir().unwrap();
        let store = FsModelStore::new(root.path(), vec![desc], Box::new(downloader));

        let err = store.ensure(&ModelId("bad".into())).unwrap_err();
        assert!(matches!(err, WispError::Model(_)));

        let dir = root.path().join("bad");
        assert!(!dir.join(COMPLETE_MARKER).exists());
        assert!(!dir.join("bad.bin").exists());
        assert!(!dir.join("bad.bin.part").exists());
        assert!(store.installed().unwrap().is_empty());
    }

    #[test]
    fn ensure_unknown_model_errors() {
        let (downloader, _calls) = FakeDownloader::new(HashMap::new());
        let root = tempfile::tempdir().unwrap();
        let store = FsModelStore::new(root.path(), vec![], Box::new(downloader));
        assert!(store.ensure(&ModelId("nope".into())).is_err());
    }

    #[test]
    fn remove_deletes_installed_model() {
        let url = "https://example/m.bin";
        let bytes = b"data".to_vec();
        let desc = single_file_descriptor("m1", url, "m.bin", &bytes);
        let (downloader, _calls) = fake_for(url, &bytes);

        let root = tempfile::tempdir().unwrap();
        let store = FsModelStore::new(root.path(), vec![desc], Box::new(downloader));

        store.ensure(&ModelId("m1".into())).unwrap();
        assert_eq!(store.installed().unwrap().len(), 1);

        store.remove(&ModelId("m1".into())).unwrap();
        assert!(store.installed().unwrap().is_empty());
        assert!(!root.path().join("m1").exists());
    }
}
