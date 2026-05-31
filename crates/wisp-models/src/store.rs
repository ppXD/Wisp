//! Filesystem-backed [`ModelStore`].

use std::fs;
use std::path::{Path, PathBuf};

use wisp_core::error::{Result, WispError};
use wisp_core::model::{ModelDescriptor, ModelFile, ModelId, ModelStore};

use crate::checksum::verify_file;
use crate::coreml::CoremlAsset;
use crate::download::FileDownloader;

/// Marker file written into a model's directory once every file is downloaded and verified.
const COMPLETE_MARKER: &str = ".wisp-ok";

/// A [`ModelStore`] that keeps each model in `root/<model-id>/`, downloading missing files
/// through an injected [`FileDownloader`] and verifying each file's SHA-256 (or, for an unpinned
/// file, at least its size).
///
/// Downloads are atomic: each file lands in a `*.part` temporary, is verified, and only then
/// renamed into place — and a failed transfer removes its `*.part`, so an interrupted run never
/// leaves a partial or corrupt file that looks valid.
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

    /// Returns the local directory for `id` if it is fully installed, else `None`.
    pub fn local_path(&self, id: &ModelId) -> Option<PathBuf> {
        if self.is_complete(id) {
            Some(self.model_dir(id))
        } else {
            None
        }
    }

    /// Verifies `path` against `file`'s checksum. An empty checksum means "unpinned" (the whole
    /// catalog today): there is no hash to match, so fall back to a size check — at least reject a
    /// truncated/short transfer instead of accepting it as a complete model.
    fn verify(&self, path: &Path, file: &ModelFile) -> Result<()> {
        if file.sha256.is_empty() {
            let len = fs::metadata(path)?.len();
            if len < file.size_bytes {
                return Err(WispError::Model(format!(
                    "{}: got {len} bytes, expected at least {} — incomplete download",
                    file.name, file.size_bytes
                )));
            }
            return Ok(());
        }
        verify_file(path, &file.sha256)
    }

    fn fetch_file(
        &self,
        dir: &Path,
        file: &ModelFile,
        on_bytes: &mut dyn FnMut(u64),
    ) -> Result<()> {
        let dest = dir.join(&file.name);

        if dest.is_file() && self.verify(&dest, file).is_ok() {
            on_bytes(file.size_bytes); // already present — count it as done
            return Ok(());
        }

        if file.sha256.is_empty() {
            eprintln!(
                "wisp-models: downloading {} without a pinned checksum (size-checked only)",
                file.name
            );
        }

        let part = dir.join(format!("{}.part", file.name));
        if let Err(e) = self
            .downloader
            .download_with_progress(&file.url, &part, on_bytes)
        {
            let _ = fs::remove_file(&part); // don't leave a partial transfer behind
            return Err(e);
        }

        if let Err(e) = self.verify(&part, file) {
            let _ = fs::remove_file(&part);
            return Err(e);
        }

        fs::rename(&part, &dest)?;
        Ok(())
    }

    /// Like [`ensure`](ModelStore::ensure) but reports download progress as `(downloaded, total)`
    /// bytes across all of the model's files, so a caller can drive a progress bar.
    pub fn ensure_with_progress(
        &self,
        id: &ModelId,
        on_progress: &mut dyn FnMut(u64, u64),
    ) -> Result<PathBuf> {
        let descriptor = self.descriptor(id)?;
        let dir = self.model_dir(id);
        let total = descriptor.total_size_bytes();

        if self.is_complete(id) {
            on_progress(total, total);
            return Ok(dir);
        }

        fs::create_dir_all(&dir)?;

        let mut done = 0u64;
        for file in &descriptor.files {
            self.fetch_file(&dir, file, &mut |file_bytes| {
                on_progress((done + file_bytes).min(total), total);
            })?;
            done += file.size_bytes;
            on_progress(done.min(total), total);
        }

        fs::write(dir.join(COMPLETE_MARKER), id.as_str())?;
        on_progress(total, total);
        Ok(dir)
    }

    /// Whether the Core ML encoder for `asset` is already unpacked next to the model.
    pub fn coreml_installed(&self, id: &ModelId, asset: &CoremlAsset) -> bool {
        self.model_dir(id).join(&asset.dir_name).is_dir()
    }

    /// Downloads and unpacks the optional Core ML encoder for `id` next to the model, so whisper.cpp
    /// runs the encoder on the Neural Engine. The model itself must already be installed. Reports
    /// `(downloaded, total)` like [`ensure_with_progress`](Self::ensure_with_progress); idempotent —
    /// a no-op once the directory is present.
    pub fn ensure_coreml_with_progress(
        &self,
        id: &ModelId,
        asset: &CoremlAsset,
        on_progress: &mut dyn FnMut(u64, u64),
    ) -> Result<()> {
        let total = asset.size_bytes;
        let dir = self.model_dir(id);

        if dir.join(&asset.dir_name).is_dir() {
            on_progress(total, total);
            return Ok(());
        }
        if !self.is_complete(id) {
            return Err(WispError::Model(format!(
                "install the model {} before its Core ML encoder",
                id.as_str()
            )));
        }

        let zip_part = dir.join(format!("{}.zip.part", asset.dir_name));
        if let Err(e) =
            self.downloader
                .download_with_progress(&asset.url, &zip_part, &mut |bytes| {
                    on_progress(bytes.min(total), total);
                })
        {
            let _ = fs::remove_file(&zip_part);
            return Err(e);
        }

        if let Err(e) = unzip_into(&zip_part, &dir) {
            let _ = fs::remove_file(&zip_part);
            return Err(e);
        }
        let _ = fs::remove_file(&zip_part);

        on_progress(total, total);
        Ok(())
    }
}

/// Extracts every entry of the zip at `zip_path` into `dest`, creating directories as needed and
/// skipping any entry whose path would escape `dest` (zip-slip). Used to unpack a Core ML
/// `.mlmodelc` directory next to its model.
fn unzip_into(zip_path: &Path, dest: &Path) -> Result<()> {
    let reader = fs::File::open(zip_path)?;
    let mut archive = zip::ZipArchive::new(reader)
        .map_err(|e| WispError::Model(format!("open Core ML archive: {e}")))?;

    for i in 0..archive.len() {
        let mut entry = archive
            .by_index(i)
            .map_err(|e| WispError::Model(format!("read Core ML archive entry: {e}")))?;
        let Some(rel) = entry.enclosed_name() else {
            continue; // unsafe path — skip rather than escape `dest`
        };

        let out = dest.join(rel);
        if entry.is_dir() {
            fs::create_dir_all(&out)?;
        } else {
            if let Some(parent) = out.parent() {
                fs::create_dir_all(parent)?;
            }
            let mut writer = fs::File::create(&out)?;
            std::io::copy(&mut entry, &mut writer)?;
        }
    }
    Ok(())
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
        self.ensure_with_progress(id, &mut |_, _| {})
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
            self.download_with_progress(url, dest, &mut |_| {})
        }

        fn download_with_progress(
            &self,
            url: &str,
            dest: &Path,
            on_bytes: &mut dyn FnMut(u64),
        ) -> Result<()> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            let bytes = self
                .files
                .get(url)
                .ok_or_else(|| WispError::Model(format!("no fake for {url}")))?;
            std::fs::write(dest, bytes)?;
            // Report in two steps to exercise cumulative progress accounting.
            on_bytes((bytes.len() / 2) as u64);
            on_bytes(bytes.len() as u64);
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
            description: String::new(),
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
    fn unpinned_complete_file_installs() {
        // An unpinned file (empty checksum) whose download arrives complete passes the size check
        // and installs normally.
        let url = "https://example/u.bin";
        let bytes = b"unpinned".to_vec();
        let mut desc = single_file_descriptor("u", url, "u.bin", &bytes);
        desc.files[0].sha256 = String::new(); // unpinned
        let (downloader, calls) = fake_for(url, &bytes);
        let root = tempfile::tempdir().unwrap();
        let store = FsModelStore::new(root.path(), vec![desc], Box::new(downloader));

        let dir = store.ensure(&ModelId("u".into())).unwrap();
        assert!(dir.join("u.bin").is_file());
        assert!(store.local_path(&ModelId("u".into())).is_some());
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn unpinned_download_rejected_when_truncated() {
        // Unpinned, so there is no checksum — a short transfer must still be rejected on size, not
        // accepted as a complete model.
        let url = "https://example/t.bin";
        let mut desc = single_file_descriptor("t", url, "t.bin", b"short");
        desc.files[0].sha256 = String::new(); // unpinned
        desc.files[0].size_bytes = 100; // but 100 expected → the 5 served bytes are truncated
        let (downloader, _calls) = fake_for(url, b"short");
        let root = tempfile::tempdir().unwrap();
        let store = FsModelStore::new(root.path(), vec![desc], Box::new(downloader));

        let err = store.ensure(&ModelId("t".into())).unwrap_err();
        assert!(matches!(err, WispError::Model(_)));

        let dir = root.path().join("t");
        assert!(!dir.join(COMPLETE_MARKER).exists());
        assert!(!dir.join("t.bin").exists());
        assert!(
            !dir.join("t.bin.part").exists(),
            "the .part must be cleaned up"
        );
    }

    /// A downloader that writes a partial file and then fails — mimics a connection dropped
    /// mid-transfer, which leaves a `*.part` the store must clean up.
    struct PartialThenError;
    impl FileDownloader for PartialThenError {
        fn download(&self, url: &str, dest: &Path) -> Result<()> {
            self.download_with_progress(url, dest, &mut |_| {})
        }
        fn download_with_progress(
            &self,
            _url: &str,
            dest: &Path,
            _on_bytes: &mut dyn FnMut(u64),
        ) -> Result<()> {
            std::fs::write(dest, b"partial").unwrap();
            Err(WispError::Model("connection dropped".into()))
        }
    }

    #[test]
    fn download_error_removes_the_part_file() {
        let desc = single_file_descriptor("e", "https://example/e.bin", "e.bin", b"whatever");
        let root = tempfile::tempdir().unwrap();
        let store = FsModelStore::new(root.path(), vec![desc], Box::new(PartialThenError));

        let err = store.ensure(&ModelId("e".into())).unwrap_err();
        assert!(matches!(err, WispError::Model(_)));

        let dir = root.path().join("e");
        assert!(
            !dir.join("e.bin.part").exists(),
            "a failed download must not leave its .part behind"
        );
        assert!(!dir.join(COMPLETE_MARKER).exists());
    }

    #[test]
    fn ensure_unknown_model_errors() {
        let (downloader, _calls) = FakeDownloader::new(HashMap::new());
        let root = tempfile::tempdir().unwrap();
        let store = FsModelStore::new(root.path(), vec![], Box::new(downloader));
        assert!(store.ensure(&ModelId("nope".into())).is_err());
    }

    #[test]
    fn ensure_with_progress_reports_cumulative_bytes() {
        let a = b"aaaaaaaa".to_vec(); // 8 bytes
        let b = b"bbbb".to_vec(); // 4 bytes
        let total = (a.len() + b.len()) as u64;

        let mut files = HashMap::new();
        files.insert("https://x/a".to_string(), a.clone());
        files.insert("https://x/b".to_string(), b.clone());
        let (downloader, _calls) = FakeDownloader::new(files);

        let desc = ModelDescriptor {
            id: ModelId("m".into()),
            family: ModelFamily::SenseVoice,
            quant: Quant::Q8,
            display_name: "m".into(),
            files: vec![
                ModelFile {
                    name: "a.onnx".into(),
                    url: "https://x/a".into(),
                    sha256: String::new(),
                    size_bytes: a.len() as u64,
                },
                ModelFile {
                    name: "b.onnx".into(),
                    url: "https://x/b".into(),
                    sha256: String::new(),
                    size_bytes: b.len() as u64,
                },
            ],
            languages: vec![],
            description: String::new(),
        };

        let root = tempfile::tempdir().unwrap();
        let store = FsModelStore::new(root.path(), vec![desc], Box::new(downloader));

        let mut updates: Vec<(u64, u64)> = Vec::new();
        store
            .ensure_with_progress(&ModelId("m".into()), &mut |d, t| updates.push((d, t)))
            .unwrap();

        assert!(!updates.is_empty());
        assert!(
            updates.iter().all(|(_, t)| *t == total),
            "total is constant"
        );
        assert!(
            updates.windows(2).all(|w| w[0].0 <= w[1].0),
            "downloaded is non-decreasing"
        );
        assert_eq!(updates.last().unwrap().0, total, "ends at 100%");
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

    fn make_zip(entries: &[(&str, &[u8])]) -> Vec<u8> {
        use std::io::Write;
        use zip::write::SimpleFileOptions;
        let mut zw = zip::ZipWriter::new(std::io::Cursor::new(Vec::new()));
        let opts = SimpleFileOptions::default();
        for (name, data) in entries {
            zw.start_file(*name, opts).unwrap();
            zw.write_all(data).unwrap();
        }
        zw.finish().unwrap().into_inner()
    }

    #[test]
    fn ensure_coreml_downloads_unpacks_next_to_the_model_and_is_idempotent() {
        let model_url = "https://example/m.bin";
        let model_bytes = b"model".to_vec();
        let desc = single_file_descriptor("m", model_url, "ggml-test-q8_0.bin", &model_bytes);

        // A fake `.mlmodelc.zip` holding one nested file, as the real archives do.
        let coreml_url = "https://example/ggml-test-encoder.mlmodelc.zip";
        let zip_bytes = make_zip(&[("ggml-test-encoder.mlmodelc/coremldata.bin", b"weights")]);
        let asset = CoremlAsset {
            url: coreml_url.into(),
            dir_name: "ggml-test-encoder.mlmodelc".into(),
            size_bytes: zip_bytes.len() as u64,
        };

        let mut files = HashMap::new();
        files.insert(model_url.to_string(), model_bytes);
        files.insert(coreml_url.to_string(), zip_bytes);
        let (downloader, calls) = FakeDownloader::new(files);

        let root = tempfile::tempdir().unwrap();
        let store = FsModelStore::new(root.path(), vec![desc], Box::new(downloader));
        let id = ModelId("m".into());

        // The encoder requires the model itself to be installed first.
        assert!(store
            .ensure_coreml_with_progress(&id, &asset, &mut |_, _| {})
            .is_err());

        store.ensure(&id).unwrap();
        assert!(!store.coreml_installed(&id, &asset));

        store
            .ensure_coreml_with_progress(&id, &asset, &mut |_, _| {})
            .unwrap();

        let dir = store.local_path(&id).unwrap();
        let nested = dir.join("ggml-test-encoder.mlmodelc/coremldata.bin");
        assert!(nested.is_file(), "the .mlmodelc unpacks next to the model");
        assert_eq!(fs::read(&nested).unwrap(), b"weights");
        assert!(
            !dir.join("ggml-test-encoder.mlmodelc.zip.part").exists(),
            "the downloaded archive is cleaned up"
        );
        assert!(store.coreml_installed(&id, &asset));

        // Idempotent: re-running with the directory present does not re-download.
        let downloads = calls.load(Ordering::SeqCst);
        store
            .ensure_coreml_with_progress(&id, &asset, &mut |_, _| {})
            .unwrap();
        assert_eq!(calls.load(Ordering::SeqCst), downloads);
    }
}
