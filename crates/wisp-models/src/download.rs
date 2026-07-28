//! Pluggable file downloading.

use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::path::Path;
use std::sync::{Arc, RwLock};
use std::time::Duration;

use wisp_core::error::{Result, WispError};

/// Public Hugging Face endpoint commonly reachable from mainland China. Users can replace this
/// with an organisation-controlled mirror in Settings.
pub const DEFAULT_HF_MIRROR: &str = "https://hf-mirror.com";

/// How Hugging Face downloads choose between the official endpoint and the configured mirror.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum DownloadSource {
    /// Try the endpoint that last succeeded first, failing over automatically.
    #[default]
    Auto,
    /// Use the official endpoint only.
    Official,
    /// Try the configured mirror first, then fall back to the official endpoint.
    MirrorFirst,
}

/// Proxy selection for model downloads.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ProxyMode {
    /// Honour `ALL_PROXY`, `HTTPS_PROXY`, and `HTTP_PROXY` when inherited by the app.
    #[default]
    System,
    /// Bypass proxy environment variables.
    Direct,
    /// Route downloads through the explicit HTTP/SOCKS proxy URL.
    Custom,
}

/// Runtime model-download policy. This is intentionally independent from app persistence/UI types,
/// so the downloader stays reusable in tests and other Wisp frontends.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DownloadConfig {
    pub source: DownloadSource,
    pub mirror_url: String,
    pub proxy_mode: ProxyMode,
    pub proxy_url: String,
    pub connect_timeout_secs: u64,
    pub read_timeout_secs: u64,
    pub retries: u8,
}

impl Default for DownloadConfig {
    fn default() -> Self {
        Self {
            source: DownloadSource::Auto,
            mirror_url: DEFAULT_HF_MIRROR.to_owned(),
            proxy_mode: ProxyMode::System,
            proxy_url: String::new(),
            connect_timeout_secs: 8,
            read_timeout_secs: 45,
            retries: 3,
        }
    }
}

impl DownloadConfig {
    pub fn validate(&self) -> Result<()> {
        if !self.mirror_url.starts_with("https://") {
            return Err(WispError::Model(
                "model mirror must use an https:// URL".to_owned(),
            ));
        }
        if self.proxy_mode == ProxyMode::Custom && self.proxy_url.trim().is_empty() {
            return Err(WispError::Model(
                "enter an HTTP or SOCKS proxy URL for custom proxy mode".to_owned(),
            ));
        }
        if self.proxy_mode == ProxyMode::Custom {
            ureq::Proxy::new(self.proxy_url.trim())
                .map_err(|error| WispError::Model(format!("invalid proxy URL: {error}")))?;
        }
        if self.connect_timeout_secs == 0 || self.read_timeout_secs == 0 || self.retries == 0 {
            return Err(WispError::Model(
                "download timeouts and retry count must be greater than zero".to_owned(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug)]
struct DownloadState {
    config: DownloadConfig,
    preferred_mirror: Option<bool>,
}

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

/// A regional, resumable [`FileDownloader`] backed by `ureq` over HTTPS.
#[cfg(feature = "http")]
#[derive(Clone, Debug)]
pub struct HttpDownloader {
    state: Arc<RwLock<DownloadState>>,
}

#[cfg(feature = "http")]
impl Default for HttpDownloader {
    fn default() -> Self {
        Self::with_config(DownloadConfig::default())
            .expect("the built-in download configuration is valid")
    }
}

#[cfg(feature = "http")]
impl HttpDownloader {
    pub fn with_config(config: DownloadConfig) -> Result<Self> {
        config.validate()?;
        Ok(Self {
            state: Arc::new(RwLock::new(DownloadState {
                config,
                preferred_mirror: None,
            })),
        })
    }

    pub fn config(&self) -> Result<DownloadConfig> {
        self.state
            .read()
            .map(|state| state.config.clone())
            .map_err(|_| WispError::Model("download settings lock poisoned".to_owned()))
    }

    pub fn update_config(&self, config: DownloadConfig) -> Result<()> {
        config.validate()?;
        let mut state = self
            .state
            .write()
            .map_err(|_| WispError::Model("download settings lock poisoned".to_owned()))?;
        state.config = config;
        state.preferred_mirror = None;
        Ok(())
    }

    fn candidate_urls(&self, url: &str) -> Vec<String> {
        let Ok(state) = self.state.read() else {
            return vec![url.to_owned()];
        };
        candidate_urls(url, &state.config, state.preferred_mirror)
    }

    fn remember_success(&self, original: &str, successful: &str) {
        if !original.starts_with("https://huggingface.co/") {
            return;
        }
        if let Ok(mut state) = self.state.write() {
            state.preferred_mirror = Some(successful != original);
        }
    }

    fn agent(config: &DownloadConfig) -> Result<ureq::Agent> {
        let mut builder = ureq::AgentBuilder::new()
            .timeout_connect(Duration::from_secs(config.connect_timeout_secs))
            .timeout_read(Duration::from_secs(config.read_timeout_secs))
            .timeout_write(Duration::from_secs(config.read_timeout_secs))
            .redirects(10)
            .user_agent(concat!("Wisp/", env!("CARGO_PKG_VERSION")));

        builder = match config.proxy_mode {
            ProxyMode::System => builder.try_proxy_from_env(true),
            ProxyMode::Direct => builder.try_proxy_from_env(false),
            ProxyMode::Custom => {
                let proxy = ureq::Proxy::new(config.proxy_url.trim())
                    .map_err(|error| WispError::Model(format!("invalid proxy URL: {error}")))?;
                builder.proxy(proxy)
            }
        };
        Ok(builder.build())
    }
}

fn candidate_urls(
    url: &str,
    config: &DownloadConfig,
    preferred_mirror: Option<bool>,
) -> Vec<String> {
    let Some(path) = url.strip_prefix("https://huggingface.co/") else {
        return vec![url.to_owned()];
    };
    if config.source == DownloadSource::Official {
        return vec![url.to_owned()];
    }

    let mirror = format!("{}/{}", config.mirror_url.trim_end_matches('/'), path);
    let mirror_first =
        config.source == DownloadSource::MirrorFirst || preferred_mirror == Some(true);
    let mut urls = if mirror_first {
        vec![mirror, url.to_owned()]
    } else {
        vec![url.to_owned(), mirror]
    };
    urls.dedup();
    urls
}

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
        let config = self.config()?;
        let agent = Self::agent(&config)?;
        let candidates = self.candidate_urls(url);
        let mut errors = Vec::new();

        for round in 0..config.retries {
            for candidate in &candidates {
                match transfer_once(&agent, candidate, dest, on_bytes) {
                    Ok(()) => {
                        self.remember_success(url, candidate);
                        return Ok(());
                    }
                    Err(error) => errors.push(format!("{candidate}: {error}")),
                }
            }
            if round + 1 < config.retries {
                let delay = 300u64.saturating_mul(1u64 << round.min(4));
                std::thread::sleep(Duration::from_millis(delay));
            }
        }

        let attempted = errors
            .iter()
            .rev()
            .take(candidates.len().max(1))
            .cloned()
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect::<Vec<_>>()
            .join(" | ");
        Err(WispError::Model(format!(
            "download failed after {} retry rounds; partial data was kept for retry. {attempted}",
            config.retries
        )))
    }
}

#[cfg(feature = "http")]
fn transfer_once(
    agent: &ureq::Agent,
    url: &str,
    dest: &Path,
    on_bytes: &mut dyn FnMut(u64),
) -> Result<()> {
    let offset = dest.metadata().map(|metadata| metadata.len()).unwrap_or(0);
    let mut request = agent.get(url).set("Accept-Encoding", "identity");
    if offset > 0 {
        request = request.set("Range", &format!("bytes={offset}-"));
    }

    let response = match request.call() {
        Ok(response) => response,
        Err(ureq::Error::Status(416, response)) if offset > 0 => {
            let complete = response
                .header("Content-Range")
                .and_then(|value| value.rsplit_once('/'))
                .and_then(|(_, total)| total.parse::<u64>().ok())
                == Some(offset);
            if complete {
                on_bytes(offset);
                return Ok(());
            }
            return Err(WispError::Model(format!(
                "server rejected resume at byte {offset}"
            )));
        }
        Err(error) => return Err(WispError::Model(error.to_string())),
    };

    let append = offset > 0 && response.status() == 206;
    if append {
        let expected_prefix = format!("bytes {offset}-");
        if !response
            .header("Content-Range")
            .is_some_and(|value| value.starts_with(&expected_prefix))
        {
            return Err(WispError::Model(
                "server returned an invalid Content-Range".to_owned(),
            ));
        }
    }

    let expected = response
        .header("Content-Length")
        .and_then(|value| value.parse::<u64>().ok());
    let mut reader = response.into_reader();
    let mut output = if append {
        OpenOptions::new().create(true).append(true).open(dest)?
    } else {
        File::create(dest)?
    };
    let start = if append { offset } else { 0 };
    on_bytes(start);

    let mut buffer = vec![0u8; 64 * 1024];
    let mut written = 0u64;
    loop {
        let count = reader.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        output.write_all(&buffer[..count])?;
        written += count as u64;
        on_bytes(start + written);
    }
    output.flush()?;

    if let Some(total) = expected {
        if total != written {
            return Err(WispError::Model(format!(
                "truncated response ({written} of {total} bytes)"
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::{TcpListener, TcpStream};
    use std::thread;

    fn read_request(stream: &mut TcpStream) -> String {
        let mut bytes = Vec::new();
        let mut buffer = [0u8; 1024];
        while !bytes.ends_with(b"\r\n\r\n") {
            let count = stream.read(&mut buffer).unwrap();
            assert!(count > 0);
            bytes.extend_from_slice(&buffer[..count]);
        }
        String::from_utf8(bytes).unwrap()
    }

    fn direct_config(retries: u8) -> DownloadConfig {
        DownloadConfig {
            source: DownloadSource::Official,
            proxy_mode: ProxyMode::Direct,
            connect_timeout_secs: 2,
            read_timeout_secs: 2,
            retries,
            ..DownloadConfig::default()
        }
    }

    #[test]
    fn automatic_source_falls_back_between_official_and_mirror() {
        let downloader = HttpDownloader::with_config(DownloadConfig::default()).unwrap();
        let official = "https://huggingface.co/org/model/resolve/main/model.onnx";
        let mirror = "https://hf-mirror.com/org/model/resolve/main/model.onnx";
        assert_eq!(downloader.candidate_urls(official), vec![official, mirror]);

        downloader.remember_success(official, mirror);
        assert_eq!(downloader.candidate_urls(official), vec![mirror, official]);
    }

    #[test]
    fn mirror_first_uses_custom_endpoint_then_official() {
        let config = DownloadConfig {
            source: DownloadSource::MirrorFirst,
            mirror_url: "https://models.example.cn/hf".to_owned(),
            ..DownloadConfig::default()
        };
        let downloader = HttpDownloader::with_config(config).unwrap();
        assert_eq!(
            downloader.candidate_urls("https://huggingface.co/org/model/resolve/main/model.onnx"),
            vec![
                "https://models.example.cn/hf/org/model/resolve/main/model.onnx",
                "https://huggingface.co/org/model/resolve/main/model.onnx",
            ]
        );
    }

    #[test]
    fn official_source_and_non_hugging_face_urls_are_not_rewritten() {
        let config = DownloadConfig {
            source: DownloadSource::Official,
            ..DownloadConfig::default()
        };
        let downloader = HttpDownloader::with_config(config).unwrap();
        assert_eq!(
            downloader.candidate_urls("https://huggingface.co/org/model/resolve/main/file.bin"),
            vec!["https://huggingface.co/org/model/resolve/main/file.bin"]
        );
        assert_eq!(
            downloader.candidate_urls("https://github.com/org/repo/releases/download/v1/file.bin"),
            vec!["https://github.com/org/repo/releases/download/v1/file.bin"]
        );
    }

    #[test]
    fn cloned_downloaders_receive_live_setting_updates() {
        let store_downloader = HttpDownloader::default();
        let settings_handle = store_downloader.clone();
        settings_handle
            .update_config(DownloadConfig {
                source: DownloadSource::MirrorFirst,
                mirror_url: "https://models.example.cn".to_owned(),
                ..DownloadConfig::default()
            })
            .unwrap();

        assert_eq!(
            store_downloader
                .candidate_urls("https://huggingface.co/org/model/resolve/main/model.onnx")[0],
            "https://models.example.cn/org/model/resolve/main/model.onnx"
        );
    }

    #[test]
    fn settings_reject_insecure_mirror_and_incomplete_custom_proxy() {
        let insecure = DownloadConfig {
            mirror_url: "http://models.example.cn".to_owned(),
            ..DownloadConfig::default()
        };
        assert!(HttpDownloader::with_config(insecure).is_err());

        let missing_proxy = DownloadConfig {
            proxy_mode: ProxyMode::Custom,
            proxy_url: String::new(),
            ..DownloadConfig::default()
        };
        assert!(HttpDownloader::with_config(missing_proxy).is_err());
    }

    #[test]
    fn proxy_urls_cover_http_and_socks() {
        for proxy_url in [
            "http://127.0.0.1:7890",
            "socks5://127.0.0.1:1080",
            "socks4a://proxy.example:1080",
        ] {
            let config = DownloadConfig {
                proxy_mode: ProxyMode::Custom,
                proxy_url: proxy_url.to_owned(),
                ..DownloadConfig::default()
            };
            assert!(HttpDownloader::with_config(config).is_ok(), "{proxy_url}");
        }
    }

    #[test]
    fn resumes_an_existing_partial_with_an_http_range() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let server = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let request = read_request(&mut stream);
            assert!(request.contains("Range: bytes=6-"));
            stream
                .write_all(
                    b"HTTP/1.1 206 Partial Content\r\nContent-Length: 5\r\nContent-Range: bytes 6-10/11\r\nConnection: close\r\n\r\nworld",
                )
                .unwrap();
        });

        let temp = tempfile::tempdir().unwrap();
        let dest = temp.path().join("model.part");
        std::fs::write(&dest, b"hello ").unwrap();
        let downloader = HttpDownloader::with_config(direct_config(1)).unwrap();
        let mut progress = Vec::new();
        downloader
            .download_with_progress(&format!("http://{address}/model"), &dest, &mut |bytes| {
                progress.push(bytes)
            })
            .unwrap();

        server.join().unwrap();
        assert_eq!(std::fs::read(dest).unwrap(), b"hello world");
        assert_eq!(progress.last(), Some(&11));
    }

    #[test]
    fn retries_a_dropped_transfer_from_its_last_byte() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let server = thread::spawn(move || {
            let (mut first, _) = listener.accept().unwrap();
            let request = read_request(&mut first);
            assert!(!request.contains("Range:"));
            first
                .write_all(
                    b"HTTP/1.1 200 OK\r\nContent-Length: 11\r\nConnection: close\r\n\r\nhello ",
                )
                .unwrap();
            drop(first);

            let (mut second, _) = listener.accept().unwrap();
            let request = read_request(&mut second);
            assert!(request.contains("Range: bytes=6-"));
            second
                .write_all(
                    b"HTTP/1.1 206 Partial Content\r\nContent-Length: 5\r\nContent-Range: bytes 6-10/11\r\nConnection: close\r\n\r\nworld",
                )
                .unwrap();
        });

        let temp = tempfile::tempdir().unwrap();
        let dest = temp.path().join("model.part");
        let downloader = HttpDownloader::with_config(direct_config(2)).unwrap();
        downloader
            .download_with_progress(&format!("http://{address}/model"), &dest, &mut |_| {})
            .unwrap();

        server.join().unwrap();
        assert_eq!(std::fs::read(dest).unwrap(), b"hello world");
    }

    #[test]
    #[ignore = "downloads a real model asset; run explicitly for release validation"]
    fn real_hugging_face_mirror_download_smoke() {
        let config = DownloadConfig {
            source: DownloadSource::MirrorFirst,
            proxy_mode: ProxyMode::Direct,
            retries: 1,
            ..DownloadConfig::default()
        };
        let downloader = HttpDownloader::with_config(config).unwrap();
        let temp = tempfile::tempdir().unwrap();
        let dest = temp.path().join("tokens.txt.part");
        downloader
            .download(
                "https://huggingface.co/csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/resolve/main/tokens.txt",
                &dest,
            )
            .unwrap();
        assert!(dest.metadata().unwrap().len() > 300_000);
    }
}
