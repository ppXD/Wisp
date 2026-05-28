//! Runtime configuration and environment-variable overrides.

use std::path::{Path, PathBuf};

use crate::model::ModelId;

/// Environment variable overriding the directory where models are stored.
///
/// Pinned by a test — renaming it breaks operators who set it, so the change must be deliberate.
pub const MODEL_DIR_ENV: &str = "WISP_MODEL_DIR";

/// Environment variable overriding the base URL models are downloaded from (e.g. to point at a
/// private mirror in an air-gapped deployment).
pub const DOWNLOAD_BASE_URL_ENV: &str = "WISP_DOWNLOAD_BASE_URL";

/// Resolved runtime configuration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Config {
    /// Directory holding downloaded models.
    pub model_dir: PathBuf,
    /// Base URL for model downloads.
    pub download_base_url: String,
    /// Currently selected model, if any.
    pub active_model: Option<ModelId>,
}

impl Config {
    /// Builds a config from explicit override values (pure; no environment access).
    ///
    /// `None` overrides fall back to the supplied defaults. This is the unit-testable core that
    /// [`Config::from_env`] wraps.
    pub fn resolve(
        model_dir_override: Option<&str>,
        base_url_override: Option<&str>,
        default_model_dir: &Path,
        default_base_url: &str,
    ) -> Self {
        let model_dir = model_dir_override
            .map(PathBuf::from)
            .unwrap_or_else(|| default_model_dir.to_path_buf());

        let download_base_url = base_url_override
            .map(str::to_owned)
            .unwrap_or_else(|| default_base_url.to_owned());

        Self {
            model_dir,
            download_base_url,
            active_model: None,
        }
    }

    /// Builds a config by reading the [`MODEL_DIR_ENV`] and [`DOWNLOAD_BASE_URL_ENV`] overrides,
    /// falling back to the supplied defaults.
    pub fn from_env(default_model_dir: &Path, default_base_url: &str) -> Self {
        let model_dir = std::env::var(MODEL_DIR_ENV).ok();
        let base_url = std::env::var(DOWNLOAD_BASE_URL_ENV).ok();

        Self::resolve(
            model_dir.as_deref(),
            base_url.as_deref(),
            default_model_dir,
            default_base_url,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn env_var_names_are_pinned() {
        assert_eq!(MODEL_DIR_ENV, "WISP_MODEL_DIR");
        assert_eq!(DOWNLOAD_BASE_URL_ENV, "WISP_DOWNLOAD_BASE_URL");
    }

    #[test]
    fn overrides_take_precedence() {
        let cfg = Config::resolve(
            Some("/custom/models"),
            Some("https://mirror.example/m"),
            Path::new("/default/models"),
            "https://hf.example",
        );
        assert_eq!(cfg.model_dir, PathBuf::from("/custom/models"));
        assert_eq!(cfg.download_base_url, "https://mirror.example/m");
    }

    #[test]
    fn defaults_used_when_no_override() {
        let cfg = Config::resolve(
            None,
            None,
            Path::new("/default/models"),
            "https://hf.example",
        );
        assert_eq!(cfg.model_dir, PathBuf::from("/default/models"));
        assert_eq!(cfg.download_base_url, "https://hf.example");
        assert!(cfg.active_model.is_none());
    }
}
