//! Error type shared across `wisp-core`.

/// The crate-wide result alias.
pub type Result<T> = std::result::Result<T, WispError>;

/// All errors surfaced by Wisp's core.
///
/// Marked `#[non_exhaustive]` so new variants can be added without breaking downstream
/// `match` statements.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum WispError {
    /// Audio capture or conversion failed.
    #[error("audio: {0}")]
    Audio(String),

    /// The ASR engine failed to transcribe.
    #[error("engine: {0}")]
    Engine(String),

    /// A model could not be resolved, downloaded, or verified.
    #[error("model: {0}")]
    Model(String),

    /// Configuration was invalid.
    #[error("config: {0}")]
    Config(String),

    /// An underlying I/O operation failed.
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_includes_context() {
        let err = WispError::Engine("model not loaded".into());
        assert_eq!(err.to_string(), "engine: model not loaded");
    }

    #[test]
    fn io_error_converts() {
        let io = std::io::Error::new(std::io::ErrorKind::NotFound, "nope");
        let err: WispError = io.into();
        assert!(matches!(err, WispError::Io(_)));
    }
}
