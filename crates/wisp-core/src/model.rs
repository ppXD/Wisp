//! Model catalog types and the [`ModelStore`] trait.

use std::path::PathBuf;

use crate::error::Result;

/// Family of transcription model.
///
/// `#[non_exhaustive]` so new families don't break downstream matches.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ModelFamily {
    /// OpenAI Whisper and derivatives (e.g. large-v3-turbo, distil).
    Whisper,
    /// NVIDIA Parakeet transducer models.
    Parakeet,
    /// sherpa-onnx SenseVoice (single-file multilingual model).
    SenseVoice,
}

/// Quantization level of a model's weights.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Quant {
    /// 32-bit float (full precision).
    F32,
    /// 16-bit float.
    F16,
    /// 8-bit integer.
    Q8,
    /// 5-bit integer.
    Q5,
    /// 4-bit integer.
    Q4,
    /// Any other scheme, named verbatim.
    Other(String),
}

/// Stable identifier for a catalog entry (e.g. `"whisper-large-v3-turbo-q5"`).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ModelId(pub String);

impl ModelId {
    /// Borrows the id as a string slice.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// A single downloadable file belonging to a model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelFile {
    /// File name as stored on disk (e.g. `"ggml-large-v3-turbo-q5_0.bin"`).
    pub name: String,
    /// Fully-qualified download URL.
    pub url: String,
    /// Lowercase hex SHA-256 of the file's contents, for verification.
    pub sha256: String,
    /// Expected size in bytes.
    pub size_bytes: u64,
}

/// A catalog entry describing an installable model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelDescriptor {
    /// Stable identifier.
    pub id: ModelId,
    /// Model family.
    pub family: ModelFamily,
    /// Quantization level.
    pub quant: Quant,
    /// Human-readable name for the UI.
    pub display_name: String,
    /// Files that make up the model (one `.bin`, or several for ONNX bundles).
    pub files: Vec<ModelFile>,
    /// Languages the model supports; empty means unspecified/multilingual.
    pub languages: Vec<String>,
    /// Short human guidance on the size / speed / accuracy tradeoff, shown in the picker.
    pub description: String,
}

impl ModelDescriptor {
    /// Total on-disk size across all files.
    pub fn total_size_bytes(&self) -> u64 {
        self.files.iter().map(|f| f.size_bytes).sum()
    }
}

/// Stores and provides transcription models.
///
/// Implementations own the catalog, downloading, verification, and on-disk layout. The trait is
/// intentionally narrow so a mock, a filesystem store, or a future remote store all fit.
pub trait ModelStore: Send + Sync {
    /// All models offered by the catalog.
    fn available(&self) -> Vec<ModelDescriptor>;

    /// Ids of models already downloaded and verified locally.
    fn installed(&self) -> Result<Vec<ModelId>>;

    /// Ensures `id` is present locally (downloading + verifying if needed) and returns the
    /// directory containing its files.
    fn ensure(&self, id: &ModelId) -> Result<PathBuf>;

    /// Removes a downloaded model from local storage.
    fn remove(&self, id: &ModelId) -> Result<()>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn total_size_sums_files() {
        let d = ModelDescriptor {
            id: ModelId("m".into()),
            family: ModelFamily::Whisper,
            quant: Quant::Q5,
            display_name: "M".into(),
            files: vec![
                ModelFile {
                    name: "a".into(),
                    url: "u".into(),
                    sha256: "x".into(),
                    size_bytes: 10,
                },
                ModelFile {
                    name: "b".into(),
                    url: "u".into(),
                    sha256: "y".into(),
                    size_bytes: 32,
                },
            ],
            languages: vec![],
            description: String::new(),
        };
        assert_eq!(d.total_size_bytes(), 42);
    }

    #[test]
    fn model_id_as_str() {
        assert_eq!(ModelId("whisper".into()).as_str(), "whisper");
    }
}
