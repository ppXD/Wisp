//! Local ONNX text-embedding backend: wraps `fastembed` to satisfy [`wisp_library::Embedder`].
//!
//! `fastembed` downloads the chosen model from the HuggingFace hub on first use, tokenizes, runs it
//! through ONNX Runtime, and pools — so this crate only adds the notes-library contract: a small
//! catalog of vetted multilingual models (the app pairs it with a custom slot) plus E5's distinct
//! passage/query prefixes.

use std::path::PathBuf;

use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};
use wisp_library::{Embedder, LibraryError, Result};

/// A vetted, downloadable embedding model the picker can offer.
pub struct CatalogModel {
    /// Stable id persisted as the user's choice and used to look the model back up.
    pub id: &'static str,
    /// Human-facing name for the picker.
    pub label: &'static str,
    /// Embedding dimension (vector length stored per chunk).
    pub dim: usize,
    /// Approximate download size in MiB, for the picker.
    pub size_mb: u32,
    /// The fastembed model this maps to.
    model: EmbeddingModel,
    /// Whether the model expects E5-style `query:` / `passage:` prefixes.
    e5: bool,
}

/// Built-in catalog: small, permissive, multilingual (incl. Chinese) models. The app shows these
/// plus a custom-model slot for anything newer off the leaderboard.
pub const CATALOG: &[CatalogModel] = &[
    CatalogModel {
        id: "e5-small",
        label: "Multilingual E5 small",
        dim: 384,
        size_mb: 120,
        model: EmbeddingModel::MultilingualE5Small,
        e5: true,
    },
    CatalogModel {
        id: "e5-base",
        label: "Multilingual E5 base",
        dim: 768,
        size_mb: 280,
        model: EmbeddingModel::MultilingualE5Base,
        e5: true,
    },
    CatalogModel {
        id: "e5-large",
        label: "Multilingual E5 large",
        dim: 1024,
        size_mb: 560,
        model: EmbeddingModel::MultilingualE5Large,
        e5: true,
    },
];

/// Looks up a catalog model by its stable id.
pub fn catalog_model(id: &str) -> Option<&'static CatalogModel> {
    CATALOG.iter().find(|m| m.id == id)
}

/// A loaded local embedder. Build with [`FastEmbedder::load`]; the model downloads on first use into
/// `cache_dir` and is cached there afterward.
pub struct FastEmbedder {
    model: TextEmbedding,
    dim: usize,
    e5: bool,
}

impl FastEmbedder {
    /// Loads a catalog model, downloading it into `cache_dir` if not already cached.
    pub fn load(model: &CatalogModel, cache_dir: PathBuf) -> Result<Self> {
        let opts = InitOptions::new(model.model.clone())
            .with_cache_dir(cache_dir)
            .with_show_download_progress(false);
        let loaded =
            TextEmbedding::try_new(opts).map_err(|e| LibraryError::Embed(e.to_string()))?;
        Ok(Self {
            model: loaded,
            dim: model.dim,
            e5: model.e5,
        })
    }

    /// Applies the E5 instruction prefix when the model wants one.
    fn prefixed(&self, kind: &str, text: &str) -> String {
        if self.e5 {
            format!("{kind}: {text}")
        } else {
            text.to_owned()
        }
    }
}

impl Embedder for FastEmbedder {
    fn dim(&self) -> usize {
        self.dim
    }

    fn embed_passages(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let docs: Vec<String> = texts.iter().map(|t| self.prefixed("passage", t)).collect();
        self.model
            .embed(docs, None)
            .map_err(|e| LibraryError::Embed(e.to_string()))
    }

    fn embed_query(&self, text: &str) -> Result<Vec<f32>> {
        let q = self.prefixed("query", text);
        let mut out = self
            .model
            .embed(vec![q], None)
            .map_err(|e| LibraryError::Embed(e.to_string()))?;
        Ok(out.pop().unwrap_or_default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn catalog_ids_are_unique_and_findable() {
        let mut seen = std::collections::HashSet::new();
        for m in CATALOG {
            assert!(seen.insert(m.id), "duplicate catalog id {}", m.id);
            assert_eq!(catalog_model(m.id).unwrap().id, m.id);
        }
        assert!(catalog_model("nope").is_none());
    }

    // Downloads ~120 MB and runs real inference, so it is opt-in: `cargo test -- --ignored`.
    #[test]
    #[ignore]
    fn e5_small_embeds_and_normalizes() {
        let dir = std::env::temp_dir().join("wisp-embed-test");
        let embedder = FastEmbedder::load(catalog_model("e5-small").unwrap(), dir).unwrap();
        assert_eq!(embedder.dim(), 384);

        let passages = embedder
            .embed_passages(&["the budget was approved"])
            .unwrap();
        assert_eq!(passages[0].len(), 384);
        let norm: f32 = passages[0].iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-3,
            "passage vector not L2-normalized: {norm}"
        );

        // A query about the same topic is more similar than an unrelated one.
        let q = embedder.embed_query("budget").unwrap();
        let dot = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>();
        let unrelated = embedder
            .embed_passages(&["the cat slept on the mat"])
            .unwrap();
        assert!(dot(&q, &passages[0]) > dot(&q, &unrelated[0]));
    }
}
