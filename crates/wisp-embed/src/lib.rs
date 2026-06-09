//! Local ONNX text-embedding backend: wraps `fastembed` to satisfy [`wisp_library::Embedder`].
//!
//! `fastembed` downloads the chosen model from the HuggingFace hub on first use, tokenizes, runs it
//! through ONNX Runtime, and pools — so this crate only adds the notes-library contract: a small
//! catalog of vetted multilingual models (the app pairs it with a custom slot) plus E5's distinct
//! passage/query prefixes.

use std::path::PathBuf;

use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};
use wisp_library::{Embedder, LibraryError, Result};

mod cloud;
pub use cloud::{cloud_catalog_model, CloudCatalogModel, CloudEmbedder, CLOUD_CATALOG};

/// A vetted, downloadable embedding model the picker can offer.
pub struct CatalogModel {
    /// Stable id persisted as the user's choice and used to look the model back up.
    pub id: &'static str,
    /// Human-facing name for the picker.
    pub label: &'static str,
    /// Provider / family this belongs to (the picker's left-pane grouping), e.g. `"Multilingual E5"`.
    pub group: &'static str,
    /// Embedding dimension (vector length stored per chunk).
    pub dim: usize,
    /// Approximate fp32 download size in MiB (fastembed serves full-precision ONNX).
    pub size_mb: u32,
    /// The fastembed model this maps to.
    model: EmbeddingModel,
    /// Instruction prepended to stored passages — empty for symmetric models, `"passage: "` for E5.
    passage_prefix: &'static str,
    /// Instruction prepended to a search query — empty for symmetric models, `"query: "` for E5.
    query_prefix: &'static str,
}

/// Built-in catalog: small, permissive, multilingual / Chinese models that fastembed serves
/// directly. E5 is multilingual and asymmetric (passage/query prefixes); BGE-zh is Chinese-tuned and
/// symmetric (v1.5 needs no instruction). The app pairs this with a custom-model slot, and a
/// raw-ONNX path later covers decoder models like Qwen3-Embedding and quantized downloads.
pub const CATALOG: &[CatalogModel] = &[
    CatalogModel {
        id: "e5-small",
        label: "Multilingual E5 small",
        group: "Multilingual E5",
        dim: 384,
        size_mb: 470,
        model: EmbeddingModel::MultilingualE5Small,
        passage_prefix: "passage: ",
        query_prefix: "query: ",
    },
    CatalogModel {
        id: "e5-base",
        label: "Multilingual E5 base",
        group: "Multilingual E5",
        dim: 768,
        size_mb: 1100,
        model: EmbeddingModel::MultilingualE5Base,
        passage_prefix: "passage: ",
        query_prefix: "query: ",
    },
    CatalogModel {
        id: "e5-large",
        label: "Multilingual E5 large",
        group: "Multilingual E5",
        dim: 1024,
        size_mb: 2200,
        model: EmbeddingModel::MultilingualE5Large,
        passage_prefix: "passage: ",
        query_prefix: "query: ",
    },
    CatalogModel {
        id: "bge-small-zh",
        label: "BGE small · Chinese",
        group: "BGE · Chinese",
        dim: 512,
        size_mb: 95,
        model: EmbeddingModel::BGESmallZHV15,
        passage_prefix: "",
        query_prefix: "",
    },
    CatalogModel {
        id: "bge-large-zh",
        label: "BGE large · Chinese",
        group: "BGE · Chinese",
        dim: 1024,
        size_mb: 1300,
        model: EmbeddingModel::BGELargeZHV15,
        passage_prefix: "",
        query_prefix: "",
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
    passage_prefix: &'static str,
    query_prefix: &'static str,
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
            passage_prefix: model.passage_prefix,
            query_prefix: model.query_prefix,
        })
    }
}

impl Embedder for FastEmbedder {
    fn dim(&self) -> usize {
        self.dim
    }

    fn embed_passages(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let docs: Vec<String> = texts
            .iter()
            .map(|t| format!("{}{t}", self.passage_prefix))
            .collect();
        self.model
            .embed(docs, None)
            .map_err(|e| LibraryError::Embed(e.to_string()))
    }

    fn embed_query(&self, text: &str) -> Result<Vec<f32>> {
        let q = format!("{}{text}", self.query_prefix);
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

    #[test]
    fn catalog_prefixes_match_family() {
        // E5 is asymmetric (both prefixes set); BGE-zh v1.5 is symmetric (no instruction).
        for m in CATALOG {
            if m.id.starts_with("e5") {
                assert_eq!(m.passage_prefix, "passage: ", "{}", m.id);
                assert_eq!(m.query_prefix, "query: ", "{}", m.id);
            } else if m.id.starts_with("bge") {
                assert_eq!(m.passage_prefix, "", "{}", m.id);
                assert_eq!(m.query_prefix, "", "{}", m.id);
            }
        }
    }

    // Downloads ~470 MB (fp32) and runs real inference, so it is opt-in: `cargo test -- --ignored`.
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
