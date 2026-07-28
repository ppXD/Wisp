//! Local + cloud text-embedding backends for the Wisp notes library.
//!
//! Local models run on the generic [`OrtEmbedder`] (raw ONNX Runtime): each catalog entry carries a
//! [`Recipe`] (ONNX file + pooling + prompts + dim) plus the HuggingFace repo and files to fetch, so
//! a model is just "download these files, load them with this recipe". This one path covers encoder
//! models (CLS / Mean pooling) and decoder models (last-token, e.g. Qwen3). Cloud models call an
//! OpenAI-compatible `/embeddings` endpoint instead.

use std::path::{Path, PathBuf};

use wisp_library::{LibraryError, Result};
use wisp_models::{FileDownloader, HttpDownloader};

mod cloud;
pub use cloud::{cloud_catalog_model, CloudCatalogModel, CloudEmbedder, CLOUD_CATALOG};

mod ort_embed;
pub use ort_embed::{OrtEmbedder, Pooling, Recipe};

/// A vetted, downloadable embedding model the picker can offer.
pub struct CatalogModel {
    /// Stable id persisted as the user's choice and used to look the model back up.
    pub id: &'static str,
    /// Human-facing name for the picker.
    pub label: &'static str,
    /// Provider / family this belongs to (the picker's left-pane grouping), e.g. `"Multilingual E5"`.
    pub group: &'static str,
    /// Approximate download size in MiB (full-precision ONNX), shown before download.
    pub size_mb: u32,
    /// HuggingFace repo the model files download from.
    pub repo: &'static str,
    /// Files fetched into the model dir (the ONNX file + any external weights + tokenizer + config),
    /// as repo-relative paths.
    pub files: &'static [&'static str],
    /// How to load + run the model (ONNX file within the dir, pooling, prompts, dim).
    pub recipe: Recipe,
}

/// Built-in catalog: small, permissive, multilingual / Chinese models. E5 is multilingual and
/// asymmetric (passage/query prefixes) with Mean pooling; BGE-zh is Chinese-tuned and symmetric
/// (v1.5 needs no instruction) with CLS pooling. The app pairs this with a custom-model slot.
pub const CATALOG: &[CatalogModel] = &[
    CatalogModel {
        id: "e5-small",
        label: "Multilingual E5 small",
        group: "Multilingual E5",
        size_mb: 470,
        repo: "intfloat/multilingual-e5-small",
        files: &[
            "onnx/model.onnx",
            "tokenizer.json",
            "tokenizer_config.json",
            "config.json",
            "special_tokens_map.json",
        ],
        recipe: Recipe {
            onnx_file: "onnx/model.onnx",
            pooling: Pooling::Mean,
            passage_prefix: "passage: ",
            query_prefix: "query: ",
            normalize: true,
            dim: 384,
            max_length: 512,
        },
    },
    CatalogModel {
        id: "e5-base",
        label: "Multilingual E5 base",
        group: "Multilingual E5",
        size_mb: 1100,
        repo: "intfloat/multilingual-e5-base",
        files: &[
            "onnx/model.onnx",
            "tokenizer.json",
            "tokenizer_config.json",
            "config.json",
            "special_tokens_map.json",
        ],
        recipe: Recipe {
            onnx_file: "onnx/model.onnx",
            pooling: Pooling::Mean,
            passage_prefix: "passage: ",
            query_prefix: "query: ",
            normalize: true,
            dim: 768,
            max_length: 512,
        },
    },
    CatalogModel {
        id: "e5-large",
        label: "Multilingual E5 large",
        group: "Multilingual E5",
        size_mb: 2200,
        // This export keeps weights in an external `model.onnx_data` next to `model.onnx`; ONNX
        // Runtime loads it automatically when both sit in the same dir, so both are in `files`.
        repo: "Qdrant/multilingual-e5-large-onnx",
        files: &[
            "model.onnx",
            "model.onnx_data",
            "tokenizer.json",
            "tokenizer_config.json",
            "config.json",
            "special_tokens_map.json",
        ],
        recipe: Recipe {
            onnx_file: "model.onnx",
            pooling: Pooling::Mean,
            passage_prefix: "passage: ",
            query_prefix: "query: ",
            normalize: true,
            dim: 1024,
            max_length: 512,
        },
    },
    CatalogModel {
        id: "bge-small-zh",
        label: "BGE small · Chinese",
        group: "BGE · Chinese",
        size_mb: 95,
        repo: "Xenova/bge-small-zh-v1.5",
        files: &[
            "onnx/model.onnx",
            "tokenizer.json",
            "tokenizer_config.json",
            "config.json",
            "special_tokens_map.json",
        ],
        recipe: Recipe {
            onnx_file: "onnx/model.onnx",
            pooling: Pooling::Cls,
            passage_prefix: "",
            query_prefix: "",
            normalize: true,
            dim: 512,
            max_length: 512,
        },
    },
    CatalogModel {
        id: "bge-large-zh",
        label: "BGE large · Chinese",
        group: "BGE · Chinese",
        size_mb: 1300,
        repo: "Xenova/bge-large-zh-v1.5",
        files: &[
            "onnx/model.onnx",
            "tokenizer.json",
            "tokenizer_config.json",
            "config.json",
            "special_tokens_map.json",
        ],
        recipe: Recipe {
            onnx_file: "onnx/model.onnx",
            pooling: Pooling::Cls,
            passage_prefix: "",
            query_prefix: "",
            normalize: true,
            dim: 1024,
            max_length: 512,
        },
    },
    CatalogModel {
        id: "bge-m3",
        label: "BGE-M3 · multilingual",
        group: "BGE · Multilingual",
        size_mb: 570,
        repo: "Xenova/bge-m3",
        files: &[
            "onnx/model_quantized.onnx",
            "tokenizer.json",
            "tokenizer_config.json",
            "config.json",
            "special_tokens_map.json",
        ],
        recipe: Recipe {
            onnx_file: "onnx/model_quantized.onnx",
            pooling: Pooling::Cls,
            passage_prefix: "",
            query_prefix: "",
            normalize: true,
            dim: 1024,
            max_length: 512,
        },
    },
    CatalogModel {
        id: "gte-multilingual-base",
        label: "GTE multilingual base",
        group: "GTE",
        size_mb: 1220,
        repo: "onnx-community/gte-multilingual-base",
        files: &[
            "onnx/model.onnx",
            "tokenizer.json",
            "tokenizer_config.json",
            "config.json",
            "special_tokens_map.json",
        ],
        recipe: Recipe {
            onnx_file: "onnx/model.onnx",
            pooling: Pooling::Cls,
            passage_prefix: "",
            query_prefix: "",
            normalize: true,
            dim: 768,
            max_length: 512,
        },
    },
    CatalogModel {
        id: "qwen3-0.6b",
        label: "Qwen3 Embedding 0.6B",
        group: "Qwen3",
        size_mb: 600,
        repo: "onnx-community/Qwen3-Embedding-0.6B-ONNX",
        files: &[
            "onnx/model_quantized.onnx",
            "tokenizer.json",
            "tokenizer_config.json",
            "config.json",
            "special_tokens_map.json",
        ],
        recipe: Recipe {
            onnx_file: "onnx/model_quantized.onnx",
            // Decoder embedder: last-token pooling, and the query carries Qwen3's instruct prompt
            // while stored passages do not (asymmetric).
            pooling: Pooling::LastToken,
            passage_prefix: "",
            query_prefix: "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: ",
            normalize: true,
            dim: 1024,
            max_length: 512,
        },
    },
];

/// Looks up a catalog model by its stable id.
pub fn catalog_model(id: &str) -> Option<&'static CatalogModel> {
    CATALOG.iter().find(|m| m.id == id)
}

/// Downloads a catalog model's files from its HF repo into `dir` (creating it). Each file lands in a
/// sibling `*.part` first and is renamed into place only after the full body is written, so an
/// interrupted download never leaves a half-written file behind; already-present files are skipped,
/// making a re-run resume. Callers observe progress by watching `dir` grow.
pub fn download_model(model: &CatalogModel, dir: &Path) -> Result<()> {
    download_model_with(model, dir, &HttpDownloader::default())
}

/// Downloads with Wisp's shared regional downloader, so the app can apply its live mirror/proxy
/// settings to embedding models as well as transcription models.
pub fn download_model_with(
    model: &CatalogModel,
    dir: &Path,
    downloader: &dyn FileDownloader,
) -> Result<()> {
    for file in model.files {
        let dest = dir.join(file);
        if dest.exists() {
            continue;
        }

        if let Some(parent) = dest.parent() {
            std::fs::create_dir_all(parent).map_err(io_err)?;
        }

        let url = format!("https://huggingface.co/{}/resolve/main/{file}", model.repo);
        fetch_atomic(&url, &dest, downloader)?;
    }
    Ok(())
}

/// Streams `url` into `dest` via a sibling `*.part`, renamed into place only after the whole body is
/// written. A failed transfer retains only the non-final `*.part`, allowing the next attempt to
/// continue with an HTTP Range request.
fn fetch_atomic(url: &str, dest: &Path, downloader: &dyn FileDownloader) -> Result<()> {
    let mut part = dest.as_os_str().to_owned();
    part.push(".part");
    let part = PathBuf::from(part);

    downloader
        .download(url, &part)
        .map_err(|error| LibraryError::Embed(error.to_string()))?;

    std::fs::rename(&part, dest).map_err(io_err)
}

fn io_err(e: std::io::Error) -> LibraryError {
    LibraryError::Embed(format!("io: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};
    use wisp_core::error::Result as ModelResult;
    use wisp_models::FileDownloader;

    struct RecordingDownloader(Arc<Mutex<Vec<String>>>);

    impl FileDownloader for RecordingDownloader {
        fn download(&self, url: &str, dest: &Path) -> ModelResult<()> {
            self.0.lock().unwrap().push(url.to_owned());
            std::fs::write(dest, url)?;
            Ok(())
        }
    }

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
    fn catalog_download_can_use_the_shared_regional_downloader() {
        let temp = tempfile::tempdir().unwrap();
        let calls = Arc::new(Mutex::new(Vec::new()));
        let downloader = RecordingDownloader(Arc::clone(&calls));

        download_model_with(&CATALOG[0], temp.path(), &downloader).unwrap();

        assert_eq!(calls.lock().unwrap().len(), CATALOG[0].files.len());
        for file in CATALOG[0].files {
            assert!(temp.path().join(file).is_file(), "{file}");
        }
    }

    #[test]
    fn catalog_prefixes_match_family() {
        // E5 is asymmetric (both prefixes set); BGE-zh v1.5 is symmetric (no instruction).
        for m in CATALOG {
            if m.id.starts_with("e5") {
                assert_eq!(m.recipe.passage_prefix, "passage: ", "{}", m.id);
                assert_eq!(m.recipe.query_prefix, "query: ", "{}", m.id);
            } else if m.id.starts_with("bge") {
                assert_eq!(m.recipe.passage_prefix, "", "{}", m.id);
                assert_eq!(m.recipe.query_prefix, "", "{}", m.id);
            }
        }
    }

    #[test]
    fn catalog_files_include_the_recipe_onnx_and_tokenizer() {
        // The downloaded file set must contain the ONNX the recipe loads and the tokenizer the
        // embedder reads, or a download would "succeed" yet fail to load.
        for m in CATALOG {
            assert!(
                m.files.contains(&m.recipe.onnx_file),
                "{} files omit its recipe onnx_file {}",
                m.id,
                m.recipe.onnx_file
            );
            assert!(
                m.files.contains(&"tokenizer.json"),
                "{} omits tokenizer.json",
                m.id
            );
            assert!(m.recipe.dim > 0, "{} has a zero dim", m.id);
        }
    }

    // The real picker path for a catalog model: self-download its files, then load + embed via the
    // recipe — exactly what download_embedding_model + activation do. ~95 MB (bge-small-zh). Opt-in:
    // `cargo test -- --ignored`.
    #[test]
    #[ignore]
    fn download_model_then_load_and_embed() {
        use wisp_library::Embedder;

        let model = catalog_model("bge-small-zh").unwrap();
        let dir = std::env::temp_dir().join("wisp-embed-dl-bge-small-zh");
        download_model(model, &dir).unwrap();

        // Every declared file landed and no `*.part` lingered.
        for f in model.files {
            assert!(dir.join(f).exists(), "missing downloaded file {f}");
            let mut part = dir.join(f).into_os_string();
            part.push(".part");
            assert!(!Path::new(&part).exists(), "leftover .part for {f}");
        }

        // A second call is a no-op (files already present) and must still succeed.
        download_model(model, &dir).unwrap();

        let emb = OrtEmbedder::load(&dir, &model.recipe).unwrap();
        assert_eq!(emb.dim(), model.recipe.dim);
        let v = emb.embed_query("预算").unwrap();
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-3,
            "embedding not normalized: {norm}"
        );
    }
}
