//! Generic recipe-driven ONNX embedder: runs ANY local ONNX text-embedding model — encoder
//! (CLS / Mean pooling) or decoder (last-token, e.g. Qwen3-Embedding) — directly on ONNX Runtime via
//! `ort`, with the tokenizer + pooling + prompts described by a small [`Recipe`]. This is the unified
//! local backend; a model's files are downloaded into a per-model dir and loaded from there.
//!
//! The inference path mirrors fastembed's (tokenize → `input_ids`/`attention_mask`/`token_type_ids`
//! → run → pool → L2-normalize), but with full control over pooling + prompts so decoder models work
//! too — which fastembed's CLS/Mean-only pooling cannot express.

use std::path::Path;

use ndarray::{Array2, ArrayViewD};
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Value;
use tokenizers::{PaddingParams, PaddingStrategy, Tokenizer, TruncationParams};
use wisp_library::{Embedder, LibraryError, Result};

/// How a model's token embeddings collapse into one sentence vector.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Pooling {
    /// First (`[CLS]`) token — BGE, many BERT sentence models.
    Cls,
    /// Attention-masked mean of all tokens — E5, most sentence-transformers.
    Mean,
    /// Last non-pad token — decoder embedders (Qwen3-Embedding, gte-Qwen2).
    LastToken,
}

/// Everything needed to run one ONNX embedding model. Catalog models carry a hardcoded recipe; a
/// custom import infers one from the repo's config files.
pub struct Recipe {
    /// ONNX file path within the model dir, e.g. `"onnx/model.onnx"`.
    pub onnx_file: &'static str,
    pub pooling: Pooling,
    /// Instruction prepended to a stored passage (empty for symmetric models, `"passage: "` for E5).
    pub passage_prefix: &'static str,
    /// Instruction prepended to a search query (`"query: "` for E5).
    pub query_prefix: &'static str,
    pub normalize: bool,
    pub dim: usize,
    pub max_length: usize,
}

/// A loaded ONNX embedding model.
pub struct OrtEmbedder {
    session: Session,
    tokenizer: Tokenizer,
    pooling: Pooling,
    passage_prefix: String,
    query_prefix: String,
    normalize: bool,
    dim: usize,
    needs_token_type_ids: bool,
}

impl OrtEmbedder {
    /// Loads a model from `model_dir` (containing the recipe's ONNX file + `tokenizer.json` +
    /// `tokenizer_config.json`) per `recipe`.
    pub fn load(model_dir: &Path, recipe: &Recipe) -> Result<Self> {
        let mut tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json"))
            .map_err(|e| LibraryError::Embed(format!("tokenizer.json: {e}")))?;

        let max_length = read_max_length(model_dir)
            .map(|m| m.min(recipe.max_length))
            .unwrap_or(recipe.max_length);
        tokenizer
            .with_padding(Some(PaddingParams {
                strategy: PaddingStrategy::BatchLongest,
                ..Default::default()
            }))
            .with_truncation(Some(TruncationParams {
                max_length,
                ..Default::default()
            }))
            .map_err(|e| LibraryError::Embed(format!("tokenizer config: {e}")))?;

        let threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        let session = Session::builder()
            .map_err(ort_err)?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(ort_err)?
            .with_intra_threads(threads)
            .map_err(ort_err)?
            .commit_from_file(model_dir.join(recipe.onnx_file))
            .map_err(ort_err)?;

        let needs_token_type_ids = session.inputs.iter().any(|i| i.name == "token_type_ids");

        Ok(Self {
            session,
            tokenizer,
            pooling: recipe.pooling,
            passage_prefix: recipe.passage_prefix.to_owned(),
            query_prefix: recipe.query_prefix.to_owned(),
            normalize: recipe.normalize,
            dim: recipe.dim,
            needs_token_type_ids,
        })
    }

    /// Tokenizes, runs the model, pools, and (optionally) L2-normalizes one batch of texts.
    fn run(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let refs: Vec<&str> = texts.iter().map(String::as_str).collect();
        let encodings = self
            .tokenizer
            .encode_batch(refs, true)
            .map_err(|e| LibraryError::Embed(format!("encode: {e}")))?;

        let seq = encodings[0].len();
        let batch = encodings.len();

        let mut ids = Vec::with_capacity(batch * seq);
        let mut mask = Vec::with_capacity(batch * seq);
        let mut types = Vec::with_capacity(batch * seq);
        for enc in &encodings {
            ids.extend(enc.get_ids().iter().map(|x| *x as i64));
            mask.extend(enc.get_attention_mask().iter().map(|x| *x as i64));
            types.extend(enc.get_type_ids().iter().map(|x| *x as i64));
        }

        let ids = Array2::from_shape_vec((batch, seq), ids).map_err(shape_err)?;
        let mask_arr = Array2::from_shape_vec((batch, seq), mask).map_err(shape_err)?;
        let type_arr = Array2::from_shape_vec((batch, seq), types).map_err(shape_err)?;

        let ids_value = Value::from_array(ids).map_err(ort_err)?;
        let mask_value = Value::from_array(mask_arr.view()).map_err(ort_err)?;
        let mut inputs = ort::inputs![
            "input_ids" => ids_value,
            "attention_mask" => mask_value,
        ]
        .map_err(ort_err)?;
        if self.needs_token_type_ids {
            inputs.push((
                "token_type_ids".into(),
                Value::from_array(type_arr).map_err(ort_err)?.into(),
            ));
        }

        let outputs = self.session.run(inputs).map_err(ort_err)?;
        let key = pick_output(outputs.keys().collect::<Vec<_>>())?;
        let tensor: ArrayViewD<f32> = outputs
            .get(key.as_str())
            .ok_or_else(|| LibraryError::Embed("model output missing".to_owned()))?
            .try_extract_tensor::<f32>()
            .map_err(ort_err)?;

        let pooled = self.pool(&tensor, &mask_arr)?;
        Ok(pooled
            .outer_iter()
            .map(|row| {
                let mut v = row.to_vec();
                if self.normalize {
                    l2_normalize(&mut v);
                }
                v
            })
            .collect())
    }

    /// Collapses `[batch, seq, hidden]` token embeddings to `[batch, hidden]` per the pooling mode.
    /// A model that already emits a pooled `[batch, hidden]` tensor is passed through unchanged.
    fn pool(&self, tensor: &ArrayViewD<f32>, mask: &Array2<i64>) -> Result<Array2<f32>> {
        if tensor.ndim() == 2 {
            return tensor
                .to_owned()
                .into_dimensionality()
                .map_err(|e| LibraryError::Embed(format!("output reshape: {e}")));
        }
        if tensor.ndim() != 3 {
            return Err(LibraryError::Embed(format!(
                "unexpected model output rank {}",
                tensor.ndim()
            )));
        }

        let (batch, seq, hidden) = (tensor.shape()[0], tensor.shape()[1], tensor.shape()[2]);
        let mut out = Array2::<f32>::zeros((batch, hidden));
        for b in 0..batch {
            match self.pooling {
                Pooling::Cls => {
                    for h in 0..hidden {
                        out[[b, h]] = tensor[[b, 0, h]];
                    }
                }
                Pooling::Mean => {
                    let mut count = 0f32;
                    for t in 0..seq {
                        if mask[[b, t]] != 0 {
                            count += 1.0;
                            for h in 0..hidden {
                                out[[b, h]] += tensor[[b, t, h]];
                            }
                        }
                    }
                    if count > 0.0 {
                        for h in 0..hidden {
                            out[[b, h]] /= count;
                        }
                    }
                }
                Pooling::LastToken => {
                    let last = (0..seq).rev().find(|&t| mask[[b, t]] != 0).unwrap_or(0);
                    for h in 0..hidden {
                        out[[b, h]] = tensor[[b, last, h]];
                    }
                }
            }
        }
        Ok(out)
    }
}

impl Embedder for OrtEmbedder {
    fn dim(&self) -> usize {
        self.dim
    }

    fn embed_passages(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let prefixed: Vec<String> = texts
            .iter()
            .map(|t| format!("{}{t}", self.passage_prefix))
            .collect();
        self.run(&prefixed)
    }

    fn embed_query(&self, text: &str) -> Result<Vec<f32>> {
        let q = format!("{}{text}", self.query_prefix);
        Ok(self.run(&[q])?.pop().unwrap_or_default())
    }
}

/// Picks the embedding output: the token-level `last_hidden_state` (which we pool) when present,
/// otherwise a pre-pooled `sentence_embedding`, otherwise the first output.
fn pick_output(keys: Vec<&str>) -> Result<String> {
    for preferred in [
        "last_hidden_state",
        "token_embeddings",
        "sentence_embedding",
    ] {
        if keys.contains(&preferred) {
            return Ok(preferred.to_owned());
        }
    }
    keys.first()
        .map(|k| (*k).to_owned())
        .ok_or_else(|| LibraryError::Embed("model returned no outputs".to_owned()))
}

fn l2_normalize(v: &mut [f32]) {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in v {
            *x /= norm;
        }
    }
}

/// Reads `model_max_length` from `tokenizer_config.json` so we never feed past the model's limit.
fn read_max_length(dir: &Path) -> Option<usize> {
    let bytes = std::fs::read(dir.join("tokenizer_config.json")).ok()?;
    let cfg: serde_json::Value = serde_json::from_slice(&bytes).ok()?;
    let raw = cfg.get("model_max_length")?.as_f64()?;
    // Some configs use a sentinel like 1e30; clamp to something sane.
    if raw.is_finite() && raw > 0.0 && raw < 1e7 {
        Some(raw as usize)
    } else {
        None
    }
}

fn ort_err(e: ort::Error) -> LibraryError {
    LibraryError::Embed(format!("onnx: {e}"))
}

fn shape_err(e: ndarray::ShapeError) -> LibraryError {
    LibraryError::Embed(format!("tensor shape: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pick_output_prefers_last_hidden_state() {
        assert_eq!(
            pick_output(vec!["pooler_output", "last_hidden_state"]).unwrap(),
            "last_hidden_state"
        );
        assert_eq!(
            pick_output(vec!["sentence_embedding"]).unwrap(),
            "sentence_embedding"
        );
        assert_eq!(pick_output(vec!["only"]).unwrap(), "only");
        assert!(pick_output(vec![]).is_err());
    }

    #[test]
    fn normalize_makes_unit_length() {
        let mut v = vec![3.0f32, 4.0];
        l2_normalize(&mut v);
        let n = (v[0] * v[0] + v[1] * v[1]).sqrt();
        assert!((n - 1.0).abs() < 1e-6);
        assert!((v[0] - 0.6).abs() < 1e-6);
    }

    fn download_file(repo: &str, file: &str, dest: &std::path::Path) {
        if dest.exists() {
            return;
        }
        if let Some(p) = dest.parent() {
            std::fs::create_dir_all(p).unwrap();
        }
        let url = format!("https://huggingface.co/{repo}/resolve/main/{file}");
        let resp = ureq::get(&url).call().unwrap();
        let mut reader = resp.into_reader();
        let mut out = std::fs::File::create(dest).unwrap();
        std::io::copy(&mut reader, &mut out).unwrap();
    }

    // Downloads ~470 MB (fp32 ONNX) and runs real inference, so it is opt-in: `cargo test -- --ignored`.
    // Proves the whole path — tokenize → ONNX → mean-pool → normalize — is correct, by checking the
    // vector is unit-length, the right dimension, and that an E5 query lands closer to a related
    // passage than an unrelated one.
    #[test]
    #[ignore]
    fn e5_small_embeds_with_correct_semantics() {
        let dir = std::env::temp_dir().join("wisp-ort-e5-small");
        let repo = "intfloat/multilingual-e5-small";
        for f in [
            "onnx/model.onnx",
            "tokenizer.json",
            "config.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
        ] {
            download_file(repo, f, &dir.join(f));
        }

        let recipe = Recipe {
            onnx_file: "onnx/model.onnx",
            pooling: Pooling::Mean,
            passage_prefix: "passage: ",
            query_prefix: "query: ",
            normalize: true,
            dim: 384,
            max_length: 512,
        };
        let emb = OrtEmbedder::load(&dir, &recipe).unwrap();
        assert_eq!(emb.dim(), 384);

        let passages = emb
            .embed_passages(&["the annual budget was approved", "the cat slept on the mat"])
            .unwrap();
        assert_eq!(passages[0].len(), 384);
        let norm: f32 = passages[0].iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-3,
            "passage not L2-normalized: {norm}"
        );

        let q = emb.embed_query("budget").unwrap();
        let dot = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>();
        assert!(
            dot(&q, &passages[0]) > dot(&q, &passages[1]),
            "the budget query should match the budget passage more than the cat one"
        );
    }

    // Verifies the CLS-pooling path (and a Chinese model) — `cargo test -- --ignored`. ~95 MB.
    #[test]
    #[ignore]
    fn bge_small_zh_embeds_with_cls_pooling() {
        let dir = std::env::temp_dir().join("wisp-ort-bge-small-zh");
        let repo = "Xenova/bge-small-zh-v1.5";
        for f in [
            "onnx/model.onnx",
            "tokenizer.json",
            "config.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
        ] {
            download_file(repo, f, &dir.join(f));
        }

        let recipe = Recipe {
            onnx_file: "onnx/model.onnx",
            pooling: Pooling::Cls,
            passage_prefix: "",
            query_prefix: "",
            normalize: true,
            dim: 512,
            max_length: 512,
        };
        let emb = OrtEmbedder::load(&dir, &recipe).unwrap();
        assert_eq!(emb.dim(), 512);

        let passages = emb
            .embed_passages(&["年度预算已经批准", "猫睡在垫子上"])
            .unwrap();
        assert_eq!(passages[0].len(), 512);

        let q = emb.embed_query("预算").unwrap();
        let dot = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>();
        assert!(
            dot(&q, &passages[0]) > dot(&q, &passages[1]),
            "the budget query should match the budget passage more than the cat one"
        );
    }
}
