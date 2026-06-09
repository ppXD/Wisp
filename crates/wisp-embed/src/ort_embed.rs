//! Generic recipe-driven ONNX embedder: runs ANY local ONNX text-embedding model — encoder
//! (CLS / Mean pooling) or decoder (last-token, e.g. Qwen3-Embedding) — directly on ONNX Runtime via
//! `ort`, with the tokenizer + pooling + prompts described by a small [`Recipe`]. This is the unified
//! local backend; a model's files are downloaded into a per-model dir and loaded from there.
//!
//! The inference path mirrors fastembed's (tokenize → `input_ids`/`attention_mask`/`token_type_ids`
//! → run → pool → L2-normalize), but with full control over pooling + prompts so decoder models work
//! too — which fastembed's CLS/Mean-only pooling cannot express.

use std::path::Path;

use ndarray::{Array2, Array4, ArrayViewD};
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

/// For decoder models exported with a KV cache (Qwen3-Embedding et al.): the `past_key_values.*` input
/// names plus the per-head dims needed to feed an EMPTY cache (`past_sequence_length = 0`) on the
/// single forward pass we run for embeddings. Encoder models have none of these and skip it entirely.
struct KvCache {
    names: Vec<String>,
    num_heads: usize,
    head_dim: usize,
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
    needs_position_ids: bool,
    kv_cache: Option<KvCache>,
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
        let needs_position_ids = session.inputs.iter().any(|i| i.name == "position_ids");
        let kv_cache = detect_kv_cache(&session, model_dir)?;

        Ok(Self {
            session,
            tokenizer,
            pooling: recipe.pooling,
            passage_prefix: recipe.passage_prefix.to_owned(),
            query_prefix: recipe.query_prefix.to_owned(),
            normalize: recipe.normalize,
            dim: recipe.dim,
            needs_token_type_ids,
            needs_position_ids,
            kv_cache,
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
        if self.needs_position_ids {
            inputs.push((
                "position_ids".into(),
                Value::from_array(position_ids_from_mask(&mask_arr))
                    .map_err(ort_err)?
                    .into(),
            ));
        }
        if let Some(kv) = &self.kv_cache {
            // Prefill forward: feed an empty `[batch, heads, 0, head_dim]` cache for every layer.
            let empty = Array4::<f32>::zeros((batch, kv.num_heads, 0, kv.head_dim));
            for name in &kv.names {
                inputs.push((
                    name.as_str().into(),
                    Value::from_array(empty.view()).map_err(ort_err)?.into(),
                ));
            }
        }

        let outputs = self.session.run(inputs).map_err(ort_err)?;
        let key = pick_output(outputs.keys().collect::<Vec<_>>())?;
        let tensor: ArrayViewD<f32> = outputs
            .get(key.as_str())
            .ok_or_else(|| LibraryError::Embed("model output missing".to_owned()))?
            .try_extract_tensor::<f32>()
            .map_err(ort_err)?;

        let pooled = pool_tokens(&tensor, &mask_arr, self.pooling)?;
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
}

/// Collapses `[batch, seq, hidden]` token embeddings to `[batch, hidden]` per `pooling`. A model that
/// already emits a pooled `[batch, hidden]` tensor is passed through unchanged. Pure (no model), so it
/// is unit-tested directly with synthetic tensors.
fn pool_tokens(
    tensor: &ArrayViewD<f32>,
    mask: &Array2<i64>,
    pooling: Pooling,
) -> Result<Array2<f32>> {
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
        match pooling {
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

/// Builds `position_ids` from the attention mask for decoder models (Qwen3 et al.) that take it as an
/// explicit input: each real token gets its 0-based index among the unmasked tokens; padding holds the
/// running index. Equals a plain `arange` for right-padded batches and stays correct under
/// left-padding. Pure (no model), so it is unit-tested directly.
fn position_ids_from_mask(mask: &Array2<i64>) -> Array2<i64> {
    let (batch, seq) = (mask.shape()[0], mask.shape()[1]);
    let mut pos = Array2::<i64>::zeros((batch, seq));
    for b in 0..batch {
        let mut next = 0i64;
        for t in 0..seq {
            pos[[b, t]] = next;
            if mask[[b, t]] != 0 {
                next += 1;
            }
        }
    }
    pos
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

/// Detects whether `session` is a decoder exported with a KV cache (has `past_key_values.*` inputs)
/// and, if so, the dims of the empty cache to feed. Returns `None` for encoder models (which have no
/// such inputs and run unchanged).
fn detect_kv_cache(session: &Session, model_dir: &Path) -> Result<Option<KvCache>> {
    let names: Vec<String> = session
        .inputs
        .iter()
        .map(|i| i.name.clone())
        .filter(|n| n.starts_with("past_key_values"))
        .collect();
    if names.is_empty() {
        return Ok(None);
    }

    let (num_heads, head_dim) = read_kv_dims(model_dir).ok_or_else(|| {
        LibraryError::Embed(
            "decoder model has a KV cache but config.json lacks num_key_value_heads / head_dim"
                .to_owned(),
        )
    })?;

    Ok(Some(KvCache {
        names,
        num_heads,
        head_dim,
    }))
}

/// Reads `(num_key_value_heads, head_dim)` from `config.json` — the per-head dims of the empty KV
/// cache a decoder model wants. Falls back to `num_attention_heads` and `hidden_size /
/// num_attention_heads` when the explicit fields are absent. Pure, so it is unit-tested directly.
fn read_kv_dims(dir: &Path) -> Option<(usize, usize)> {
    let bytes = std::fs::read(dir.join("config.json")).ok()?;
    let cfg: serde_json::Value = serde_json::from_slice(&bytes).ok()?;

    let attn_heads = cfg.get("num_attention_heads").and_then(|v| v.as_u64());
    let kv_heads = cfg
        .get("num_key_value_heads")
        .and_then(|v| v.as_u64())
        .or(attn_heads)? as usize;

    let head_dim = match cfg.get("head_dim").and_then(|v| v.as_u64()) {
        Some(h) => h as usize,
        None => (cfg.get("hidden_size")?.as_u64()? / attn_heads?) as usize,
    };

    (kv_heads > 0 && head_dim > 0).then_some((kv_heads, head_dim))
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

    fn dyn3(shape: [usize; 3], data: Vec<f32>) -> ndarray::ArrayD<f32> {
        ndarray::Array::from_shape_vec(ndarray::IxDyn(&shape), data).unwrap()
    }
    fn mask(shape: (usize, usize), data: Vec<i64>) -> Array2<i64> {
        Array2::from_shape_vec(shape, data).unwrap()
    }

    #[test]
    fn pool_cls_takes_the_first_token() {
        // tokens: [1,2], [3,4], [5,6]
        let t = dyn3([1, 3, 2], vec![1., 2., 3., 4., 5., 6.]);
        let m = mask((1, 3), vec![1, 1, 1]);
        let out = pool_tokens(&t.view(), &m, Pooling::Cls).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(out[[0, 0]], 1.0);
        assert_eq!(out[[0, 1]], 2.0);
    }

    #[test]
    fn pool_mean_averages_only_unmasked_tokens() {
        // the third token is padding and must be ignored, despite its huge values.
        let t = dyn3([1, 3, 2], vec![1., 2., 3., 4., 100., 100.]);
        let m = mask((1, 3), vec![1, 1, 0]);
        let out = pool_tokens(&t.view(), &m, Pooling::Mean).unwrap();
        assert_eq!(out[[0, 0]], 2.0); // (1+3)/2
        assert_eq!(out[[0, 1]], 3.0); // (2+4)/2
    }

    #[test]
    fn pool_mean_with_all_padding_is_zero_not_nan() {
        let t = dyn3([1, 2, 2], vec![1., 2., 3., 4.]);
        let m = mask((1, 2), vec![0, 0]);
        let out = pool_tokens(&t.view(), &m, Pooling::Mean).unwrap();
        assert_eq!(out[[0, 0]], 0.0);
        assert!(out[[0, 1]].is_finite());
    }

    #[test]
    fn pool_last_token_takes_the_last_unmasked_token() {
        // tokens 0..3, padding at the end — the last real token is index 2.
        let t = dyn3([1, 4, 2], vec![1., 1., 2., 2., 3., 3., 9., 9.]);
        let m = mask((1, 4), vec![1, 1, 1, 0]);
        let out = pool_tokens(&t.view(), &m, Pooling::LastToken).unwrap();
        assert_eq!(out[[0, 0]], 3.0);
        assert_eq!(out[[0, 1]], 3.0);
    }

    #[test]
    fn pool_handles_a_multi_row_batch_with_distinct_masks() {
        // row0 uses both tokens, row1 only the first.
        let t = dyn3([2, 2, 1], vec![1., 3., 10., 20.]);
        let m = mask((2, 2), vec![1, 1, 1, 0]);
        let out = pool_tokens(&t.view(), &m, Pooling::Mean).unwrap();
        assert_eq!(out[[0, 0]], 2.0); // (1+3)/2
        assert_eq!(out[[1, 0]], 10.0); // only the first token counts
    }

    #[test]
    fn pool_passes_through_a_prepooled_2d_output() {
        let t =
            ndarray::Array::from_shape_vec(ndarray::IxDyn(&[2, 3]), vec![1., 2., 3., 4., 5., 6.])
                .unwrap();
        let m = mask((2, 1), vec![1, 1]);
        let out = pool_tokens(&t.view(), &m, Pooling::Mean).unwrap();
        assert_eq!(out.shape(), &[2, 3]);
        assert_eq!(out[[1, 2]], 6.0);
    }

    #[test]
    fn pool_rejects_an_unexpected_output_rank() {
        let t = ndarray::Array::from_shape_vec(ndarray::IxDyn(&[4]), vec![1., 2., 3., 4.]).unwrap();
        let m = mask((1, 4), vec![1, 1, 1, 1]);
        assert!(pool_tokens(&t.view(), &m, Pooling::Mean).is_err());
    }

    #[test]
    fn position_ids_index_real_tokens_per_row() {
        // row0 right-padded (real tokens 0,1,2 then pad); row1 left-padded (pad, pad, then real).
        let m = mask((2, 4), vec![1, 1, 1, 0, 0, 0, 1, 1]);
        let p = position_ids_from_mask(&m);
        assert_eq!(p.shape(), &[2, 4]);
        // right-padding: real tokens count up from 0.
        assert_eq!([p[[0, 0]], p[[0, 1]], p[[0, 2]]], [0, 1, 2]);
        // left-padding: the first real token still starts at 0, not at its absolute index.
        assert_eq!([p[[1, 2]], p[[1, 3]]], [0, 1]);
    }

    #[test]
    fn kv_dims_read_explicit_and_derive_head_dim() {
        let dir = std::env::temp_dir().join(format!("wisp-ort-kvdims-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();

        // Explicit head_dim + grouped-query KV heads (Qwen3-0.6B shape).
        std::fs::write(
            dir.join("config.json"),
            br#"{"num_attention_heads": 16, "num_key_value_heads": 8, "head_dim": 128}"#,
        )
        .unwrap();
        assert_eq!(read_kv_dims(&dir), Some((8, 128)));

        // No head_dim and no GQA: fall back to hidden_size / num_attention_heads and full KV heads.
        std::fs::write(
            dir.join("config.json"),
            br#"{"num_attention_heads": 12, "hidden_size": 768}"#,
        )
        .unwrap();
        assert_eq!(read_kv_dims(&dir), Some((12, 64)));

        std::fs::remove_dir_all(&dir).ok();
        assert_eq!(read_kv_dims(&dir), None); // missing file
    }

    #[test]
    fn max_length_reads_value_and_rejects_sentinels() {
        let dir = std::env::temp_dir().join(format!("wisp-ort-maxlen-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();

        std::fs::write(
            dir.join("tokenizer_config.json"),
            br#"{"model_max_length": 512}"#,
        )
        .unwrap();
        assert_eq!(read_max_length(&dir), Some(512));

        // The 1e30 sentinel some configs use must not be taken literally.
        std::fs::write(
            dir.join("tokenizer_config.json"),
            br#"{"model_max_length": 1e30}"#,
        )
        .unwrap();
        assert_eq!(read_max_length(&dir), None);

        std::fs::remove_dir_all(&dir).ok();
        assert_eq!(read_max_length(&dir), None); // missing file
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

    // Verifies the EXTERNAL-WEIGHTS path: e5-large ships `model.onnx` plus a separate
    // `model.onnx_data`, which ONNX Runtime loads automatically when both sit in the dir. ~2.2 GB.
    #[test]
    #[ignore]
    fn e5_large_embeds_with_external_weights() {
        let dir = std::env::temp_dir().join("wisp-ort-e5-large");
        let repo = "Qdrant/multilingual-e5-large-onnx";
        for f in [
            "model.onnx",
            "model.onnx_data",
            "tokenizer.json",
            "tokenizer_config.json",
            "config.json",
            "special_tokens_map.json",
        ] {
            download_file(repo, f, &dir.join(f));
        }

        let recipe = Recipe {
            onnx_file: "model.onnx",
            pooling: Pooling::Mean,
            passage_prefix: "passage: ",
            query_prefix: "query: ",
            normalize: true,
            dim: 1024,
            max_length: 512,
        };
        let emb = OrtEmbedder::load(&dir, &recipe).unwrap();
        assert_eq!(emb.dim(), 1024);

        let passages = emb
            .embed_passages(&["the annual budget was approved", "the cat slept on the mat"])
            .unwrap();
        assert_eq!(passages[0].len(), 1024);

        let q = emb.embed_query("budget").unwrap();
        let dot = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>();
        assert!(
            dot(&q, &passages[0]) > dot(&q, &passages[1]),
            "the budget query should match the budget passage more than the cat one"
        );
    }

    // Proves the DECODER path end-to-end: last-token pooling + the auto-detected `position_ids` input
    // + Qwen3's instruct query prompt, on the real Qwen3-Embedding-0.6B (int8 ONNX, ~600 MB, to keep
    // the download manageable). This is what fastembed cannot do. `cargo test -- --ignored`.
    #[test]
    #[ignore]
    fn qwen3_embedding_decoder_last_token() {
        let dir = std::env::temp_dir().join("wisp-ort-qwen3-0_6b");
        let repo = "onnx-community/Qwen3-Embedding-0.6B-ONNX";
        for f in [
            "onnx/model_quantized.onnx",
            "tokenizer.json",
            "tokenizer_config.json",
            "config.json",
            "special_tokens_map.json",
        ] {
            download_file(repo, f, &dir.join(f));
        }

        let recipe = Recipe {
            onnx_file: "onnx/model_quantized.onnx",
            pooling: Pooling::LastToken,
            passage_prefix: "",
            query_prefix:
                "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: ",
            normalize: true,
            dim: 1024,
            max_length: 512,
        };
        let emb = OrtEmbedder::load(&dir, &recipe).unwrap();
        assert_eq!(emb.dim(), 1024);

        let passages = emb
            .embed_passages(&["the annual budget was approved", "the cat slept on the mat"])
            .unwrap();
        assert_eq!(passages[0].len(), 1024);
        let norm: f32 = passages[0].iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-3,
            "passage not L2-normalized: {norm}"
        );

        let q = emb.embed_query("budget").unwrap();
        let dot = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>();
        assert!(
            dot(&q, &passages[0]) > dot(&q, &passages[1]),
            "the budget query should rank the budget passage above the cat one"
        );
    }

    // BGE-M3 (XLM-RoBERTa encoder, CLS pooling, 1024-dim, 100+ languages) via the int8 ONNX (~600 MB).
    #[test]
    #[ignore]
    fn bge_m3_embeds_with_cls_pooling() {
        let dir = std::env::temp_dir().join("wisp-ort-bge-m3");
        let repo = "Xenova/bge-m3";
        for f in [
            "onnx/model_quantized.onnx",
            "tokenizer.json",
            "tokenizer_config.json",
            "config.json",
            "special_tokens_map.json",
        ] {
            download_file(repo, f, &dir.join(f));
        }

        let recipe = Recipe {
            onnx_file: "onnx/model_quantized.onnx",
            pooling: Pooling::Cls,
            passage_prefix: "",
            query_prefix: "",
            normalize: true,
            dim: 1024,
            max_length: 512,
        };
        let emb = OrtEmbedder::load(&dir, &recipe).unwrap();
        assert_eq!(emb.dim(), 1024);

        let passages = emb
            .embed_passages(&["年度预算已经批准", "猫睡在垫子上"])
            .unwrap();
        assert_eq!(passages[0].len(), 1024);

        let q = emb.embed_query("预算").unwrap();
        let dot = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>();
        assert!(
            dot(&q, &passages[0]) > dot(&q, &passages[1]),
            "the budget query should rank the budget passage above the cat one"
        );
    }

    // gte-multilingual-base (Alibaba mGTE encoder, CLS pooling, 768-dim) via the fp32 ONNX (~300 MB).
    #[test]
    #[ignore]
    fn gte_multilingual_embeds_with_cls_pooling() {
        let dir = std::env::temp_dir().join("wisp-ort-gte-multilingual");
        let repo = "onnx-community/gte-multilingual-base";
        for f in [
            "onnx/model.onnx",
            "tokenizer.json",
            "tokenizer_config.json",
            "config.json",
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
            dim: 768,
            max_length: 512,
        };
        let emb = OrtEmbedder::load(&dir, &recipe).unwrap();
        assert_eq!(emb.dim(), 768);

        let passages = emb
            .embed_passages(&["the annual budget was approved", "the cat slept on the mat"])
            .unwrap();
        assert_eq!(passages[0].len(), 768);

        let q = emb.embed_query("budget").unwrap();
        let dot = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>();
        assert!(
            dot(&q, &passages[0]) > dot(&q, &passages[1]),
            "the budget query should rank the budget passage above the cat one"
        );
    }
}
