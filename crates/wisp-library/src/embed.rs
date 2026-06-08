//! Embedding + hybrid-retrieval primitives for the notes library.
//!
//! The [`Embedder`] trait abstracts "text → vector" so the store stays backend-agnostic: a local
//! ONNX model, a cloud API, or (in tests) a deterministic stub all satisfy it. The free functions
//! are the math the store needs once vectors exist — a dot product for cosine similarity, vector
//! (de)serialization for BLOB storage, transcript chunking, and Reciprocal Rank Fusion for blending
//! the full-text and semantic rankings into one hybrid result.

use crate::Result;

/// Turns text into a fixed-length embedding vector.
///
/// Implementations MUST return L2-normalized vectors of length [`Embedder::dim`], so cosine
/// similarity reduces to a plain dot product ([`dot`]). It is `Send + Sync` so the store can hold
/// one behind the same mutex as its connection.
pub trait Embedder: Send + Sync {
    /// Length of every vector this embedder produces.
    fn dim(&self) -> usize;

    /// Embeds each input string into a `dim()`-length, L2-normalized vector, preserving order.
    fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>>;
}

/// Dot product of two equal-length vectors — cosine similarity when both are L2-normalized. Returns
/// `0.0` if the lengths differ, so a vector stored by a different (wrong-dimension) model is simply
/// treated as unrelated rather than panicking.
pub(crate) fn dot(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Serializes a vector to little-endian bytes for BLOB storage.
pub(crate) fn to_bytes(v: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 4);
    for x in v {
        out.extend_from_slice(&x.to_le_bytes());
    }
    out
}

/// Deserializes a little-endian byte blob back into a vector (trailing partial bytes are ignored).
pub(crate) fn from_bytes(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Groups consecutive segment texts into chunks of at most `max_chars` characters, never splitting
/// an individual segment — so each chunk is a coherent unit to embed. A single segment longer than
/// `max_chars` becomes its own (oversized) chunk; blank segments are skipped. Empty input yields no
/// chunks.
pub(crate) fn chunk_texts(texts: &[&str], max_chars: usize) -> Vec<String> {
    let mut chunks = Vec::new();
    let mut cur = String::new();

    for t in texts {
        let t = t.trim();
        if t.is_empty() {
            continue;
        }
        if !cur.is_empty() && cur.chars().count() + 1 + t.chars().count() > max_chars {
            chunks.push(std::mem::take(&mut cur));
        }
        if !cur.is_empty() {
            cur.push(' ');
        }
        cur.push_str(t);
    }
    if !cur.is_empty() {
        chunks.push(cur);
    }
    chunks
}

/// Reciprocal Rank Fusion. Each ranking lists ids in descending relevance; an id's fused score is
/// the sum over the rankings it appears in of `1 / (k + rank)` (0-based rank). `k` (commonly 60)
/// damps the weight of deep ranks. Returns `(id, score)` pairs, highest score first, ties broken by
/// id for determinism.
pub(crate) fn rrf_fuse(rankings: &[Vec<String>], k: f64) -> Vec<(String, f64)> {
    use std::collections::HashMap;

    let mut scores: HashMap<&str, f64> = HashMap::new();
    for ranking in rankings {
        for (rank, id) in ranking.iter().enumerate() {
            *scores.entry(id.as_str()).or_insert(0.0) += 1.0 / (k + rank as f64);
        }
    }

    let mut out: Vec<(String, f64)> = scores
        .into_iter()
        .map(|(id, s)| (id.to_owned(), s))
        .collect();
    out.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dot_is_cosine_for_normalized_and_zero_on_mismatch() {
        assert!((dot(&[1.0, 0.0], &[1.0, 0.0]) - 1.0).abs() < 1e-6);
        assert!(dot(&[1.0, 0.0], &[0.0, 1.0]).abs() < 1e-6); // orthogonal
        assert_eq!(dot(&[1.0, 2.0, 3.0], &[1.0, 2.0]), 0.0); // length mismatch → 0
    }

    #[test]
    fn vector_bytes_round_trip() {
        let v = vec![0.0_f32, 1.5, -2.25, 1e9, -1e-9];
        assert_eq!(from_bytes(&to_bytes(&v)), v);
        assert!(from_bytes(&[]).is_empty());
    }

    #[test]
    fn chunk_texts_groups_without_splitting_segments() {
        // Three short segments fit in one chunk.
        assert_eq!(chunk_texts(&["aa", "bb", "cc"], 100), vec!["aa bb cc"]);
        // Budget forces a new chunk, but never mid-segment.
        assert_eq!(
            chunk_texts(&["aaaa", "bbbb", "cccc"], 9),
            vec!["aaaa bbbb", "cccc"]
        );
        // A single over-budget segment becomes its own chunk.
        assert_eq!(chunk_texts(&["abcdefghij"], 4), vec!["abcdefghij"]);
        // Blanks are skipped; empty input yields nothing.
        assert_eq!(chunk_texts(&["", "  ", "x"], 100), vec!["x"]);
        assert!(chunk_texts(&[], 100).is_empty());
    }

    #[test]
    fn chunk_texts_is_char_safe_for_cjk() {
        // Budget counts characters, not bytes, so multi-byte segments group correctly.
        let chunks = chunk_texts(&["预算讨论", "排期计划"], 5);
        assert_eq!(chunks, vec!["预算讨论".to_owned(), "排期计划".to_owned()]);
    }

    #[test]
    fn rrf_prefers_ids_ranked_high_in_multiple_lists() {
        let a = vec!["x".to_owned(), "y".to_owned(), "z".to_owned()];
        let b = vec!["y".to_owned(), "w".to_owned()];
        let fused = rrf_fuse(&[a, b], 60.0);
        let order: Vec<&str> = fused.iter().map(|(id, _)| id.as_str()).collect();
        // y is in both lists (and ranked high in b) → it wins.
        assert_eq!(order[0], "y");
        // Every id from either list appears exactly once.
        assert_eq!(fused.len(), 4);
    }

    #[test]
    fn rrf_single_list_keeps_its_order_and_empty_is_empty() {
        let single = vec!["a".to_owned(), "b".to_owned(), "c".to_owned()];
        let fused = rrf_fuse(&[single], 60.0);
        let order: Vec<&str> = fused.iter().map(|(id, _)| id.as_str()).collect();
        assert_eq!(order, ["a", "b", "c"]);
        assert!(rrf_fuse(&[], 60.0).is_empty());
    }
}
