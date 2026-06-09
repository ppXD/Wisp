//! Cloud text-embedding backend: calls an OpenAI-compatible `/embeddings` endpoint to satisfy
//! [`wisp_library::Embedder`], so semantic search can use a hosted model with the user's own API key
//! instead of a local download. The text-embedding-3 family is symmetric, so no passage/query
//! prefixes are needed. Uses the runtime-free `ureq` client, so it is safe to call from any thread.

use std::time::Duration;

use wisp_library::{Embedder, LibraryError, Result};

/// A hosted embedding model the picker can offer. The API key is looked up separately (by
/// `provider`) so it stays in the app's shared key store rather than in the catalog.
pub struct CloudCatalogModel {
    /// Stable id persisted as the user's choice and used to look the model back up.
    pub id: &'static str,
    /// Human-facing name for the picker.
    pub label: &'static str,
    /// Embedding dimension (vector length stored per chunk).
    pub dim: usize,
    /// Key-store provider id whose API key authorizes this model (e.g. `"openai"`).
    pub provider: &'static str,
    /// OpenAI-compatible API base, no trailing slash, e.g. `https://api.openai.com/v1`.
    pub base_url: &'static str,
    /// The model name sent in the request body.
    pub model: &'static str,
}

/// Built-in hosted models. OpenAI's text-embedding-3 family is the canonical, widely-keyed cloud
/// embedder; more OpenAI-compatible providers slot in as sibling entries.
pub const CLOUD_CATALOG: &[CloudCatalogModel] = &[
    CloudCatalogModel {
        id: "openai-3-small",
        label: "OpenAI · text-embedding-3-small",
        dim: 1536,
        provider: "openai",
        base_url: "https://api.openai.com/v1",
        model: "text-embedding-3-small",
    },
    CloudCatalogModel {
        id: "openai-3-large",
        label: "OpenAI · text-embedding-3-large",
        dim: 3072,
        provider: "openai",
        base_url: "https://api.openai.com/v1",
        model: "text-embedding-3-large",
    },
];

/// Looks up a hosted model by its stable id.
pub fn cloud_catalog_model(id: &str) -> Option<&'static CloudCatalogModel> {
    CLOUD_CATALOG.iter().find(|m| m.id == id)
}

/// An embedder backed by a hosted OpenAI-compatible `/embeddings` endpoint.
pub struct CloudEmbedder {
    agent: ureq::Agent,
    endpoint: String,
    model: String,
    api_key: String,
    dim: usize,
}

impl CloudEmbedder {
    /// Builds an embedder for an explicit endpoint + model.
    pub fn new(base_url: &str, model: &str, api_key: &str, dim: usize) -> Self {
        let agent = ureq::AgentBuilder::new()
            .timeout_connect(Duration::from_secs(15))
            .timeout(Duration::from_secs(60))
            .build();
        Self {
            agent,
            endpoint: format!("{}/embeddings", base_url.trim_end_matches('/')),
            model: model.to_owned(),
            api_key: api_key.to_owned(),
            dim,
        }
    }

    /// Builds an embedder for a catalog model with the given API key.
    pub fn from_catalog(model: &CloudCatalogModel, api_key: &str) -> Self {
        Self::new(model.base_url, model.model, api_key, model.dim)
    }

    /// POSTs `inputs` to the endpoint and returns one L2-normalized vector per input, in input order.
    fn embed_all(&self, inputs: &[&str]) -> Result<Vec<Vec<f32>>> {
        if inputs.is_empty() {
            return Ok(Vec::new());
        }

        let body = serde_json::json!({ "model": self.model, "input": inputs });

        let response = self
            .agent
            .post(&self.endpoint)
            .set("Authorization", &format!("Bearer {}", self.api_key))
            .send_json(body)
            .map_err(|e| LibraryError::Embed(request_error(e)))?;

        let value: serde_json::Value = response.into_json().map_err(|e| {
            LibraryError::Embed(format!("cloud embeddings: bad JSON response: {e}"))
        })?;

        parse_embeddings(&value, self.dim)
    }
}

impl Embedder for CloudEmbedder {
    fn dim(&self) -> usize {
        self.dim
    }

    fn embed_passages(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        self.embed_all(texts)
    }

    fn embed_query(&self, text: &str) -> Result<Vec<f32>> {
        Ok(self.embed_all(&[text])?.pop().unwrap_or_default())
    }
}

/// Turns a `ureq` failure into a message that surfaces the API's own error body (e.g. an invalid-key
/// 401), which is far more useful to the user than a bare transport string.
fn request_error(e: ureq::Error) -> String {
    match e {
        ureq::Error::Status(code, response) => {
            let body = response.into_string().unwrap_or_default();
            format!("cloud embeddings HTTP {code}: {}", body.trim())
        }
        ureq::Error::Transport(t) => format!("cloud embeddings request failed: {t}"),
    }
}

/// Parses an OpenAI-style `{ "data": [{ "index": i, "embedding": [...] }] }` body into one
/// L2-normalized vector per input, ordered by `index`. Errors if the shape is wrong or a vector's
/// length doesn't match the expected dimension.
fn parse_embeddings(value: &serde_json::Value, dim: usize) -> Result<Vec<Vec<f32>>> {
    let data = value
        .get("data")
        .and_then(|d| d.as_array())
        .ok_or_else(|| {
            LibraryError::Embed("cloud embeddings: response has no data array".to_owned())
        })?;

    let mut indexed: Vec<(u64, Vec<f32>)> = Vec::with_capacity(data.len());
    for item in data {
        let embedding = item
            .get("embedding")
            .and_then(|e| e.as_array())
            .ok_or_else(|| {
                LibraryError::Embed("cloud embeddings: item has no embedding".to_owned())
            })?;

        let mut vector: Vec<f32> = embedding
            .iter()
            .map(|x| x.as_f64().unwrap_or(0.0) as f32)
            .collect();

        if vector.len() != dim {
            return Err(LibraryError::Embed(format!(
                "cloud embeddings: expected dim {dim}, got {}",
                vector.len()
            )));
        }

        normalize(&mut vector);
        let index = item
            .get("index")
            .and_then(|i| i.as_u64())
            .unwrap_or(indexed.len() as u64);
        indexed.push((index, vector));
    }

    indexed.sort_by_key(|(i, _)| *i);
    Ok(indexed.into_iter().map(|(_, v)| v).collect())
}

/// Scales a vector to unit L2 length in place, so a dot product equals cosine similarity. A zero
/// vector is left unchanged.
fn normalize(v: &mut [f32]) {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in v {
            *x /= norm;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn catalog_ids_are_unique_and_findable() {
        let mut seen = std::collections::HashSet::new();
        for m in CLOUD_CATALOG {
            assert!(seen.insert(m.id), "duplicate cloud id {}", m.id);
            assert_eq!(cloud_catalog_model(m.id).unwrap().id, m.id);
        }
        assert!(cloud_catalog_model("nope").is_none());
    }

    #[test]
    fn new_builds_the_embeddings_endpoint_and_trims_slash() {
        let e = CloudEmbedder::new("https://api.openai.com/v1/", "m", "k", 3);
        assert_eq!(e.endpoint, "https://api.openai.com/v1/embeddings");
        assert_eq!(e.dim(), 3);
    }

    #[test]
    fn parse_orders_by_index_and_normalizes() {
        let body = serde_json::json!({
            "data": [
                { "index": 1, "embedding": [0.0, 3.0, 4.0] },
                { "index": 0, "embedding": [3.0, 0.0, 4.0] },
            ]
        });
        let out = parse_embeddings(&body, 3).unwrap();
        assert_eq!(out.len(), 2);

        // The item with index 0 was second in the array but must sort first.
        assert!((out[0][0] - 0.6).abs() < 1e-6); // 3/5
        assert!((out[0][2] - 0.8).abs() < 1e-6); // 4/5

        for v in &out {
            let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!((n - 1.0).abs() < 1e-6, "vector not L2-normalized: {n}");
        }
    }

    #[test]
    fn parse_rejects_a_dimension_mismatch() {
        let body = serde_json::json!({ "data": [{ "index": 0, "embedding": [1.0, 2.0] }] });
        assert!(parse_embeddings(&body, 3).is_err());
    }

    #[test]
    fn parse_rejects_a_missing_data_array() {
        let body = serde_json::json!({ "error": { "message": "bad key" } });
        assert!(parse_embeddings(&body, 3).is_err());
    }
}
