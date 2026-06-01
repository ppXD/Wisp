//! Vendor-agnostic description of a cloud transcription provider and its models.
//!
//! Cloud transcription is opt-in — Wisp runs on-device by default. A [`CloudProvider`] names the
//! HTTP API shape it speaks ([`CloudProtocol`], which selects the engine adapter), a default
//! `base_url` (overridable, so OpenAI-compatible vendors reuse one adapter), how to authenticate
//! ([`CloudAuth`]), and the models it offers. A new vendor is just new catalog data; a new API
//! shape is one new `CloudProtocol` variant plus its adapter — neither touches existing consumers.

/// The HTTP API shape a provider speaks, selecting the engine adapter that talks to it.
///
/// `#[non_exhaustive]`: new protocols (Deepgram, AssemblyAI, …) are added without breaking matches.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum CloudProtocol {
    /// OpenAI `/v1/audio/transcriptions` — and every OpenAI-compatible endpoint (e.g. Groq),
    /// reached by overriding `base_url`.
    OpenAi,
    /// Google Gemini `models/{model}:generateContent` with inline base64 audio; the key is a `?key=`
    /// query parameter and the transcript comes back as the candidate's text.
    Gemini,
    /// OpenAI-compatible `/chat/completions` carrying the audio as an `input_audio` content part
    /// (e.g. Alibaba DashScope's Qwen audio models). Bearer-authed; transcript is the message content.
    OpenAiChatAudio,
}

/// How to authenticate a request: a header carrying the API key, with a scheme prefix.
///
/// Covers the common header-credential schemes — `Authorization: Bearer <key>` (OpenAI),
/// `Authorization: Token <key>` (Deepgram), or a bare key — without baking any vendor in.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CloudAuth {
    /// The header that carries the credential, e.g. `"Authorization"`.
    pub header: String,
    /// Text prefixed to the key in that header, e.g. `"Bearer "`; empty for a bare key.
    pub scheme: String,
}

impl CloudAuth {
    /// `Authorization: Bearer <key>` — OpenAI and every OpenAI-compatible vendor.
    pub fn bearer() -> Self {
        Self {
            header: "Authorization".to_owned(),
            scheme: "Bearer ".to_owned(),
        }
    }

    /// The header value carrying `key` under this scheme, e.g. `"Bearer sk-…"`.
    pub fn header_value(&self, key: &str) -> String {
        format!("{}{key}", self.scheme)
    }
}

/// One model a [`CloudProvider`] offers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CloudModel {
    /// Wire id sent to the API, e.g. `"gpt-4o-transcribe"`.
    pub id: String,
    /// Human-readable name for the picker.
    pub display_name: String,
    /// Supports real-time streaming transcription (the Live path).
    pub streaming: bool,
    /// Supports whole-file batch transcription (the File path).
    pub batch: bool,
    /// Language codes the model handles; empty means multilingual / auto-detect.
    pub languages: Vec<String>,
    /// Short guidance shown in the picker.
    pub description: String,
}

/// A cloud transcription provider and the models it serves.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CloudProvider {
    /// Stable id, e.g. `"openai"`.
    pub id: String,
    /// Human-readable name, e.g. `"OpenAI"`.
    pub display_name: String,
    /// The API shape this provider speaks (selects the adapter).
    pub protocol: CloudProtocol,
    /// Default API endpoint; a user may override it for a self-hosted or compatible gateway.
    pub base_url: String,
    /// Where the user creates their API key (the provider's console "API keys" page) — surfaced as
    /// a "Get a key" link in the key manager.
    pub keys_url: String,
    /// How to authenticate with the user's API key.
    pub auth: CloudAuth,
    /// Models offered, in display order.
    pub models: Vec<CloudModel>,
}

impl CloudProvider {
    /// The model with `id`, if this provider offers it.
    pub fn model(&self, id: &str) -> Option<&CloudModel> {
        self.models.iter().find(|m| m.id == id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bearer_auth_formats_the_header_value() {
        let auth = CloudAuth::bearer();
        assert_eq!(auth.header, "Authorization");
        assert_eq!(auth.header_value("sk-abc"), "Bearer sk-abc");
    }

    #[test]
    fn a_bare_scheme_passes_the_key_verbatim() {
        let auth = CloudAuth {
            header: "Authorization".to_owned(),
            scheme: String::new(),
        };
        assert_eq!(auth.header_value("xyz"), "xyz");
    }

    #[test]
    fn provider_looks_up_its_models_by_id() {
        let provider = CloudProvider {
            id: "p".to_owned(),
            display_name: "P".to_owned(),
            protocol: CloudProtocol::OpenAi,
            base_url: "https://example".to_owned(),
            keys_url: "https://example/keys".to_owned(),
            auth: CloudAuth::bearer(),
            models: vec![CloudModel {
                id: "m1".to_owned(),
                display_name: "M1".to_owned(),
                streaming: true,
                batch: true,
                languages: vec![],
                description: String::new(),
            }],
        };
        assert_eq!(provider.model("m1").unwrap().display_name, "M1");
        assert!(provider.model("nope").is_none());
    }
}
