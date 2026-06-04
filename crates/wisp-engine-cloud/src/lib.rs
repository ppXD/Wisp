//! Cloud transcription engine — uploads audio to a [`CloudProvider`]'s HTTP API and returns the
//! transcript, behind the same [`AsrEngine`] trait as the on-device engines.
//!
//! This is the File/batch path (one POST per call): the engine encodes the 16 kHz audio it is given
//! as a WAV and posts it to the provider's transcription endpoint. The app windows long files (for
//! the provider's upload-size limit and for progress), so each call sees a bounded clip. Live
//! streaming over a realtime socket is a separate, later addition.
//!
//! Only the `OpenAi` protocol is wired today; it also covers every OpenAI-compatible endpoint
//! (e.g. Groq) via the provider's `base_url`. The request building and response parsing are pure and
//! unit-tested; only the HTTP round-trip needs the network.

use std::io::Cursor;
use std::time::Duration;

use wisp_core::cloud::{CloudProtocol, CloudProvider};
use wisp_core::engine::{AsrEngine, ClipOptions, EngineInfo, TranscriptionResult};
use wisp_core::error::{Result, WispError};
use wisp_core::transcript::{AudioSourceKind, TranscriptSegment};

/// An [`AsrEngine`] backed by a cloud transcription API.
pub struct CloudEngine {
    /// Full transcription endpoint, e.g. `https://api.openai.com/v1/audio/transcriptions`.
    endpoint: String,
    /// Auth header name and its value (scheme + key), e.g. `("Authorization", "Bearer sk-…")`.
    auth_header: String,
    auth_value: String,
    /// Wire model id, e.g. `gpt-4o-transcribe`.
    model: String,
    /// Language code, or empty to let the provider auto-detect.
    language: String,
}

impl CloudEngine {
    /// Builds an engine for `model_id` of `provider`, authenticating with `api_key`. `language` is a
    /// provider language code, or empty for auto-detect. Errors if the model is unknown, can't do
    /// file transcription, or speaks a protocol not wired yet.
    pub fn new(
        provider: &CloudProvider,
        model_id: &str,
        api_key: &str,
        language: &str,
    ) -> Result<Self> {
        let model = provider
            .model(model_id)
            .ok_or_else(|| WispError::Engine(format!("{} has no model {model_id}", provider.id)))?;
        if !model.batch {
            return Err(WispError::Engine(format!(
                "{model_id} does not support file transcription"
            )));
        }
        if api_key.trim().is_empty() {
            return Err(WispError::Engine(format!(
                "{} needs an API key",
                provider.display_name
            )));
        }

        let path = match provider.protocol {
            CloudProtocol::OpenAi => "/audio/transcriptions",
            other => {
                return Err(WispError::Engine(format!(
                    "cloud protocol {other:?} is not supported yet"
                )))
            }
        };

        Ok(Self {
            endpoint: format!("{}{path}", provider.base_url.trim_end_matches('/')),
            auth_header: provider.auth.header.clone(),
            auth_value: provider.auth.header_value(api_key),
            model: model_id.to_owned(),
            language: language.to_owned(),
        })
    }

    /// Uploads `audio` and returns the transcript as a single segment spanning the clip.
    fn transcribe_upload(&self, audio: &[f32], sample_rate: u32) -> Result<TranscriptionResult> {
        let wav = encode_wav_16bit(audio, sample_rate);
        let (content_type, body) = build_multipart(&wav, &self.model, &self.language);

        let response = ureq::post(&self.endpoint)
            .set(&self.auth_header, &self.auth_value)
            .set("Content-Type", &content_type)
            .timeout(Duration::from_secs(300))
            .send_bytes(&body)
            .map_err(|e| WispError::Engine(format!("cloud transcribe request: {e}")))?;

        let json = response
            .into_string()
            .map_err(|e| WispError::Engine(format!("cloud transcribe read: {e}")))?;

        Ok(to_result(&parse_transcription(&json)?, audio, sample_rate))
    }
}

impl AsrEngine for CloudEngine {
    fn info(&self) -> EngineInfo {
        EngineInfo {
            name: format!("cloud:{}", self.model),
            streaming: false,
        }
    }

    fn transcribe(&mut self, audio: &[f32], sample_rate: u32) -> Result<TranscriptionResult> {
        self.transcribe_upload(audio, sample_rate)
    }

    fn transcribe_clip(
        &mut self,
        audio: &[f32],
        sample_rate: u32,
        _options: ClipOptions<'_>,
    ) -> Result<TranscriptionResult> {
        self.transcribe_upload(audio, sample_rate)
    }
}

/// Encodes 16 kHz mono `f32` `audio` (range −1..1) as in-memory 16-bit PCM WAV bytes.
fn encode_wav_16bit(audio: &[f32], sample_rate: u32) -> Vec<u8> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };

    let mut cursor = Cursor::new(Vec::new());
    {
        let mut writer = hound::WavWriter::new(&mut cursor, spec).expect("in-memory WAV writer");
        for &sample in audio {
            let pcm = (sample.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
            writer
                .write_sample(pcm)
                .expect("write WAV sample to memory");
        }
        writer.finalize().expect("finalize in-memory WAV");
    }
    cursor.into_inner()
}

/// Appends one text form-data field to a multipart body.
fn append_field(body: &mut Vec<u8>, boundary: &str, name: &str, value: &str) {
    body.extend_from_slice(
        format!(
            "--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n"
        )
        .as_bytes(),
    );
}

/// Builds the `multipart/form-data` body to upload `wav` for `model` (with `language` if non-empty),
/// returning the `Content-Type` header value and the body bytes.
fn build_multipart(wav: &[u8], model: &str, language: &str) -> (String, Vec<u8>) {
    // A fixed boundary is safe here: it is far longer and more specific than anything that occurs in
    // 16-bit PCM audio or the short text fields.
    const BOUNDARY: &str = "----wispMultipartBoundaryqZ3xK9pLrT8vB2";

    let mut body = Vec::new();
    append_field(&mut body, BOUNDARY, "model", model);
    if !language.is_empty() {
        append_field(&mut body, BOUNDARY, "language", language);
    }
    append_field(&mut body, BOUNDARY, "response_format", "json");

    body.extend_from_slice(
        format!(
            "--{BOUNDARY}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"audio.wav\"\r\nContent-Type: audio/wav\r\n\r\n"
        )
        .as_bytes(),
    );
    body.extend_from_slice(wav);
    body.extend_from_slice(format!("\r\n--{BOUNDARY}--\r\n").as_bytes());

    (format!("multipart/form-data; boundary={BOUNDARY}"), body)
}

/// The transcript text from a provider response. OpenAI-shaped APIs return `{ "text": "…" }`.
fn parse_transcription(json: &str) -> Result<String> {
    #[derive(serde::Deserialize)]
    struct Response {
        text: String,
    }

    let response: Response = serde_json::from_str(json).map_err(|e| {
        let snippet: String = json.chars().take(200).collect();
        WispError::Engine(format!(
            "cloud response was not transcription JSON: {e} — got: {snippet}"
        ))
    })?;
    Ok(response.text)
}

/// Wraps `text` (spanning `audio` at `sample_rate`) as a single File segment; empty text → empty.
fn to_result(text: &str, audio: &[f32], sample_rate: u32) -> TranscriptionResult {
    let text = text.trim();
    if text.is_empty() {
        return TranscriptionResult::empty();
    }
    let secs = audio.len() as f32 / sample_rate.max(1) as f32;
    let segment = TranscriptSegment::new(
        0,
        text,
        Duration::ZERO..Duration::from_secs_f32(secs),
        AudioSourceKind::File,
    );
    TranscriptionResult {
        segments: vec![segment],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wisp_core::cloud::{CloudAuth, CloudModel, CloudProtocol, CloudProvider};

    fn provider() -> CloudProvider {
        CloudProvider {
            id: "openai".to_owned(),
            display_name: "OpenAI".to_owned(),
            protocol: CloudProtocol::OpenAi,
            base_url: "https://api.example.com/v1/".to_owned(), // trailing slash, to test trimming
            keys_url: "https://example.com/keys".to_owned(),
            auth: CloudAuth::bearer(),
            models: vec![
                CloudModel {
                    id: "gpt-4o-transcribe".to_owned(),
                    display_name: "GPT-4o".to_owned(),
                    streaming: true,
                    batch: true,
                    languages: vec![],
                    description: String::new(),
                },
                CloudModel {
                    id: "stream-only".to_owned(),
                    display_name: "Stream only".to_owned(),
                    streaming: true,
                    batch: false,
                    languages: vec![],
                    description: String::new(),
                },
            ],
        }
    }

    #[test]
    fn new_builds_endpoint_and_auth_and_rejects_bad_inputs() {
        let engine = CloudEngine::new(&provider(), "gpt-4o-transcribe", "sk-abc", "en").unwrap();
        assert_eq!(
            engine.endpoint,
            "https://api.example.com/v1/audio/transcriptions"
        );
        assert_eq!(engine.auth_header, "Authorization");
        assert_eq!(engine.auth_value, "Bearer sk-abc");

        // Unknown model, a non-batch model, and a blank key are all rejected.
        assert!(CloudEngine::new(&provider(), "nope", "sk", "").is_err());
        assert!(CloudEngine::new(&provider(), "stream-only", "sk", "").is_err());
        assert!(CloudEngine::new(&provider(), "gpt-4o-transcribe", "  ", "").is_err());
    }

    #[test]
    fn wav_roundtrips_through_hound() {
        let audio = vec![0.0, 0.5, -0.5, 1.0, -1.0];
        let bytes = encode_wav_16bit(&audio, 16_000);

        let reader = hound::WavReader::new(Cursor::new(bytes)).unwrap();
        assert_eq!(reader.spec().sample_rate, 16_000);
        assert_eq!(reader.spec().channels, 1);
        assert_eq!(reader.spec().bits_per_sample, 16);
        assert_eq!(reader.len(), audio.len() as u32);
    }

    #[test]
    fn multipart_carries_model_file_and_optional_language() {
        let wav = b"RIFFfakewavdata".to_vec();
        let (content_type, body) = build_multipart(&wav, "gpt-4o-transcribe", "en");
        let text = String::from_utf8_lossy(&body);

        assert!(content_type.starts_with("multipart/form-data; boundary="));
        assert!(text.contains("name=\"model\"") && text.contains("gpt-4o-transcribe"));
        assert!(text.contains("name=\"language\"") && text.contains("\r\nen\r\n"));
        assert!(
            text.contains("filename=\"audio.wav\"") && text.contains("Content-Type: audio/wav")
        );
        assert!(text.contains("RIFFfakewavdata"));
        assert!(text.trim_end().ends_with("--")); // closing boundary

        // Empty language omits the field entirely.
        let (_, body) = build_multipart(&wav, "m", "");
        assert!(!String::from_utf8_lossy(&body).contains("name=\"language\""));
    }

    #[test]
    fn parses_transcription_text_and_errors_on_garbage() {
        assert_eq!(
            parse_transcription(r#"{"text":"hello world"}"#).unwrap(),
            "hello world"
        );
        assert!(parse_transcription(r#"{"error":"bad key"}"#).is_err());
        assert!(parse_transcription("not json").is_err());
    }

    #[test]
    fn to_result_is_one_segment_or_empty() {
        let audio = vec![0.0f32; 32_000]; // 2 s at 16 kHz
        let result = to_result("  hi there  ", &audio, 16_000);
        assert_eq!(result.segments.len(), 1);
        assert_eq!(result.segments[0].text, "hi there");

        assert!(to_result("   ", &audio, 16_000).segments.is_empty());
    }
}
