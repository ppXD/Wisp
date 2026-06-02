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
use wisp_core::transcript::{AudioSourceKind, SpeakerId, TranscriptSegment};

/// The OpenAI model that returns diarized (speaker-labelled) segments. File only — realtime
/// transcription events carry no speaker, so this is the one way to get cloud speaker labels.
const OPENAI_DIARIZE_MODEL: &str = "gpt-4o-transcribe-diarize";

mod streaming;
pub use streaming::OpenAiRealtimeEngine;

/// An [`AsrEngine`] backed by a cloud transcription API. One engine spans several wire protocols —
/// the OpenAI transcription endpoint, Gemini's `generateContent`, and OpenAI-compatible
/// chat-with-audio — with the protocol choosing how a clip is uploaded and how the transcript reads
/// back.
pub struct CloudEngine {
    protocol: CloudProtocol,
    /// Base URL without a trailing slash, e.g. `https://api.openai.com/v1`.
    base_url: String,
    /// Auth header name + value for header-authed protocols (`Authorization`, `Bearer …`).
    auth_header: String,
    auth_value: String,
    /// The raw key, for query-param-authed protocols (Gemini's `?key=`).
    api_key: String,
    /// Wire model id, e.g. `gpt-4o-transcribe` / `gemini-2.0-flash` / `qwen3-asr-flash`.
    model: String,
    /// Language code, or empty to let the provider auto-detect.
    language: String,
}

impl CloudEngine {
    /// Builds an engine for `model_id` of `provider`, authenticating with `api_key`. `language` is a
    /// provider language code, or empty for auto-detect. Errors if the model is unknown, can't do
    /// file transcription, or the key is blank.
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

        Ok(Self {
            protocol: provider.protocol,
            base_url: provider.base_url.trim_end_matches('/').to_owned(),
            auth_header: provider.auth.header.clone(),
            auth_value: provider.auth.header_value(api_key),
            api_key: api_key.to_owned(),
            model: model_id.to_owned(),
            language: language.to_owned(),
        })
    }

    /// Uploads `audio` and returns the transcript — one segment spanning the clip, or one per speaker
    /// turn for the diarized model.
    fn transcribe_upload(&self, audio: &[f32], sample_rate: u32) -> Result<TranscriptionResult> {
        let wav = encode_wav_16bit(audio, sample_rate);

        // The diarized OpenAI model returns speaker-labelled segments — parse them directly rather
        // than as one block of text.
        if self.protocol == CloudProtocol::OpenAi && self.model == OPENAI_DIARIZE_MODEL {
            return parse_diarized(&self.post_multipart_raw(&wav)?);
        }

        let text = match self.protocol {
            CloudProtocol::OpenAi => self.post_multipart(&wav),
            CloudProtocol::Gemini => self.post_gemini(&wav),
            CloudProtocol::OpenAiChatAudio => self.post_chat_audio(&wav),
            other => Err(WispError::Engine(format!(
                "cloud protocol {other:?} is not supported yet"
            ))),
        }?;
        Ok(to_result(&text, audio, sample_rate))
    }

    /// OpenAI `/audio/transcriptions`: a multipart upload, returning the raw response body.
    fn post_multipart_raw(&self, wav: &[u8]) -> Result<String> {
        let endpoint = format!("{}/audio/transcriptions", self.base_url);
        let (content_type, body) = build_multipart(wav, &self.model, &self.language);
        let sent = ureq::post(&endpoint)
            .set(&self.auth_header, &self.auth_value)
            .set("Content-Type", &content_type)
            .timeout(Duration::from_secs(300))
            .send_bytes(&body);
        body_or_error(sent)
    }

    /// OpenAI `/audio/transcriptions`: a multipart upload returning `{ "text": … }`.
    fn post_multipart(&self, wav: &[u8]) -> Result<String> {
        parse_openai_transcription(&self.post_multipart_raw(wav)?)
    }

    /// Gemini `generateContent`: JSON with inline base64 audio; the key rides in the query string.
    fn post_gemini(&self, wav: &[u8]) -> Result<String> {
        let url = format!(
            "{}/models/{}:generateContent?key={}",
            self.base_url, self.model, self.api_key
        );
        let sent = ureq::post(&url)
            .set("Content-Type", "application/json")
            .timeout(Duration::from_secs(300))
            .send_string(&build_gemini_body(wav, &self.language));
        parse_gemini(&body_or_error(sent)?)
    }

    /// OpenAI-compatible `/chat/completions` with the audio as an `input_audio` content part.
    fn post_chat_audio(&self, wav: &[u8]) -> Result<String> {
        let endpoint = format!("{}/chat/completions", self.base_url);
        let sent = ureq::post(&endpoint)
            .set(&self.auth_header, &self.auth_value)
            .set("Content-Type", "application/json")
            .timeout(Duration::from_secs(300))
            .send_string(&build_chat_audio_body(wav, &self.model, &self.language));
        parse_chat_completion(&body_or_error(sent)?)
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

    fn reports_clip_progress(&self) -> bool {
        // The diarized model must see the whole file in one upload so its speaker labels stay
        // consistent — windowing would diarize each chunk independently. Reporting here stops the app
        // from windowing the clip (it forwards the whole file to one `transcribe_clip` call instead).
        self.protocol == CloudProtocol::OpenAi && self.model == OPENAI_DIARIZE_MODEL
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

    // The diarized model needs `diarized_json` to return per-speaker segments; everyone else gets the
    // plain `{ "text": … }`.
    let response_format = if model == OPENAI_DIARIZE_MODEL {
        "diarized_json"
    } else {
        "json"
    };
    append_field(&mut body, BOUNDARY, "response_format", response_format);

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

/// The transcript text from an OpenAI-shaped transcription response: `{ "text": "…" }`.
fn parse_openai_transcription(json: &str) -> Result<String> {
    #[derive(serde::Deserialize)]
    struct Response {
        text: String,
    }

    let response: Response = serde_json::from_str(json).map_err(|e| not_json_error(json, e))?;
    Ok(response.text)
}

/// Parses an OpenAI `diarized_json` response (`{ segments: [{ start, end, text, speaker }] }`) into
/// one [`TranscriptSegment`] per speaker turn, with timestamps and a stable speaker index.
fn parse_diarized(json: &str) -> Result<TranscriptionResult> {
    #[derive(serde::Deserialize)]
    struct Response {
        #[serde(default)]
        segments: Vec<DiarizedSegment>,
    }
    #[derive(serde::Deserialize)]
    struct DiarizedSegment {
        #[serde(default)]
        start: f64,
        #[serde(default)]
        end: f64,
        #[serde(default)]
        text: String,
        #[serde(default)]
        speaker: String,
    }

    let response: Response = serde_json::from_str(json).map_err(|e| not_json_error(json, e))?;

    // Map each speaker label ("A"/"B"/a name) to a stable 0-based index by first appearance.
    let mut labels: Vec<String> = Vec::new();
    let mut segments = Vec::new();
    for (index, seg) in response.segments.iter().enumerate() {
        let text = seg.text.trim();
        if text.is_empty() {
            continue;
        }

        let speaker = labels
            .iter()
            .position(|l| *l == seg.speaker)
            .unwrap_or_else(|| {
                labels.push(seg.speaker.clone());
                labels.len() - 1
            });

        let mut segment = TranscriptSegment::new(
            index as u64,
            text,
            Duration::from_secs_f64(seg.start)..Duration::from_secs_f64(seg.end),
            AudioSourceKind::File,
        );
        segment.speaker = Some(SpeakerId(speaker as u32));
        segments.push(segment);
    }

    Ok(TranscriptionResult { segments })
}

/// Standard base64 of `bytes` — the encoding both JSON-body protocols use for the audio.
pub(crate) fn b64(bytes: &[u8]) -> String {
    use base64::Engine;
    base64::engine::general_purpose::STANDARD.encode(bytes)
}

/// The instruction sent to chat-style models (Gemini, Qwen) to get a clean verbatim transcript.
fn transcribe_prompt(language: &str) -> String {
    let base = "Transcribe the audio exactly as spoken, verbatim. Output only the transcript text \
                — no commentary, labels, or quotation marks.";
    if language.is_empty() {
        base.to_owned()
    } else {
        format!("{base} The audio language is \"{language}\".")
    }
}

/// Gemini `generateContent` request body: one user turn with the prompt and inline base64 audio.
fn build_gemini_body(wav: &[u8], language: &str) -> String {
    serde_json::json!({
        "contents": [{
            "role": "user",
            "parts": [
                { "text": transcribe_prompt(language) },
                { "inlineData": { "mimeType": "audio/wav", "data": b64(wav) } }
            ]
        }],
        "generationConfig": { "temperature": 0 }
    })
    .to_string()
}

/// The transcript from a Gemini response — the concatenated text parts of the first candidate.
fn parse_gemini(json: &str) -> Result<String> {
    #[derive(serde::Deserialize)]
    struct Resp {
        #[serde(default)]
        candidates: Vec<Candidate>,
    }
    #[derive(serde::Deserialize)]
    struct Candidate {
        content: Content,
    }
    #[derive(serde::Deserialize)]
    struct Content {
        #[serde(default)]
        parts: Vec<Part>,
    }
    #[derive(serde::Deserialize)]
    struct Part {
        #[serde(default)]
        text: String,
    }

    let resp: Resp = serde_json::from_str(json).map_err(|e| not_json_error(json, e))?;
    Ok(resp
        .candidates
        .first()
        .map(|c| c.content.parts.iter().map(|p| p.text.as_str()).collect())
        .unwrap_or_default())
}

/// OpenAI-compatible `/chat/completions` body carrying the audio as an `input_audio` content part.
fn build_chat_audio_body(wav: &[u8], model: &str, language: &str) -> String {
    serde_json::json!({
        "model": model,
        "messages": [{
            "role": "user",
            "content": [
                { "type": "input_audio", "input_audio": { "data": b64(wav), "format": "wav" } },
                { "type": "text", "text": transcribe_prompt(language) }
            ]
        }]
    })
    .to_string()
}

/// The transcript from an OpenAI-style chat completion: `choices[0].message.content`.
fn parse_chat_completion(json: &str) -> Result<String> {
    #[derive(serde::Deserialize)]
    struct Resp {
        #[serde(default)]
        choices: Vec<Choice>,
    }
    #[derive(serde::Deserialize)]
    struct Choice {
        message: Message,
    }
    #[derive(serde::Deserialize)]
    struct Message {
        #[serde(default)]
        content: String,
    }

    let resp: Resp = serde_json::from_str(json).map_err(|e| not_json_error(json, e))?;
    Ok(resp
        .choices
        .first()
        .map(|c| c.message.content.clone())
        .unwrap_or_default())
}

/// A uniform "response wasn't the JSON we expected" error carrying a short snippet of what we got.
fn not_json_error(json: &str, e: serde_json::Error) -> WispError {
    let snippet: String = json.chars().take(200).collect();
    WispError::Engine(format!(
        "cloud response wasn't the expected JSON: {e} — got: {snippet}"
    ))
}

/// Turns a ureq send result into the response body text, surfacing the API's own error body (key
/// rejected, model unknown, quota exceeded) on a non-2xx status rather than a bare status code.
fn body_or_error(sent: std::result::Result<ureq::Response, ureq::Error>) -> Result<String> {
    match sent {
        Ok(response) => response
            .into_string()
            .map_err(|e| WispError::Engine(format!("cloud transcribe read: {e}"))),
        Err(ureq::Error::Status(code, response)) => {
            let detail: String = response
                .into_string()
                .unwrap_or_default()
                .chars()
                .take(300)
                .collect();
            Err(WispError::Engine(format!(
                "cloud transcribe failed (HTTP {code}): {detail}"
            )))
        }
        Err(e) => Err(WispError::Engine(format!("cloud transcribe request: {e}"))),
    }
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
    fn new_trims_base_url_and_auth_and_rejects_bad_inputs() {
        let engine = CloudEngine::new(&provider(), "gpt-4o-transcribe", "sk-abc", "en").unwrap();
        assert_eq!(engine.base_url, "https://api.example.com/v1"); // trailing slash trimmed
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
    fn parses_openai_transcription_text_and_errors_on_garbage() {
        assert_eq!(
            parse_openai_transcription(r#"{"text":"hello world"}"#).unwrap(),
            "hello world"
        );
        assert!(parse_openai_transcription(r#"{"error":"bad key"}"#).is_err());
        assert!(parse_openai_transcription("not json").is_err());
    }

    #[test]
    fn parse_diarized_maps_speakers_and_drops_blanks() {
        let json = r#"{"segments":[
            {"start":0.0,"end":1.5,"text":"Hello.","speaker":"A"},
            {"start":1.5,"end":3.0,"text":"Hi there.","speaker":"B"},
            {"start":3.0,"end":3.2,"text":"   ","speaker":"A"},
            {"start":3.2,"end":4.0,"text":"Bye.","speaker":"A"}
        ]}"#;
        let result = parse_diarized(json).unwrap();

        assert_eq!(result.segments.len(), 3, "the blank segment is dropped");
        assert_eq!(result.segments[0].text, "Hello.");
        assert_eq!(result.segments[0].speaker, Some(SpeakerId(0)));
        assert_eq!(
            result.segments[1].speaker,
            Some(SpeakerId(1)),
            "new speaker → 1"
        );
        assert_eq!(
            result.segments[2].speaker,
            Some(SpeakerId(0)),
            "speaker A reused → 0"
        );
    }

    #[test]
    fn gemini_body_uses_inline_base64_audio_and_parses_candidate_text() {
        let body = build_gemini_body(b"RIFFfakewav", "yue");
        let v: serde_json::Value = serde_json::from_str(&body).unwrap();
        // Exact Gemini camelCase fields + base64 payload (verified against the API discovery doc).
        let blob = &v["contents"][0]["parts"][1]["inlineData"];
        assert_eq!(blob["mimeType"], "audio/wav");
        assert_eq!(blob["data"], b64(b"RIFFfakewav"));
        assert!(v["contents"][0]["parts"][0]["text"]
            .as_str()
            .unwrap()
            .contains("yue"));

        // Response: candidates[0].content.parts[].text, concatenated.
        assert_eq!(
            parse_gemini(
                r#"{"candidates":[{"content":{"parts":[{"text":"你好"},{"text":"世界"}]}}]}"#
            )
            .unwrap(),
            "你好世界"
        );
        assert_eq!(parse_gemini(r#"{"candidates":[]}"#).unwrap(), "");
    }

    #[test]
    fn chat_audio_body_uses_input_audio_and_parses_message_content() {
        let body = build_chat_audio_body(b"RIFFfakewav", "qwen3-asr-flash", "");
        let v: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(v["model"], "qwen3-asr-flash");
        // The OpenAI `input_audio` content part (verified against OpenAI's OpenAPI spec).
        let audio = &v["messages"][0]["content"][0];
        assert_eq!(audio["type"], "input_audio");
        assert_eq!(audio["input_audio"]["format"], "wav");
        assert_eq!(audio["input_audio"]["data"], b64(b"RIFFfakewav"));

        // Response: choices[0].message.content.
        assert_eq!(
            parse_chat_completion(
                r#"{"choices":[{"message":{"role":"assistant","content":"hi there"}}]}"#
            )
            .unwrap(),
            "hi there"
        );
        assert_eq!(parse_chat_completion(r#"{"choices":[]}"#).unwrap(), "");
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
