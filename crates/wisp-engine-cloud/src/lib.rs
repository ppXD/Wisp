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

// `ureq::Error` is large (it embeds a `Response`), so any `Result` carrying it trips clippy's
// `result_large_err`. These are infrequent network round-trips, not a hot path, so the `Result` size
// is irrelevant — boxing every cloud error would only obscure the code. Allow it crate-wide.
#![allow(clippy::result_large_err)]

use std::io::{BufRead, BufReader, Cursor};
use std::time::Duration;

use wisp_core::cloud::{CloudProtocol, CloudProvider};
use wisp_core::engine::{AsrEngine, ClipOptions, EngineInfo, TranscriptionResult};
use wisp_core::error::{Result, WispError};
use wisp_core::params::{ParamSpec, ParamValues};
use wisp_core::transcript::{AudioSourceKind, SpeakerId, TranscriptSegment};

/// The OpenAI model that returns diarized (speaker-labelled) segments. File only — realtime
/// transcription events carry no speaker, so this is the one way to get cloud speaker labels.
const OPENAI_DIARIZE_MODEL: &str = "gpt-4o-transcribe-diarize";

mod streaming;
pub use streaming::{
    assist_realtime_param_specs, build_assist_engine, build_realtime_engine, streaming_param_specs,
    ErrorSink, REALTIME_URL_ENV,
};

/// The advanced File (batch) parameters a cloud `protocol` exposes for `model`, mirroring
/// [`streaming_param_specs`] for live. The generic settings panel renders these; an empty list (a
/// protocol with no extra knobs) shows nothing. The biasing prompt and language ride in from the
/// shared File options, so they are not repeated here.
pub fn batch_param_specs(protocol: CloudProtocol, _model: &str) -> Vec<ParamSpec> {
    match protocol {
        // OpenAI's transcription endpoint caps temperature at 1; Gemini and the OpenAI-compatible
        // chat endpoint (Qwen) allow up to 2. Match each vendor's documented range.
        CloudProtocol::OpenAi => vec![temperature_spec(1.0)],
        CloudProtocol::Gemini | CloudProtocol::OpenAiChatAudio => vec![temperature_spec(2.0)],
        _ => Vec::new(),
    }
}

/// The shared temperature knob: 0 keeps the transcript faithful; higher lets the model paraphrase.
/// `max` matches the vendor's documented ceiling (OpenAI 1, Gemini/Qwen 2).
fn temperature_spec(max: f64) -> ParamSpec {
    ParamSpec::float(
        "temperature",
        "Temperature",
        "0 keeps the transcript closest to the audio. Higher lets the model smooth or guess unclear \
         speech — usually leave at 0 for transcription.",
        0.0,
        max,
        0.1,
        0.0,
    )
}

/// The advanced parameters the chat **assist** (summary / notes / live hints) exposes, for the generic
/// settings panel — the assist-side counterpart of [`batch_param_specs`]. These are the OpenAI-compatible
/// `/chat/completions` tuning knobs every assist provider speaks. Each stays at a neutral default the
/// settings panel treats as "unset", so an untouched knob is omitted from the request and the model
/// applies its own optimum — a model that only accepts its default (e.g. a reasoning model) is never
/// sent one.
pub fn assist_param_specs() -> Vec<ParamSpec> {
    vec![
        ParamSpec::float(
            "temperature",
            "Temperature",
            "Higher is more varied and creative, lower is more focused. Leave at 1 to use the model's \
             own default — some models (e.g. reasoning models) only accept that.",
            0.0,
            2.0,
            0.05,
            1.0,
        ),
        ParamSpec::float(
            "top_p",
            "Top-p",
            "Nucleus-sampling cutoff. Leave at 1 to use the model's default; lower it to keep only the \
             most likely words. Usually tune either this or temperature, not both.",
            0.0,
            1.0,
            0.05,
            1.0,
        ),
        ParamSpec::float(
            "frequency_penalty",
            "Frequency penalty",
            "Discourages repeating the same words: positive values make the reply less repetitive, \
             negative values let it repeat more. Leave at 0 for the model's default.",
            -2.0,
            2.0,
            0.1,
            0.0,
        ),
        ParamSpec::float(
            "presence_penalty",
            "Presence penalty",
            "Encourages new topics: positive values push the reply toward fresh subjects, negative \
             values keep it on the same ones. Leave at 0 for the model's default.",
            -2.0,
            2.0,
            0.1,
            0.0,
        ),
        ParamSpec::int(
            "max_tokens",
            "Max reply tokens",
            "Hard cap on the reply length, in tokens. 0 leaves it to the model (no explicit cap).",
            0,
            8192,
            0,
        ),
    ]
}

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
    /// Wire model id, e.g. `gpt-4o-transcribe` / `gemini-3.5-flash` / `qwen3-asr-flash`.
    model: String,
    /// Language code, or empty to let the provider auto-detect.
    language: String,
    /// Advanced per-vendor knobs (temperature, biasing prompt, …) resolved from the settings panel.
    params: ParamValues,
    /// Optional UI error sink for the live segment-batch path: invoked when a `transcribe` call
    /// fails so a bad key/model/URL surfaces instead of silently producing no transcript. The File
    /// path leaves this unset — its errors surface through the command's `Result`.
    on_error: Option<ErrorSink>,
    /// Last surfaced error, so a persistent failure emits once rather than once per utterance.
    last_error: Option<String>,
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
        params: ParamValues,
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
            params,
            on_error: None,
            last_error: None,
        })
    }

    /// Attaches a UI error sink for the live segment-batch path, so a failed per-utterance call
    /// surfaces instead of silently producing no transcript. Returns `self` for chaining.
    pub fn with_error_sink(mut self, sink: ErrorSink) -> Self {
        self.on_error = Some(sink);
        self
    }

    /// Emits `msg` to the error sink unless it repeats the last one (a persistent failure shouldn't
    /// spam per utterance).
    fn report_error(&mut self, msg: &str) {
        if self.last_error.as_deref() == Some(msg) {
            return;
        }

        self.last_error = Some(msg.to_owned());

        if let Some(sink) = &self.on_error {
            sink(msg);
        }
    }

    /// Uploads `audio` and returns the transcript — one segment spanning the clip, or one per speaker
    /// turn for the diarized model.
    fn transcribe_upload(&self, audio: &[f32], sample_rate: u32) -> Result<TranscriptionResult> {
        let wav = encode_wav_16bit(audio, sample_rate);

        // The diarized OpenAI model returns speaker-labelled segments — parse them directly rather
        // than as one block of text. (This non-streaming path backs the plain `transcribe`; the File
        // path goes through `transcribe_clip`, which streams for a live progress bar.)
        if self.is_diarize() {
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
        let (content_type, body) =
            build_multipart(wav, &self.model, &self.language, &self.params, false);
        let sent = send_with_retry(|| {
            ureq::post(&endpoint)
                .set(&self.auth_header, &self.auth_value)
                .set("Content-Type", &content_type)
                .timeout(Duration::from_secs(300))
                .send_bytes(&body)
        });
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
        let gemini_body = build_gemini_body(wav, &self.language, &self.params);
        let sent = send_with_retry(|| {
            ureq::post(&url)
                .set("Content-Type", "application/json")
                .timeout(Duration::from_secs(300))
                .send_string(&gemini_body)
        });
        parse_gemini(&body_or_error(sent)?)
    }

    /// OpenAI-compatible `/chat/completions` with the audio as an `input_audio` content part.
    fn post_chat_audio(&self, wav: &[u8]) -> Result<String> {
        let endpoint = format!("{}/chat/completions", self.base_url);
        let chat_body = build_chat_audio_body(wav, &self.model, &self.language, &self.params);
        let sent = send_with_retry(|| {
            ureq::post(&endpoint)
                .set(&self.auth_header, &self.auth_value)
                .set("Content-Type", "application/json")
                .timeout(Duration::from_secs(300))
                .send_string(&chat_body)
        });
        parse_chat_completion(&body_or_error(sent)?)
    }

    /// Whether this engine is the OpenAI diarized model, which has its own request shape (speaker
    /// labels, mandatory chunking) and can stream its segments as server-sent events.
    fn is_diarize(&self) -> bool {
        self.protocol == CloudProtocol::OpenAi && self.model == OPENAI_DIARIZE_MODEL
    }

    /// Streams the diarize model's speaker segments over server-sent events, driving `progress` from
    /// each segment's end timestamp (as a fraction of `total_secs`) so the File view shows a moving
    /// progress bar instead of one opaque wait.
    fn post_diarize_stream(
        &self,
        wav: &[u8],
        total_secs: f64,
        progress: Option<&dyn Fn(u8)>,
    ) -> Result<TranscriptionResult> {
        let endpoint = format!("{}/audio/transcriptions", self.base_url);
        let (content_type, body) =
            build_multipart(wav, &self.model, &self.language, &self.params, true);
        let sent = send_with_retry(|| {
            ureq::post(&endpoint)
                .set(&self.auth_header, &self.auth_value)
                .set("Content-Type", &content_type)
                .timeout(Duration::from_secs(300))
                .send_bytes(&body)
        });

        let reader = match sent {
            Ok(response) => BufReader::new(response.into_reader()),
            Err(ureq::Error::Status(code, response)) => {
                let detail: String = response
                    .into_string()
                    .unwrap_or_default()
                    .chars()
                    .take(300)
                    .collect();
                return Err(WispError::Engine(format!(
                    "cloud transcribe failed (HTTP {code}): {detail}"
                )));
            }
            Err(e) => return Err(WispError::Engine(format!("cloud transcribe request: {e}"))),
        };

        let total = total_secs.max(0.001);
        let segments = read_diarized_sse(reader, |end| {
            if let Some(report) = progress {
                report(((end / total) * 100.0).clamp(0.0, 99.0) as u8);
            }
        })?;

        if let Some(report) = progress {
            report(100);
        }
        Ok(TranscriptionResult { segments })
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
        match self.transcribe_upload(audio, sample_rate) {
            Ok(result) => {
                self.last_error = None;
                Ok(result)
            }
            Err(e) => {
                self.report_error(&e.to_string());
                Err(e)
            }
        }
    }

    fn transcribe_clip(
        &mut self,
        audio: &[f32],
        sample_rate: u32,
        options: ClipOptions<'_>,
    ) -> Result<TranscriptionResult> {
        // The diarize model streams its speaker segments, so the File view gets a moving progress bar.
        // If streaming yields nothing (e.g. the provider emitted no incremental events), fall back to a
        // single batch upload so audio with speech never returns silently empty.
        if self.is_diarize() {
            let wav = encode_wav_16bit(audio, sample_rate);
            let total_secs = audio.len() as f64 / sample_rate.max(1) as f64;

            let streamed = self.post_diarize_stream(&wav, total_secs, options.progress)?;
            if !streamed.segments.is_empty() {
                return Ok(streamed);
            }

            return parse_diarized(&self.post_multipart_raw(&wav)?);
        }

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
/// returning the `Content-Type` header value and the body bytes. `stream` asks the server to emit
/// results as server-sent events as they are produced.
fn build_multipart(
    wav: &[u8],
    model: &str,
    language: &str,
    params: &ParamValues,
    stream: bool,
) -> (String, Vec<u8>) {
    // A fixed boundary is safe here: it is far longer and more specific than anything that occurs in
    // 16-bit PCM audio or the short text fields.
    const BOUNDARY: &str = "----wispMultipartBoundaryqZ3xK9pLrT8vB2";

    let mut body = Vec::new();
    append_field(&mut body, BOUNDARY, "model", model);
    if !language.is_empty() {
        append_field(&mut body, BOUNDARY, "language", language);
    }

    // A biasing prompt (names/jargon) steers the spelling; OpenAI accepts it on every transcription
    // model. Sent only when set, so the default body is unchanged.
    let prompt = params.text("prompt", "");
    if !prompt.trim().is_empty() {
        append_field(&mut body, BOUNDARY, "prompt", prompt.trim());
    }

    // Temperature is 0 by default (most faithful); send it only when the user raised it.
    let temperature = params.float("temperature", 0.0);
    if temperature > 0.0 {
        append_field(&mut body, BOUNDARY, "temperature", &temperature.to_string());
    }

    // The diarized model needs `diarized_json` to return per-speaker segments; everyone else gets the
    // plain `{ "text": … }`.
    let response_format = if model == OPENAI_DIARIZE_MODEL {
        "diarized_json"
    } else {
        "json"
    };
    append_field(&mut body, BOUNDARY, "response_format", response_format);

    // The diarize model requires an explicit chunking strategy for inputs over 30 s; `auto` lets the
    // server normalize loudness and pick VAD boundaries. The plain transcribers reject this field, so
    // send it only for diarize.
    if model == OPENAI_DIARIZE_MODEL {
        append_field(&mut body, BOUNDARY, "chunking_strategy", "auto");
    }

    // Streaming turns the one opaque upload into a feed of segment events, so the caller can report
    // real progress. Ignored by `whisper-1`; harmless on the models we never stream.
    if stream {
        append_field(&mut body, BOUNDARY, "stream", "true");
    }

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

    let mut labels: Vec<String> = Vec::new();
    let mut segments = Vec::new();
    for seg in &response.segments {
        if let Some(segment) = diarized_segment(
            segments.len() as u64,
            seg.start..seg.end,
            &seg.text,
            &seg.speaker,
            &mut labels,
        ) {
            segments.push(segment);
        }
    }

    Ok(TranscriptionResult { segments })
}

/// Maps one diarized speaker turn into a [`TranscriptSegment`] with a stable 0-based speaker index
/// (assigned by first appearance into `labels`), or `None` for blank text. Shared by the batch JSON
/// parse and the streaming SSE reader so both index speakers identically.
fn diarized_segment(
    id: u64,
    time: std::ops::Range<f64>,
    text: &str,
    speaker: &str,
    labels: &mut Vec<String>,
) -> Option<TranscriptSegment> {
    let text = text.trim();
    if text.is_empty() {
        return None;
    }

    let speaker = labels.iter().position(|l| l == speaker).unwrap_or_else(|| {
        labels.push(speaker.to_owned());
        labels.len() - 1
    });

    let mut segment = TranscriptSegment::new(
        id,
        text,
        Duration::from_secs_f64(time.start)..Duration::from_secs_f64(time.end),
        AudioSourceKind::File,
    );
    segment.speaker = Some(SpeakerId(speaker as u32));
    Some(segment)
}

/// Reads the diarize model's server-sent event stream, appending one segment per
/// `transcript.text.segment` event and calling `on_segment(end_secs)` as each arrives (so the caller
/// can map it to progress). Non-segment events and the `[DONE]` sentinel are ignored.
fn read_diarized_sse(
    reader: impl BufRead,
    mut on_segment: impl FnMut(f64),
) -> Result<Vec<TranscriptSegment>> {
    let mut labels: Vec<String> = Vec::new();
    let mut segments = Vec::new();

    for line in reader.lines() {
        let line =
            line.map_err(|e| WispError::Engine(format!("cloud diarize stream read: {e}")))?;

        let Some(data) = line.strip_prefix("data:").map(str::trim) else {
            continue;
        };
        if data.is_empty() || data == "[DONE]" {
            continue;
        }

        let Ok(event) = serde_json::from_str::<serde_json::Value>(data) else {
            continue;
        };
        if event.get("type").and_then(serde_json::Value::as_str) != Some("transcript.text.segment")
        {
            continue;
        }

        let start = event
            .get("start")
            .and_then(serde_json::Value::as_f64)
            .unwrap_or(0.0);
        let end = event
            .get("end")
            .and_then(serde_json::Value::as_f64)
            .unwrap_or(start);
        let text = event
            .get("text")
            .and_then(serde_json::Value::as_str)
            .unwrap_or_default();
        let speaker = event
            .get("speaker")
            .and_then(serde_json::Value::as_str)
            .unwrap_or_default();

        if let Some(segment) = diarized_segment(
            segments.len() as u64,
            start..end,
            text,
            speaker,
            &mut labels,
        ) {
            segments.push(segment);
            on_segment(end);
        }
    }

    Ok(segments)
}

/// Standard base64 of `bytes` — the encoding both JSON-body protocols use for the audio.
pub(crate) fn b64(bytes: &[u8]) -> String {
    use base64::Engine;
    base64::engine::general_purpose::STANDARD.encode(bytes)
}

/// The instruction sent to chat-style models (Gemini, Qwen) to get a clean verbatim transcript,
/// optionally hinting the spoken `language` and a biasing `prompt` (names/jargon to spell correctly).
fn transcribe_prompt(language: &str, prompt: &str) -> String {
    let mut out = "Transcribe the audio exactly as spoken, verbatim. Output only the transcript \
                   text — no commentary, labels, or quotation marks."
        .to_owned();

    if !language.is_empty() {
        out.push_str(&format!(" The audio language is \"{language}\"."));
    }

    let prompt = prompt.trim();
    if !prompt.is_empty() {
        out.push_str(&format!(" Context for spelling names and terms: {prompt}"));
    }

    out
}

/// Gemini `generateContent` request body: one user turn with the prompt and inline base64 audio.
fn build_gemini_body(wav: &[u8], language: &str, params: &ParamValues) -> String {
    let mut body = serde_json::json!({
        "contents": [{
            "role": "user",
            "parts": [
                { "text": transcribe_prompt(language, &params.text("prompt", "")) },
                { "inlineData": { "mimeType": "audio/wav", "data": b64(wav) } }
            ]
        }]
    });

    // Send temperature only when the user set it — otherwise omit `generationConfig` so Gemini uses
    // the model's own default (some models reject a non-default temperature).
    if params.contains("temperature") {
        body["generationConfig"] =
            serde_json::json!({ "temperature": params.float("temperature", 0.0) });
    }

    body.to_string()
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
fn build_chat_audio_body(wav: &[u8], model: &str, language: &str, params: &ParamValues) -> String {
    let mut body = serde_json::json!({
        "model": model,
        "messages": [{
            "role": "user",
            "content": [
                { "type": "input_audio", "input_audio": { "data": b64(wav), "format": "wav" } },
                { "type": "text", "text": transcribe_prompt(language, &params.text("prompt", "")) }
            ]
        }]
    });

    // Send temperature only when the user set it — some OpenAI-compatible models only accept their
    // own default and reject any explicit value.
    if params.contains("temperature") {
        body["temperature"] = serde_json::json!(params.float("temperature", 0.0));
    }

    body.to_string()
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

/// A text chat-completion against `provider`'s OpenAI-compatible `/chat/completions` endpoint — the
/// LLM primitive behind the summary / action-item features. `system` steers the task (a built-in or
/// custom prompt); `user` carries the transcript. Returns the assistant's reply text.
///
/// Works for any OpenAI-compatible chat endpoint (OpenAI, a custom endpoint, a local Ollama). Not for
/// the Gemini `generateContent` shape — pick an OpenAI-compatible chat model for LLM tasks.
/// A chat-completion request to an OpenAI-compatible endpoint: the two messages plus optional tuning
/// knobs. Every knob is omitted from the wire request when `None`, so a provider that doesn't accept
/// one — e.g. a reasoning model that only allows the default temperature — isn't sent it.
pub struct ChatRequest<'a> {
    pub system: &'a str,
    pub user: &'a str,
    pub temperature: Option<f64>,
    pub max_tokens: Option<u32>,
    pub top_p: Option<f64>,
    pub frequency_penalty: Option<f64>,
    pub presence_penalty: Option<f64>,
}

/// The JSON body for a chat-completion POST: the two messages plus any *set* tuning knob. Each of
/// temperature / max_tokens / top_p is included only when `req` carries one, so a model that rejects
/// a non-default value (e.g. a reasoning model fixed at temperature 1) isn't sent it. `stream` adds
/// `"stream": true` for the SSE variant.
fn build_chat_body(model: &str, req: &ChatRequest, stream: bool) -> String {
    let mut body = serde_json::json!({
        "model": model,
        "messages": [
            { "role": "system", "content": req.system },
            { "role": "user", "content": req.user },
        ],
    });
    if stream {
        body["stream"] = serde_json::json!(true);
    }
    if let Some(temperature) = req.temperature {
        body["temperature"] = serde_json::json!(temperature);
    }
    if let Some(max_tokens) = req.max_tokens {
        body["max_tokens"] = serde_json::json!(max_tokens);
    }
    if let Some(top_p) = req.top_p {
        body["top_p"] = serde_json::json!(top_p);
    }
    if let Some(frequency_penalty) = req.frequency_penalty {
        body["frequency_penalty"] = serde_json::json!(frequency_penalty);
    }
    if let Some(presence_penalty) = req.presence_penalty {
        body["presence_penalty"] = serde_json::json!(presence_penalty);
    }
    body.to_string()
}

pub fn chat_completion(
    provider: &CloudProvider,
    model: &str,
    api_key: &str,
    req: &ChatRequest,
) -> Result<String> {
    let endpoint = format!(
        "{}/chat/completions",
        provider.base_url.trim_end_matches('/')
    );
    let body = build_chat_body(model, req, false);

    let sent = ureq::post(&endpoint)
        .set(&provider.auth.header, &provider.auth.header_value(api_key))
        .set("Content-Type", "application/json")
        .timeout(Duration::from_secs(300))
        .send_string(&body);

    parse_chat_completion(&body_or_error(sent)?)
}

/// Streaming variant of [`chat_completion`]: POSTs with `stream: true`, reads the SSE response, and
/// invokes `on_delta` with each content chunk as it arrives — returning the full accumulated reply. Lets
/// the assist fill the feed token-by-token. Same OpenAI-compatible endpoint + tuning as the plain call.
pub fn chat_completion_stream(
    provider: &CloudProvider,
    model: &str,
    api_key: &str,
    req: &ChatRequest,
    mut on_delta: impl FnMut(&str),
) -> Result<String> {
    let endpoint = format!(
        "{}/chat/completions",
        provider.base_url.trim_end_matches('/')
    );
    let body = build_chat_body(model, req, true);

    let sent = ureq::post(&endpoint)
        .set(&provider.auth.header, &provider.auth.header_value(api_key))
        .set("Content-Type", "application/json")
        .timeout(Duration::from_secs(300))
        .send_string(&body);

    // Reuse body_or_error's status/error formatting for the non-Ok arms; on Ok, stream the body.
    let reader = match sent {
        Ok(response) => BufReader::new(response.into_reader()),
        other => return body_or_error(other).map(|_| String::new()),
    };

    read_chat_completion_sse(reader, &mut on_delta)
}

/// Reads an OpenAI-compatible chat-completion SSE stream, calling `on_delta` with each content chunk and
/// accumulating the full reply (returned). Stops at `[DONE]`; non-JSON / role-only lines are skipped.
fn read_chat_completion_sse(
    reader: impl BufRead,
    on_delta: &mut impl FnMut(&str),
) -> Result<String> {
    #[derive(serde::Deserialize)]
    struct Chunk {
        #[serde(default)]
        choices: Vec<Choice>,
    }
    #[derive(serde::Deserialize)]
    struct Choice {
        #[serde(default)]
        delta: Delta,
    }
    #[derive(serde::Deserialize, Default)]
    struct Delta {
        #[serde(default)]
        content: Option<String>,
    }

    let mut full = String::new();
    for line in reader.lines() {
        let line = line.map_err(|e| WispError::Engine(format!("cloud chat stream read: {e}")))?;

        let Some(data) = line.strip_prefix("data:").map(str::trim) else {
            continue;
        };
        if data.is_empty() || data == "[DONE]" {
            continue;
        }

        let Ok(chunk) = serde_json::from_str::<Chunk>(data) else {
            continue;
        };
        let Some(content) = chunk
            .choices
            .into_iter()
            .next()
            .and_then(|c| c.delta.content)
        else {
            continue;
        };
        if !content.is_empty() {
            full.push_str(&content);
            on_delta(&content);
        }
    }

    Ok(full)
}

/// A uniform "response wasn't the JSON we expected" error carrying a short snippet of what we got.
fn not_json_error(json: &str, e: serde_json::Error) -> WispError {
    let snippet: String = json.chars().take(200).collect();
    WispError::Engine(format!(
        "cloud response wasn't the expected JSON: {e} — got: {snippet}"
    ))
}

/// Total HTTP attempts for a transient cloud failure (>= 1); the rest are retries. Override with the
/// env var so an operator on a flaky link can ask for more — or `1` to disable retries.
const CLOUD_RETRY_ATTEMPTS_ENV: &str = "WISP_CLOUD_RETRY_ATTEMPTS";
const DEFAULT_CLOUD_RETRY_ATTEMPTS: u32 = 3;
/// Upper bound on any single backoff / `Retry-After` wait, so a hostile or huge value can't stall a File.
const MAX_RETRY_DELAY: Duration = Duration::from_secs(30);

fn cloud_retry_attempts() -> u32 {
    std::env::var(CLOUD_RETRY_ATTEMPTS_ENV)
        .ok()
        .and_then(|v| v.parse::<u32>().ok())
        .filter(|&n| n >= 1)
        .unwrap_or(DEFAULT_CLOUD_RETRY_ATTEMPTS)
}

/// A transient failure worth retrying: any transport error (dropped connection, DNS, timeout) or a
/// throttling / temporary-server HTTP status. A 4xx like 401/404 is the caller's fault — never retried.
fn is_transient(err: &ureq::Error) -> bool {
    matches!(
        err,
        ureq::Error::Transport(_) | ureq::Error::Status(429 | 500 | 502 | 503 | 504, _)
    )
}

/// How long to wait before the next attempt: a server `Retry-After` (seconds, capped) when the status
/// carries one, else exponential backoff (0.5s, 1s, 2s, …), capped at [`MAX_RETRY_DELAY`].
fn retry_delay(err: &ureq::Error, attempt: u32) -> Duration {
    if let ureq::Error::Status(_, response) = err {
        if let Some(secs) = response
            .header("Retry-After")
            .and_then(|v| v.trim().parse::<u64>().ok())
        {
            return Duration::from_secs(secs).min(MAX_RETRY_DELAY);
        }
    }
    Duration::from_millis(500u64.saturating_mul(2u64.saturating_pow(attempt - 1)))
        .min(MAX_RETRY_DELAY)
}

/// Retries `send` while it fails transiently and attempts remain, sleeping between tries. Generic over
/// the result/error so the loop is unit-testable without real HTTP.
fn retry_send<T, E>(
    mut send: impl FnMut() -> std::result::Result<T, E>,
    max_attempts: u32,
    transient: impl Fn(&E) -> bool,
    delay: impl Fn(&E, u32) -> Duration,
) -> std::result::Result<T, E> {
    let mut attempt = 1u32;
    loop {
        match send() {
            Ok(value) => return Ok(value),
            Err(err) if attempt < max_attempts && transient(&err) => {
                std::thread::sleep(delay(&err, attempt));
                attempt += 1;
            }
            Err(err) => return Err(err),
        }
    }
}

/// Sends a cloud HTTP request with bounded retry on transient failure, so one network blip or 503 on a
/// single File window doesn't abort the whole transcription. Transcription POSTs are idempotent, so a
/// re-send is safe.
fn send_with_retry(
    send: impl FnMut() -> std::result::Result<ureq::Response, ureq::Error>,
) -> std::result::Result<ureq::Response, ureq::Error> {
    retry_send(send, cloud_retry_attempts(), is_transient, retry_delay)
}

/// Turns a ureq send result into the response body text, surfacing the API's own error body (key
/// rejected, model unknown, quota exceeded) on a non-2xx status rather than a bare status code.
// Shared HTTP-result helper for every cloud call (transcription AND chat/assist) — so its message is
// kept generic ("cloud request") rather than mislabeling an assist failure as a transcribe one.
fn body_or_error(sent: std::result::Result<ureq::Response, ureq::Error>) -> Result<String> {
    match sent {
        Ok(response) => response
            .into_string()
            .map_err(|e| WispError::Engine(format!("cloud response read: {e}"))),
        Err(ureq::Error::Status(code, response)) => {
            let detail: String = response
                .into_string()
                .unwrap_or_default()
                .chars()
                .take(300)
                .collect();
            Err(WispError::Engine(format!(
                "cloud request failed (HTTP {code}): {detail}"
            )))
        }
        Err(e) => Err(WispError::Engine(format!("cloud request error: {e}"))),
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
    use wisp_core::params::{ParamKind, ParamValue};

    fn provider() -> CloudProvider {
        CloudProvider {
            id: "openai".to_owned(),
            display_name: "OpenAI".to_owned(),
            protocol: CloudProtocol::OpenAi,
            base_url: "https://api.example.com/v1/".to_owned(), // trailing slash, to test trimming
            keys_url: "https://example.com/keys".to_owned(),
            auth: CloudAuth::bearer(),
            streaming: Some(wisp_core::cloud::StreamingProtocol::OpenAiRealtime),
            models: vec![
                CloudModel {
                    id: "gpt-4o-transcribe".to_owned(),
                    display_name: "GPT-4o".to_owned(),
                    streaming: true,
                    batch: true,
                    languages: vec![],
                    description: String::new(),
                    recommended: false,
                    diarizes: false,
                },
                CloudModel {
                    id: "stream-only".to_owned(),
                    display_name: "Stream only".to_owned(),
                    streaming: true,
                    batch: false,
                    languages: vec![],
                    description: String::new(),
                    recommended: false,
                    diarizes: false,
                },
            ],
        }
    }

    #[test]
    fn error_sink_emits_once_per_distinct_message_and_recovers() {
        use std::sync::{Arc, Mutex};

        let seen = Arc::new(Mutex::new(Vec::<String>::new()));
        let captured = Arc::clone(&seen);
        let mut engine = CloudEngine::new(
            &provider(),
            "gpt-4o-transcribe",
            "sk-abc",
            "en",
            ParamValues::new(),
        )
        .unwrap()
        .with_error_sink(Box::new(move |msg: &str| {
            captured.lock().unwrap().push(msg.to_owned());
        }));

        // A persistent failure surfaces once, not once per utterance.
        engine.report_error("boom");
        engine.report_error("boom");

        // A different failure surfaces again.
        engine.report_error("kaboom");

        // After a success clears the last error (as `transcribe` does on `Ok`), the same message
        // can surface anew rather than being suppressed forever.
        engine.last_error = None;
        engine.report_error("boom");

        assert_eq!(*seen.lock().unwrap(), vec!["boom", "kaboom", "boom"]);
    }

    #[test]
    fn new_trims_base_url_and_auth_and_rejects_bad_inputs() {
        let engine = CloudEngine::new(
            &provider(),
            "gpt-4o-transcribe",
            "sk-abc",
            "en",
            ParamValues::new(),
        )
        .unwrap();
        assert_eq!(engine.base_url, "https://api.example.com/v1"); // trailing slash trimmed
        assert_eq!(engine.auth_header, "Authorization");
        assert_eq!(engine.auth_value, "Bearer sk-abc");

        // Unknown model, a non-batch model, and a blank key are all rejected.
        assert!(CloudEngine::new(&provider(), "nope", "sk", "", ParamValues::new()).is_err());
        assert!(
            CloudEngine::new(&provider(), "stream-only", "sk", "", ParamValues::new()).is_err()
        );
        assert!(CloudEngine::new(
            &provider(),
            "gpt-4o-transcribe",
            "  ",
            "",
            ParamValues::new()
        )
        .is_err());
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
        let (content_type, body) =
            build_multipart(&wav, "gpt-4o-transcribe", "en", &ParamValues::new(), false);
        let text = String::from_utf8_lossy(&body);

        assert!(content_type.starts_with("multipart/form-data; boundary="));
        assert!(text.contains("name=\"model\"") && text.contains("gpt-4o-transcribe"));
        assert!(text.contains("name=\"language\"") && text.contains("\r\nen\r\n"));
        assert!(
            text.contains("filename=\"audio.wav\"") && text.contains("Content-Type: audio/wav")
        );
        assert!(text.contains("RIFFfakewavdata"));
        assert!(text.trim_end().ends_with("--")); // closing boundary

        // Empty language + default params omit the optional fields entirely.
        let (_, body) = build_multipart(&wav, "m", "", &ParamValues::new(), false);
        let text = String::from_utf8_lossy(&body);
        assert!(!text.contains("name=\"language\""));
        assert!(!text.contains("name=\"prompt\""), "no prompt unless set");
        assert!(
            !text.contains("name=\"temperature\""),
            "no temperature at default 0"
        );

        // A biasing prompt and a raised temperature ride along when set.
        let mut params = ParamValues::new();
        params.set("prompt", ParamValue::Text("Wisp, Tauri".to_owned()));
        params.set("temperature", ParamValue::Float(0.4));
        let (_, body) = build_multipart(&wav, "gpt-4o-transcribe", "", &params, false);
        let text = String::from_utf8_lossy(&body);
        assert!(text.contains("name=\"prompt\"") && text.contains("Wisp, Tauri"));
        assert!(text.contains("name=\"temperature\"") && text.contains("\r\n0.4\r\n"));
    }

    #[test]
    fn diarize_multipart_requests_diarized_json_and_chunking_others_omit_chunking() {
        let wav = b"RIFFfakewavdata".to_vec();

        // The diarize model needs `diarized_json` for speaker labels and `chunking_strategy=auto`,
        // which OpenAI requires for inputs over 30 s. Missing either is a hard 400. Batch mode omits
        // the stream field.
        let (_, body) = build_multipart(&wav, OPENAI_DIARIZE_MODEL, "", &ParamValues::new(), false);
        let text = String::from_utf8_lossy(&body);
        assert!(
            text.contains("name=\"response_format\"") && text.contains("\r\ndiarized_json\r\n")
        );
        assert!(text.contains("name=\"chunking_strategy\"") && text.contains("\r\nauto\r\n"));
        assert!(
            !text.contains("name=\"stream\""),
            "no stream field unless asked"
        );

        // Streaming adds the `stream` field so the server emits the segment feed.
        let (_, body) = build_multipart(&wav, OPENAI_DIARIZE_MODEL, "", &ParamValues::new(), true);
        assert!(String::from_utf8_lossy(&body).contains("name=\"stream\""));

        // The plain transcribers reject `chunking_strategy`, so it must never be sent for them.
        let (_, body) = build_multipart(&wav, "gpt-4o-transcribe", "", &ParamValues::new(), false);
        assert!(!String::from_utf8_lossy(&body).contains("chunking_strategy"));
    }

    #[test]
    fn read_diarized_sse_collects_segments_and_reports_each_end() {
        // Mirrors OpenAI's diarize stream: one `transcript.text.segment` event per line, a blank turn
        // (dropped), then non-segment events and the sentinel that must be ignored.
        let sse = concat!(
            "data: {\"type\":\"transcript.text.segment\",\"start\":0.0,\"end\":4.7,\"text\":\"Thanks for calling.\",\"speaker\":\"agent\"}\n",
            "\n",
            "data: {\"type\":\"transcript.text.segment\",\"start\":4.7,\"end\":11.8,\"text\":\"Hi there.\",\"speaker\":\"A\"}\n",
            "data: {\"type\":\"transcript.text.segment\",\"start\":12.0,\"end\":12.1,\"text\":\"   \",\"speaker\":\"A\"}\n",
            "data: {\"type\":\"transcript.text.done\",\"text\":\"summary ignored\"}\n",
            "data: [DONE]\n",
        );

        let mut ends = Vec::new();
        let segments =
            read_diarized_sse(Cursor::new(sse.as_bytes().to_vec()), |end| ends.push(end)).unwrap();

        assert_eq!(
            segments.len(),
            2,
            "two real turns; blank dropped, done/[DONE] ignored"
        );
        assert_eq!(segments[0].text, "Thanks for calling.");
        assert_eq!(segments[0].speaker, Some(SpeakerId(0)));
        assert_eq!(
            segments[1].speaker,
            Some(SpeakerId(1)),
            "second speaker → index 1"
        );
        assert_eq!(
            ends,
            vec![4.7, 11.8],
            "progress fires once per real segment, with its end time"
        );
    }

    #[test]
    fn chat_completion_sse_streams_content_deltas_and_accumulates_full_reply() {
        // Mirrors an OpenAI-compatible chat stream: a role-only opener (no content), content chunks,
        // then the sentinel. on_delta sees each content chunk; the return is the concatenation.
        let sse = concat!(
            "data: {\"choices\":[{\"delta\":{\"role\":\"assistant\"}}]}\n",
            "data: {\"choices\":[{\"delta\":{\"content\":\"Hel\"}}]}\n",
            "data: {\"choices\":[{\"delta\":{\"content\":\"lo.\"}}]}\n",
            "data: [DONE]\n",
        );

        let mut chunks = Vec::new();
        let full = read_chat_completion_sse(Cursor::new(sse.as_bytes().to_vec()), &mut |c| {
            chunks.push(c.to_owned())
        })
        .unwrap();

        assert_eq!(
            chunks,
            vec!["Hel", "lo."],
            "each content chunk, role-only skipped"
        );
        assert_eq!(full, "Hello.");
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
        let body = build_gemini_body(b"RIFFfakewav", "yue", &ParamValues::new());
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
        let body =
            build_chat_audio_body(b"RIFFfakewav", "qwen3-asr-flash", "", &ParamValues::new());
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
    fn batch_param_specs_exposes_temperature_per_cloud_protocol() {
        // Every wired batch protocol surfaces temperature; its ceiling matches the vendor's API —
        // OpenAI transcription caps at 1, Gemini and the OpenAI-compatible chat (Qwen) at 2.
        let cases = [
            (CloudProtocol::OpenAi, 1.0),
            (CloudProtocol::Gemini, 2.0),
            (CloudProtocol::OpenAiChatAudio, 2.0),
        ];
        for (protocol, expected_max) in cases {
            let spec = batch_param_specs(protocol, "any-model")
                .into_iter()
                .find(|s| s.key == "temperature")
                .unwrap_or_else(|| panic!("{protocol:?} exposes temperature"));
            match spec.kind {
                ParamKind::Float { max, .. } => {
                    assert_eq!(max, expected_max, "{protocol:?} temperature ceiling")
                }
                _ => panic!("temperature is a float slider"),
            }
        }
    }

    #[test]
    fn assist_param_specs_expose_the_chat_tuning_knobs() {
        // The assist panel surfaces temperature / top_p / max_tokens, each at a neutral default the UI
        // treats as "unset" so an untouched knob is omitted (and the model uses its own optimum).
        let specs = assist_param_specs();
        let keys: Vec<&str> = specs.iter().map(|s| s.key.as_str()).collect();
        assert_eq!(
            keys,
            [
                "temperature",
                "top_p",
                "frequency_penalty",
                "presence_penalty",
                "max_tokens"
            ]
        );

        let temperature = specs.iter().find(|s| s.key == "temperature").unwrap();
        assert!(
            matches!(temperature.kind, ParamKind::Float { min, max, .. } if min == 0.0 && max == 2.0)
        );
        assert_eq!(temperature.default, ParamValue::Float(1.0));

        let top_p = specs.iter().find(|s| s.key == "top_p").unwrap();
        assert!(
            matches!(top_p.kind, ParamKind::Float { min, max, .. } if min == 0.0 && max == 1.0)
        );
        assert_eq!(top_p.default, ParamValue::Float(1.0));

        let max_tokens = specs.iter().find(|s| s.key == "max_tokens").unwrap();
        assert!(matches!(max_tokens.kind, ParamKind::Int { .. }));
        assert_eq!(max_tokens.default, ParamValue::Int(0));
    }

    #[test]
    fn temperature_and_prompt_wire_into_gemini_and_chat_bodies() {
        let mut params = ParamValues::new();
        params.set("temperature", ParamValue::Float(0.3));
        params.set("prompt", ParamValue::Text("Wisp".to_owned()));

        let gemini: serde_json::Value =
            serde_json::from_str(&build_gemini_body(b"a", "yue", &params)).unwrap();
        assert_eq!(gemini["generationConfig"]["temperature"], 0.3);
        assert!(gemini["contents"][0]["parts"][0]["text"]
            .as_str()
            .unwrap()
            .contains("Wisp"));

        let chat: serde_json::Value = serde_json::from_str(&build_chat_audio_body(
            b"a",
            "qwen3-asr-flash",
            "en",
            &params,
        ))
        .unwrap();
        assert_eq!(chat["temperature"], 0.3);
        assert!(chat["messages"][0]["content"][1]["text"]
            .as_str()
            .unwrap()
            .contains("Wisp"));
    }

    #[test]
    fn temperature_is_omitted_when_unset() {
        // An unset temperature must not be sent — a model that only accepts its own default (e.g.
        // "temperature only supports 1") would reject a fabricated value. Omitting lets it apply the
        // model's own default.
        let params = ParamValues::new();

        let gemini: serde_json::Value =
            serde_json::from_str(&build_gemini_body(b"a", "", &params)).unwrap();
        assert!(
            gemini.get("generationConfig").is_none(),
            "no generationConfig when temperature is unset"
        );

        let chat: serde_json::Value =
            serde_json::from_str(&build_chat_audio_body(b"a", "qwen3-asr-flash", "", &params))
                .unwrap();
        assert!(
            chat.get("temperature").is_none(),
            "no temperature field when unset"
        );
    }

    #[test]
    fn chat_body_sends_tuning_only_when_set() {
        // A reasoning model (e.g. GPT-5) rejects a non-default temperature, so an unset knob must be
        // omitted — the body carries only the knobs the caller actually set.
        let bare = ChatRequest {
            system: "sys",
            user: "hi",
            temperature: None,
            max_tokens: None,
            top_p: None,
            frequency_penalty: None,
            presence_penalty: None,
        };
        let body: serde_json::Value =
            serde_json::from_str(&build_chat_body("gpt-5", &bare, false)).unwrap();
        assert_eq!(body["model"], "gpt-5");
        assert_eq!(body["messages"][0]["content"], "sys");
        assert_eq!(body["messages"][1]["content"], "hi");
        assert!(
            body.get("temperature").is_none(),
            "unset temperature omitted"
        );
        assert!(body.get("max_tokens").is_none(), "unset max_tokens omitted");
        assert!(body.get("top_p").is_none(), "unset top_p omitted");
        assert!(
            body.get("frequency_penalty").is_none(),
            "unset frequency_penalty omitted"
        );
        assert!(
            body.get("presence_penalty").is_none(),
            "unset presence_penalty omitted"
        );
        assert!(
            body.get("stream").is_none(),
            "non-stream body has no stream flag"
        );

        let tuned = ChatRequest {
            system: "sys",
            user: "hi",
            temperature: Some(0.7),
            max_tokens: Some(256),
            top_p: Some(0.9),
            frequency_penalty: Some(0.5),
            presence_penalty: Some(-0.5),
        };
        let body: serde_json::Value =
            serde_json::from_str(&build_chat_body("gpt-4o", &tuned, true)).unwrap();
        assert_eq!(body["temperature"], 0.7);
        assert_eq!(body["max_tokens"], 256);
        assert_eq!(body["top_p"], 0.9);
        assert_eq!(body["frequency_penalty"], 0.5);
        assert_eq!(body["presence_penalty"], -0.5);
        assert_eq!(body["stream"], true, "stream variant sets the stream flag");
    }

    #[test]
    fn to_result_is_one_segment_or_empty() {
        let audio = vec![0.0f32; 32_000]; // 2 s at 16 kHz
        let result = to_result("  hi there  ", &audio, 16_000);
        assert_eq!(result.segments.len(), 1);
        assert_eq!(result.segments[0].text, "hi there");

        assert!(to_result("   ", &audio, 16_000).segments.is_empty());
    }

    #[test]
    fn transient_classification_retries_throttle_and_5xx_only() {
        let status = |code: u16| {
            ureq::Error::Status(code, ureq::Response::new(code, "x", "").expect("resp"))
        };
        for code in [429u16, 500, 502, 503, 504] {
            assert!(is_transient(&status(code)), "HTTP {code} should be retried");
        }
        for code in [400u16, 401, 403, 404, 422] {
            assert!(
                !is_transient(&status(code)),
                "HTTP {code} should not be retried"
            );
        }
    }

    #[test]
    fn retry_delay_honors_a_capped_retry_after_else_backs_off_exponentially() {
        let from_raw = |raw: &str| {
            ureq::Error::Status(429, raw.parse::<ureq::Response>().expect("parse resp"))
        };
        // A server Retry-After (seconds) wins, capped at MAX_RETRY_DELAY.
        assert_eq!(
            retry_delay(
                &from_raw("HTTP/1.1 429 Too Many Requests\r\nRetry-After: 5\r\n\r\n"),
                1
            ),
            Duration::from_secs(5)
        );
        assert_eq!(
            retry_delay(
                &from_raw("HTTP/1.1 429 Too Many Requests\r\nRetry-After: 9999\r\n\r\n"),
                1
            ),
            MAX_RETRY_DELAY
        );
        // No Retry-After → exponential 0.5s, 1s, 2s …
        let no_header = ureq::Error::Status(503, ureq::Response::new(503, "x", "").expect("resp"));
        assert_eq!(retry_delay(&no_header, 1), Duration::from_millis(500));
        assert_eq!(retry_delay(&no_header, 2), Duration::from_secs(1));
        assert_eq!(retry_delay(&no_header, 3), Duration::from_secs(2));
    }

    #[test]
    fn cloud_retry_attempts_env_var_name_is_pinned() {
        // Operators tune cloud resilience via this env var; a rename silently changes their behaviour.
        assert_eq!(CLOUD_RETRY_ATTEMPTS_ENV, "WISP_CLOUD_RETRY_ATTEMPTS");
    }

    #[test]
    fn retry_send_returns_on_first_success_without_retrying() {
        let mut calls = 0;
        let out: std::result::Result<i32, i32> = retry_send(
            || {
                calls += 1;
                Ok(7)
            },
            3,
            |_| true,
            |_, _| Duration::ZERO,
        );
        assert_eq!(out, Ok(7));
        assert_eq!(calls, 1);
    }

    #[test]
    fn retry_send_retries_a_transient_error_then_succeeds() {
        let mut calls = 0;
        let out: std::result::Result<i32, i32> = retry_send(
            || {
                calls += 1;
                if calls < 3 {
                    Err(0)
                } else {
                    Ok(7)
                }
            },
            3,
            |&e| e == 0,
            |_, _| Duration::ZERO,
        );
        assert_eq!(out, Ok(7));
        assert_eq!(calls, 3, "two retries, succeeded on the third attempt");
    }

    #[test]
    fn retry_send_gives_up_after_max_attempts_on_a_persistent_transient() {
        let mut calls = 0;
        let out: std::result::Result<i32, i32> = retry_send(
            || {
                calls += 1;
                Err(0)
            },
            3,
            |&e| e == 0,
            |_, _| Duration::ZERO,
        );
        assert_eq!(out, Err(0));
        assert_eq!(calls, 3, "exactly max_attempts tries");
    }

    #[test]
    fn retry_send_does_not_retry_a_non_transient_error() {
        let mut calls = 0;
        let out: std::result::Result<i32, i32> = retry_send(
            || {
                calls += 1;
                Err(42)
            },
            3,
            |&e| e == 0,
            |_, _| Duration::ZERO,
        );
        assert_eq!(out, Err(42));
        assert_eq!(calls, 1, "a non-transient error is returned immediately");
    }
}
