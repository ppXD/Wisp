//! Realtime cloud streaming transcription — OpenAI Realtime over a WebSocket — behind the same
//! [`StreamingAsrEngine`] trait as the on-device streaming transducer, so the Live pipeline drives
//! it unchanged.
//!
//! The trait is synchronous and pull-based (`accept_waveform` returns the hypothesis right away),
//! but a WebSocket is asynchronous and network-lagged: the transcript for audio sent now arrives
//! later. We bridge that with a single background thread that owns the socket and two channels —
//! `accept_waveform` pushes audio out and drains whatever transcript events have arrived so far,
//! returning the latest hypothesis. The result is eventually-consistent (text trails the audio by a
//! round-trip), which is exactly what the Live loop already tolerates: render the growing text,
//! commit on the endpoint the server's VAD reports.
//!
//! Protocol (OpenAI GA Realtime, verified against `openapi.yaml`): connect with just the bearer
//! token (the old `OpenAI-Beta: realtime=v1` header selects the retired beta shape, now rejected),
//! `session.update`-configure, then stream `input_audio_buffer.append` frames. The model picks one of
//! three [`SessionKind`]s:
//! - **Transcription** (`gpt-4o-transcribe…`): `?intent=transcription`, dedicated ASR, server VAD,
//!   `conversation.item.input_audio_transcription.delta`/`.completed`.
//! - **Model** (`gpt-realtime…`): `?model=…`, text output steered by free-form `instructions`
//!   (transcribe or translate), `response.output_text.delta`/`.done`.
//! - **Translation** (`gpt-realtime-translate`): the `/translations` sub-path, a target output language,
//!   append-only `session.output_transcript.delta` (finalised on a gap — it has no per-turn done).
//!
//! Input is 16-bit PCM, **24 kHz**, mono, little-endian. The engine reports 24 kHz via
//! [`StreamingAsrEngine::input_sample_rate`], so the pipeline resamples the capture device to native
//! 24 kHz for us — keeping the 8–12 kHz band a 16 kHz path would drop, rather than upsampling 16 kHz.

use std::io::ErrorKind;
use std::net::TcpStream;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use tungstenite::client::IntoClientRequest;
use tungstenite::http::{header::AUTHORIZATION, HeaderName, HeaderValue};
use tungstenite::stream::MaybeTlsStream;
use tungstenite::{Message, WebSocket};

use wisp_core::cloud::StreamingProtocol;
use wisp_core::engine::{StreamingAsrEngine, StreamingResult};
use wisp_core::error::{Result, WispError};
use wisp_core::params::{ParamSpec, ParamValue, ParamValues};

use crate::b64;

/// OpenAI Realtime requires 24 kHz PCM16 input.
const REALTIME_SAMPLE_RATE: u32 = 24_000;

/// The dedicated ASR model a model session runs for its parallel input-audio transcription — the
/// gap-free verbatim stream that backs the transcript while the model's own output is the secondary
/// (translation) stream. (`gpt-4o-transcribe` is the most accurate dedicated transcriber.)
const MODEL_SESSION_TRANSCRIBE_MODEL: &str = "gpt-4o-transcribe";

/// The default `instructions` for a model session — a strict verbatim transcriber. The user edits it
/// (e.g. to translate). Heavy guardrails keep the LLM from chatting back instead of transcribing.
const DEFAULT_TRANSCRIBE_INSTRUCTIONS: &str = "You are a real-time transcription engine. Transcribe \
the user's speech into text, verbatim.\n\nRules:\n- Output ONLY the transcript — no commentary, \
labels, prefixes, or quotation marks.\n- NEVER respond conversationally and NEVER answer questions; \
transcribe the question itself.\n- Keep the speaker's exact words and original language; do not \
paraphrase, summarise, or translate.\n- If audio is unclear, transcribe what you can make out; do \
not invent content.\n\nTo translate instead, replace these instructions, e.g. \"Translate the \
speech to English. Output only the translation, nothing else.\"";

/// Base Realtime WebSocket endpoint (overridable via [`REALTIME_URL_ENV`]); the
/// per-session query (`?intent=transcription` for ASR, `?model=…` for a model session) is appended.
const DEFAULT_REALTIME_URL: &str = "wss://api.openai.com/v1/realtime";

/// How long without a translation delta before the in-progress translated line is committed. The
/// translation session streams append-only text with no per-utterance boundary, so we finalise on
/// this gap (the input is server-VAD-segmented, so deltas arrive in bursts with pauses between).
const TRANSLATION_FINALIZE_GAP: Duration = Duration::from_millis(1200);

/// For a model session's two streams (verbatim transcript + the model's text output): once one
/// stream has completed, how long to wait for the other before finalising with what we have. The
/// normal path finalises the instant both complete; this is only the fallback for a stalled stream.
const DUAL_FINALIZE_GAP: Duration = Duration::from_millis(2000);

/// Which Realtime session shape a model speaks — each has a different config and event set.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SessionKind {
    /// Dedicated ASR (`gpt-4o-transcribe…`): `?intent=transcription`, server VAD, transcript events.
    Transcription,
    /// A general realtime model with text output (`gpt-realtime…`): `?model=…`, instructions-driven,
    /// `response.output_text.*` events.
    Model,
    /// An AI assist session (any realtime model): `?model=…`, instructions-driven text output, but —
    /// unlike [`Model`](Self::Model) — *no* parallel verbatim transcription (the live pipeline already
    /// transcribes the same audio). The model's `response.output_text.*` is the sole stream, finalised
    /// on its own `.done`. Listens to the meeting and answers per turn under the user's assist prompt.
    Assist,
    /// Dedicated speech translation (`gpt-realtime-translate`): the `/translations` sub-path, a target
    /// output language, append-only `session.output_transcript.delta` (translated text, primary) plus
    /// `session.input_transcript.delta` (the source-language original, secondary) — no per-utterance
    /// done events, so finalised on a gap.
    Translation,
}

/// The session shape `model` uses.
fn session_kind(model: &str) -> SessionKind {
    if model == "gpt-realtime-translate" {
        SessionKind::Translation
    } else if model.starts_with("gpt-realtime") {
        SessionKind::Model
    } else {
        SessionKind::Transcription
    }
}

/// The WebSocket URL for `model`'s session, off the env-overridable base.
fn realtime_url(model: &str, kind: SessionKind) -> String {
    let base = std::env::var(REALTIME_URL_ENV).unwrap_or_else(|_| DEFAULT_REALTIME_URL.to_owned());
    match kind {
        SessionKind::Transcription => format!("{base}?intent=transcription"),
        // Translation has its own sub-path AND still needs the model in the query: the base rejects
        // `?intent=translation` ("only supported on /v1/realtime/translations"), and the sub-path
        // without a model rejects with `missing_model`. So: `/translations?model=…`.
        SessionKind::Translation => format!("{base}/translations?model={model}"),
        // A model session and an assist session both connect with `?model=…`; they differ only in
        // their `session.update` config (assist drops the parallel transcription) and event mapping.
        SessionKind::Model | SessionKind::Assist => format!("{base}?model={model}"),
    }
}

/// The `input_audio_buffer.append` client-event `type` for a session. The translation endpoint
/// namespaces its client events under `session.` (it accepts only `session.update`,
/// `session.input_audio_buffer.append`, `session.close`); the other sessions use the plain name.
fn audio_append_type(kind: SessionKind) -> &'static str {
    match kind {
        SessionKind::Translation => "session.input_audio_buffer.append",
        _ => "input_audio_buffer.append",
    }
}

/// Surfaces a session error the user should see (bad config, server-side error, dropped connection)
/// — wired by the app to show a notice, so a failure is never silent. Called off the audio thread.
pub type ErrorSink = Box<dyn Fn(&str) + Send>;

/// The per-vendor realtime protocol: everything that differs between providers — URL, auth, config
/// message, audio-frame encoding, event parsing, sample rate, finalisation. Cloneable so the worker
/// thread owns its own copy. A new live vendor is a new variant plus its arms in the methods below.
#[derive(Debug, Clone)]
enum Protocol {
    /// OpenAI Realtime — base64-PCM-in-JSON frames, `session.update` config, three session kinds.
    OpenAi { model: String, kind: SessionKind },
    /// Speechmatics RT — raw-binary PCM frames, `StartRecognition` config, `AddTranscript` finals.
    Speechmatics { operating_point: String },
    /// Google Gemini Live (BidiGenerateContent) — `Setup` with `inputAudioTranscription`,
    /// `realtimeInput` base64 audio, `serverContent.inputTranscription.text`. `x-goog-api-key` auth.
    Gemini { model: String },
}

impl Protocol {
    /// A short human label for the connection log.
    fn label(&self) -> String {
        match self {
            Protocol::OpenAi { model, kind } => format!("OpenAI {model} ({kind:?})"),
            Protocol::Speechmatics { operating_point } => {
                format!("Speechmatics ({operating_point})")
            }
            Protocol::Gemini { model } => format!("Gemini Live ({model})"),
        }
    }

    /// The WebSocket URL to connect to.
    fn url(&self) -> String {
        match self {
            Protocol::OpenAi { model, kind } => realtime_url(model, *kind),
            Protocol::Speechmatics { .. } => speechmatics_url(),
            Protocol::Gemini { .. } => gemini_url(),
        }
    }

    /// The auth header `(name, value)` for the handshake.
    fn auth(&self, key: &str) -> (HeaderName, String) {
        match self {
            // Gemini authenticates with an `x-goog-api-key` header carrying the raw key.
            Protocol::Gemini { .. } => (HeaderName::from_static("x-goog-api-key"), key.to_owned()),
            // OpenAI and Speechmatics use `Authorization: Bearer <key>`.
            _ => (AUTHORIZATION, format!("Bearer {key}")),
        }
    }

    /// The config message sent right after connect (`None` if config rides in the URL).
    fn config_message(&self, params: &ParamValues) -> Option<String> {
        match self {
            Protocol::OpenAi { model, kind } => Some(match kind {
                SessionKind::Transcription => build_transcription_update(model, params),
                SessionKind::Model => build_model_update(params),
                SessionKind::Assist => build_assist_update(params),
                SessionKind::Translation => build_translation_update(params),
            }),
            Protocol::Speechmatics { operating_point } => {
                Some(speechmatics_start(operating_point, params))
            }
            Protocol::Gemini { model } => Some(gemini_setup(model, params)),
        }
    }

    /// Frames one PCM16-LE chunk into a WS message — vendor-specific: base64-in-JSON or raw binary.
    fn audio_message(&self, pcm16_le: &[u8]) -> Message {
        match self {
            Protocol::OpenAi { kind, .. } => {
                Message::text(build_audio_append(pcm16_le, audio_append_type(*kind)))
            }
            Protocol::Speechmatics { .. } => Message::binary(pcm16_le.to_vec()),
            Protocol::Gemini { .. } => Message::text(gemini_audio(pcm16_le)),
        }
    }

    /// The WS message injecting an authoritative text context item, or `None` if the vendor has no such
    /// concept (only the OpenAI sessions accept `conversation.item.create`).
    fn text_item_message(&self, text: &str) -> Option<Message> {
        match self {
            Protocol::OpenAi { .. } => Some(Message::text(build_text_item(text))),
            _ => None,
        }
    }

    /// The WS message explicitly triggering a response (`response.create`), or `None` for vendors that
    /// don't separate the trigger from turn detection.
    fn create_response_message(&self) -> Option<Message> {
        match self {
            Protocol::OpenAi { .. } => Some(Message::text(build_response_create())),
            _ => None,
        }
    }

    /// Parses one incoming text frame into a [`ServerEvent`].
    fn parse(&self, text: &str) -> ServerEvent {
        match self {
            Protocol::OpenAi { kind, .. } => parse_server_event(text, *kind),
            Protocol::Speechmatics { .. } => speechmatics_parse(text),
            Protocol::Gemini { .. } => gemini_parse(text),
        }
    }

    /// The PCM16 mono rate this vendor wants the pipeline to resample to.
    fn sample_rate(&self) -> u32 {
        match self {
            Protocol::OpenAi { .. } => REALTIME_SAMPLE_RATE,
            Protocol::Speechmatics { .. } => SPEECHMATICS_SAMPLE_RATE,
            Protocol::Gemini { .. } => GEMINI_SAMPLE_RATE,
        }
    }

    /// How utterances finalise for this vendor.
    fn finalize_mode(&self) -> FinalizeMode {
        match self {
            Protocol::OpenAi { kind, .. } => FinalizeMode::for_kind(*kind),
            // Speechmatics + Gemini stream incremental input transcription we append; commit on a pause.
            Protocol::Speechmatics { .. } | Protocol::Gemini { .. } => FinalizeMode::Gap,
        }
    }
}

/// Builds the realtime engine for `protocol` + `model`, authenticating with `key`. The single entry
/// point the app calls — it picks the [`Protocol`] adapter and connects the shared bridge.
pub fn build_realtime_engine(
    protocol: StreamingProtocol,
    model: &str,
    key: &str,
    params: &ParamValues,
    on_error: ErrorSink,
) -> Result<Box<dyn StreamingAsrEngine>> {
    let proto = match protocol {
        StreamingProtocol::OpenAiRealtime => Protocol::OpenAi {
            model: model.to_owned(),
            kind: session_kind(model),
        },
        StreamingProtocol::Speechmatics => Protocol::Speechmatics {
            operating_point: model.to_owned(),
        },
        StreamingProtocol::GeminiLive => Protocol::Gemini {
            model: model.to_owned(),
        },
        other => {
            return Err(WispError::Engine(format!(
                "live streaming isn't wired for {other:?} yet"
            )))
        }
    };
    // Transcription/translation/model sessions don't self-reconnect — their callers restart on a drop.
    Ok(Box::new(RealtimeEngine::connect(
        proto, key, params, on_error, false,
    )?))
}

/// Builds an AI-assist realtime engine: an OpenAI realtime `model` that listens to the meeting audio
/// and answers each turn under `instructions` (the user's assist prompt), authenticating with `key`.
/// `params` carries optional VAD/noise knobs. Unlike [`build_realtime_engine`] the session runs no
/// parallel transcription — the model's text output is the sole, primary stream.
///
/// Assist is OpenAI-realtime-only for now (the `response.output_text` shape is OpenAI-specific); a
/// Gemini-live assist would be a new arm here.
pub fn build_assist_engine(
    model: &str,
    key: &str,
    instructions: &str,
    params: &ParamValues,
    on_error: ErrorSink,
) -> Result<Box<dyn StreamingAsrEngine>> {
    // The assist prompt rides in `instructions`, where `build_assist_update` reads it; overlay it onto
    // a params copy so the caller can still pass VAD/noise overrides alongside it.
    let mut params = params.clone();
    params.set("instructions", ParamValue::Text(instructions.to_owned()));

    let proto = Protocol::OpenAi {
        model: model.to_owned(),
        kind: SessionKind::Assist,
    };

    // The assist self-reconnects on a dropped socket so a long meeting survives a network blip.
    Ok(Box::new(RealtimeEngine::connect(
        proto, key, &params, on_error, true,
    )?))
}

/// The advanced param specs a live `protocol` + `model` exposes (for the UI), or empty if unwired.
pub fn streaming_param_specs(protocol: StreamingProtocol, model: &str) -> Vec<ParamSpec> {
    match protocol {
        StreamingProtocol::OpenAiRealtime => openai_param_specs(model),
        StreamingProtocol::Speechmatics => speechmatics_param_specs(),
        StreamingProtocol::GeminiLive => gemini_param_specs(),
        _ => Vec::new(),
    }
}

/// Speechmatics realtime endpoint (EU region by default; override for US/self-hosted).
const SPEECHMATICS_URL: &str = "wss://eu.rt.speechmatics.com/v2";
/// Env override for the Speechmatics realtime URL — air-gapped or a different region.
pub const SPEECHMATICS_URL_ENV: &str = "WISP_SPEECHMATICS_RT_URL";
/// Speechmatics realtime wants 16 kHz PCM16 mono.
const SPEECHMATICS_SAMPLE_RATE: u32 = 16_000;

fn speechmatics_url() -> String {
    std::env::var(SPEECHMATICS_URL_ENV).unwrap_or_else(|_| SPEECHMATICS_URL.to_owned())
}

/// The `StartRecognition` message: raw 16 kHz PCM16, a language + operating point, partials off (we
/// stream final segments and commit the line on a pause — see [`Protocol::finalize_mode`]).
fn speechmatics_start(operating_point: &str, params: &ParamValues) -> String {
    let language = params.text("language", "en");
    let max_delay = params.float("max_delay", 2.0);
    serde_json::json!({
        "message": "StartRecognition",
        "audio_format": {
            "type": "raw",
            "encoding": "pcm_s16le",
            "sample_rate": SPEECHMATICS_SAMPLE_RATE
        },
        "transcription_config": {
            "language": language,
            "operating_point": operating_point,
            "enable_partials": false,
            "max_delay": max_delay
        }
    })
    .to_string()
}

/// Maps a Speechmatics server message to a [`ServerEvent`]: `AddTranscript` carries a final segment
/// (appended to the growing line); `Error` surfaces; everything else (RecognitionStarted, Info,
/// AudioAdded, EndOfTranscript) is ignored.
fn speechmatics_parse(text: &str) -> ServerEvent {
    #[derive(serde::Deserialize)]
    struct Meta {
        #[serde(default)]
        transcript: String,
    }
    #[derive(serde::Deserialize)]
    struct Msg {
        message: String,
        #[serde(default)]
        metadata: Option<Meta>,
        #[serde(default)]
        reason: String,
        #[serde(rename = "type", default)]
        kind: String,
    }
    let Ok(msg) = serde_json::from_str::<Msg>(text) else {
        return ServerEvent::Other;
    };
    match msg.message.as_str() {
        "AddTranscript" => {
            ServerEvent::Delta(msg.metadata.map(|m| m.transcript).unwrap_or_default())
        }
        "Error" => ServerEvent::Error(if msg.reason.is_empty() {
            msg.kind
        } else {
            msg.reason
        }),
        _ => ServerEvent::Other,
    }
}

/// The advanced knobs a Speechmatics realtime session exposes: source language + max delay.
fn speechmatics_param_specs() -> Vec<ParamSpec> {
    vec![
        ParamSpec::enumerated(
            "language",
            "Language",
            "The spoken language — Speechmatics realtime needs it set (no auto-detect).",
            &[
                ("en", "English"),
                ("yue", "Cantonese"),
                ("cmn", "Chinese (Mandarin)"),
                ("ja", "Japanese"),
                ("ko", "Korean"),
                ("es", "Spanish"),
                ("fr", "French"),
                ("de", "German"),
            ],
            "en",
        )
        .basic(),
        ParamSpec::float(
            "max_delay",
            "Max delay",
            "How long (s) the server may wait before emitting a final — lower is snappier, higher is \
             more accurate.",
            1.0,
            4.0,
            0.5,
            2.0,
        ),
    ]
}

/// Google Gemini Live endpoint (BidiGenerateContent over WebSocket), env-overridable.
const GEMINI_URL: &str = "wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent";
/// Env override for the Gemini Live WebSocket URL.
pub const GEMINI_URL_ENV: &str = "WISP_GEMINI_LIVE_URL";
/// Gemini Live realtime audio is 16 kHz PCM16 mono.
const GEMINI_SAMPLE_RATE: u32 = 16_000;

fn gemini_url() -> String {
    std::env::var(GEMINI_URL_ENV).unwrap_or_else(|_| GEMINI_URL.to_owned())
}

/// The `Setup` message: the live model (as `models/<id>`), `inputAudioTranscription` on so the server
/// transcribes our audio, and `AUDIO` responses — the only modality the live/native-audio models
/// accept (TEXT is rejected). We ignore the model's spoken reply; only its input transcription matters.
fn gemini_setup(model: &str, params: &ParamValues) -> String {
    let mut setup = serde_json::json!({
        "model": format!("models/{model}"),
        "generationConfig": { "responseModalities": ["AUDIO"] },
        "inputAudioTranscription": {}
    });

    // A spoken-language hint sharply improves the input transcription: Gemini auto-detect can mis-read
    // Cantonese (e.g. as Thai). It rides in a system instruction since `inputAudioTranscription` itself
    // takes no language.
    if let Some(name) = gemini_language_name(&params.text("language", "")) {
        setup["systemInstruction"] = serde_json::json!({
            "parts": [{
                "text": format!("The user is speaking {name}. Transcribe their speech accurately and \
                                 verbatim in {name}; do not translate.")
            }]
        });
    }

    serde_json::json!({ "setup": setup }).to_string()
}

/// The human language name for a Gemini hint code, or `None` for auto-detect (empty/unknown).
fn gemini_language_name(code: &str) -> Option<&'static str> {
    match code {
        "yue" => Some("Cantonese"),
        "zh" => Some("Mandarin Chinese"),
        "en" => Some("English"),
        "ja" => Some("Japanese"),
        "ko" => Some("Korean"),
        _ => None,
    }
}

/// A `realtimeInput` frame carrying base64 PCM16 audio at 16 kHz.
fn gemini_audio(pcm16_le: &[u8]) -> String {
    serde_json::json!({
        "realtimeInput": {
            "audio": { "data": b64(pcm16_le), "mimeType": "audio/pcm;rate=16000" }
        }
    })
    .to_string()
}

/// Extracts the input-audio transcription from a Gemini `serverContent` message (the model's own
/// spoken reply under `modelTurn` is ignored — we only transcribe). Surfaces an `error` message.
fn gemini_parse(text: &str) -> ServerEvent {
    // Cheap pre-filter: the model's AUDIO-response frames are large base64 we don't use — only fully
    // parse frames that could carry an input transcription or an error.
    if !text.contains("inputTranscription") && !text.contains("\"error\"") {
        return ServerEvent::Other;
    }
    let Ok(v) = serde_json::from_str::<serde_json::Value>(text) else {
        return ServerEvent::Other;
    };
    if let Some(t) = v
        .pointer("/serverContent/inputTranscription/text")
        .and_then(|x| x.as_str())
    {
        return ServerEvent::Delta(t.to_owned());
    }
    if let Some(msg) = v.pointer("/error/message").and_then(|x| x.as_str()) {
        return ServerEvent::Error(msg.to_owned());
    }
    ServerEvent::Other
}

/// The advanced knobs a Gemini Live session exposes: a spoken-language hint (Gemini's auto-detect can
/// mis-read Cantonese, so setting it sharply improves accuracy).
fn gemini_param_specs() -> Vec<ParamSpec> {
    vec![ParamSpec::enumerated(
        "language",
        "Language",
        "A spoken-language hint for transcription. Auto-detect can mis-read Cantonese — set it.",
        &[
            ("", "Auto-detect"),
            ("yue", "Cantonese"),
            ("zh", "Chinese (Mandarin)"),
            ("en", "English"),
            ("ja", "Japanese"),
            ("ko", "Korean"),
        ],
        "",
    )
    .basic()]
}

/// A realtime, streaming [`StreamingAsrEngine`] backed by a vendor's WebSocket API.
///
/// Vendor-agnostic: the WS bridge (background thread owning the socket, the two channels, the
/// finalisation [`Accumulator`]) is shared, and a [`Protocol`] supplies the per-vendor specifics
/// (URL, auth, config message, audio-frame encoding, event parsing, sample rate). Construction
/// connects + configures the session (failing fast on a bad key or unreachable host).
/// A client→server message queued for the socket worker: a PCM16 audio frame, an authoritative text
/// context item, or an explicit response trigger. Text and response-trigger apply only to OpenAI
/// assist sessions; for other vendors the worker has no message for them and skips.
enum Outgoing {
    Audio(Vec<u8>),
    Text(String),
    CreateResponse,
}

struct RealtimeEngine {
    /// Messages to send (PCM16 frames at [`sample_rate`](Self::sample_rate), plus assist text items and
    /// response triggers), handed to the worker.
    out_tx: Sender<Outgoing>,
    /// Parsed transcript events from the worker.
    event_rx: Receiver<ServerEvent>,
    /// Signals the worker to close the socket and exit.
    stop: Arc<AtomicBool>,
    worker: Option<JoinHandle<()>>,
    /// Folds incoming server events into the growing (verbatim + secondary) hypotheses and decides
    /// when an utterance is final — the per-session-kind finalisation logic, unit-tested in isolation.
    acc: Accumulator,
    /// When the last server event arrived, for the gap-based finalisation the accumulator may need
    /// (translation has no per-utterance done; a model session waits a bounded time for its slower
    /// stream). `None` once an utterance has just finalised.
    last_event: Option<Instant>,
    /// Reports errors to the app so they surface in the UI instead of failing silently.
    on_error: ErrorSink,
    /// The PCM16 mono rate this session's protocol wants (the pipeline resamples to it).
    sample_rate: u32,
}

/// Env var overriding the OpenAI Realtime WebSocket URL — for an air-gapped deployment or an
/// OpenAI-compatible realtime gateway. Unset uses [`DEFAULT_REALTIME_URL`].
pub const REALTIME_URL_ENV: &str = "WISP_OPENAI_REALTIME_URL";

/// Server/semantic-VAD endpointing knobs shared by every OpenAI realtime session (transcription,
/// model, assist): how a turn is detected and when it ends. Server-VAD fields are inert under semantic
/// VAD and vice-versa, but exposing both keeps the panel one generic list.
fn realtime_endpointing_specs() -> Vec<ParamSpec> {
    vec![
        ParamSpec::enumerated(
            "vad_type",
            "Turn detection",
            "How the server decides an utterance ended. Server VAD uses a silence timer (fast, \
             predictable); Semantic VAD lets a model judge completion (fewer mid-sentence cuts).",
            &[
                ("server_vad", "Server VAD (fast)"),
                ("semantic_vad", "Semantic VAD (smart)"),
            ],
            "server_vad",
        )
        .basic(),
        ParamSpec::float(
            "vad_threshold",
            "Voice sensitivity",
            "Server VAD only: how clearly speech must stand out to count as voice. Higher = \
             stricter (fewer false starts in noise).",
            0.0,
            1.0,
            0.05,
            0.5,
        ),
        ParamSpec::int(
            "vad_silence_ms",
            "End-of-speech silence",
            "Server VAD only: how long a pause (ms) ends an utterance. Lower feels snappier; \
             higher avoids cutting mid-sentence.",
            100,
            2000,
            500,
        ),
        ParamSpec::int(
            "vad_prefix_padding_ms",
            "Lead-in padding",
            "Server VAD only: audio kept before speech starts (ms), so the first word isn't \
             clipped.",
            0,
            1000,
            300,
        ),
        ParamSpec::enumerated(
            "vad_eagerness",
            "Semantic eagerness",
            "Semantic VAD only: how soon to end a turn. Low waits longer (≤8s), high responds \
             fast (≤2s), auto = medium (≤4s).",
            &[
                ("auto", "Auto (medium)"),
                ("low", "Low (patient)"),
                ("medium", "Medium"),
                ("high", "High (snappy)"),
            ],
            "auto",
        ),
    ]
}

/// Server-side noise reduction applied before the model hears the input audio.
fn noise_reduction_spec() -> ParamSpec {
    ParamSpec::enumerated(
        "noise_reduction",
        "Noise reduction",
        "Server-side denoise before transcription.",
        &[
            ("off", "Off"),
            ("near_field", "Near-field (headset)"),
            ("far_field", "Far-field (room mic)"),
        ],
        "near_field",
    )
}

/// The advanced params the **realtime assist** exposes: turn-detection endpointing (when the assist
/// responds) + noise reduction — exactly the knobs [`build_assist_update`] already consumes. The
/// model's behaviour rides in the prompt (the session `instructions`), so it isn't repeated here;
/// language and biasing are transcription concerns that don't apply to the assist.
pub fn assist_realtime_param_specs() -> Vec<ParamSpec> {
    let mut specs = realtime_endpointing_specs();
    specs.push(noise_reduction_spec());
    specs
}

/// The tunable knobs an OpenAI realtime `model` exposes, each with a smart default — rendered by the
/// generic advanced-parameters UI. Per-model: a model session is steered by free-form `instructions`;
/// a transcription session takes a `language` + `prompt`. Both share VAD endpointing + noise knobs.
fn openai_param_specs(model: &str) -> Vec<ParamSpec> {
    // Shared by the transcription and model sessions: the spoken (source) language hint. Both run
    // gpt-4o-transcribe for the verbatim text, and a hint sharply improves accuracy on CJK — e.g.
    // Cantonese auto-detects as romanised gibberish, but `yue` transcribes proper characters.
    let language = || {
        ParamSpec::enumerated(
            "language",
            "Language",
            "The spoken language, or auto-detect. Setting it improves accuracy and latency \
                 (auto-detect mangles Cantonese in particular).",
            &[
                ("", "Auto-detect"),
                ("yue", "Cantonese"),
                ("zh", "Chinese (Mandarin)"),
                ("en", "English"),
                ("ja", "Japanese"),
                ("ko", "Korean"),
            ],
            "",
        )
        .basic()
    };

    // A translation session takes just a target language — it segments and endpoints internally.
    if session_kind(model) == SessionKind::Translation {
        return vec![
            ParamSpec::enumerated(
                "target_language",
                "Translate to",
                "The language to translate the speech into.",
                &[
                    ("en", "English"),
                    ("yue", "Cantonese"),
                    ("zh", "Chinese (Mandarin)"),
                    ("ja", "Japanese"),
                    ("ko", "Korean"),
                    ("es", "Spanish"),
                    ("fr", "French"),
                    ("de", "German"),
                ],
                "en",
            )
            .basic(),
            noise_reduction_spec(),
        ];
    }

    // A model session is steered by free-form instructions; a transcription session takes a
    // language + hints. Both share the server-VAD endpointing knobs + noise reduction.
    let mut specs = match session_kind(model) {
        SessionKind::Model => vec![
            ParamSpec::textarea(
                "instructions",
                "Instructions",
                "The system prompt steering the model — transcribes verbatim by default; edit it \
                     to translate (e.g. \"Translate the speech to English\") or change behaviour.",
                DEFAULT_TRANSCRIBE_INSTRUCTIONS,
            )
            .basic(),
            language(),
        ],
        _ => vec![
            language(),
            ParamSpec::text(
                "prompt",
                "Hints",
                "Names, jargon, or acronyms to bias the transcription (e.g. \"Acme, kubectl\").",
                "",
            )
            .basic(),
        ],
    };

    specs.extend(realtime_endpointing_specs());
    specs.push(noise_reduction_spec());

    specs
}

impl RealtimeEngine {
    /// Connects and configures a realtime session via `protocol`, authenticating with `api_key`.
    /// `params` carries the advanced knobs (missing keys fall back to defaults). Errors on a blank
    /// key, a failed connection, or a rejected handshake.
    fn connect(
        protocol: Protocol,
        api_key: &str,
        params: &ParamValues,
        on_error: ErrorSink,
        reconnect: bool,
    ) -> Result<Self> {
        let key = api_key.trim();
        if key.is_empty() {
            return Err(WispError::Engine(
                "the provider needs an API key".to_owned(),
            ));
        }

        let ws = connect_and_configure(&protocol, key, params)?;
        eprintln!("wisp: realtime connected ({})", protocol.label());

        let (out_tx, out_rx) = mpsc::channel::<Outgoing>();
        let (event_tx, event_rx) = mpsc::channel::<ServerEvent>();
        let stop = Arc::new(AtomicBool::new(false));
        let stop_worker = Arc::clone(&stop);

        let sample_rate = protocol.sample_rate();
        let mode = protocol.finalize_mode();
        // An assist session reconnects on a dropped socket (a long meeting survives a network blip); the
        // transcription/translation sessions don't (their callers restart) — so their path is unchanged.
        let reconnect = reconnect.then(|| ReconnectCtx {
            key: key.to_owned(),
            params: params.clone(),
        });
        let worker = std::thread::spawn(move || {
            run_socket(ws, &out_rx, &event_tx, &stop_worker, protocol, reconnect)
        });

        Ok(Self {
            out_tx,
            event_rx,
            stop,
            worker: Some(worker),
            acc: Accumulator::new(mode),
            last_event: None,
            on_error,
            sample_rate,
        })
    }

    /// Drains every server event received so far into the accumulator, surfacing any error to the
    /// app, and refreshing the gap clock only on content events — so the translation session's
    /// constant output-audio frames don't keep resetting it and block gap-based finalisation.
    fn drain_events(&mut self) {
        while let Ok(event) = self.event_rx.try_recv() {
            if let ServerEvent::Failed(msg) | ServerEvent::Error(msg) = &event {
                eprintln!("wisp: OpenAI Realtime error: {msg}");
                (self.on_error)(msg);
            }
            if self.acc.apply(event) {
                self.last_event = Some(Instant::now());
            }
        }
    }
}

impl StreamingAsrEngine for RealtimeEngine {
    fn accept_waveform(&mut self, sample_rate: u32, samples: &[f32]) -> StreamingResult {
        if !samples.is_empty() {
            // A send error means the worker is gone (connection dropped); we surface that through the
            // event channel, so just drop the audio here. Resample to the protocol's wanted rate.
            let _ = self.out_tx.send(Outgoing::Audio(to_pcm16(
                samples,
                sample_rate,
                self.sample_rate,
            )));
        }

        self.drain_events();

        // Idle time since the last event drives the accumulator's gap fallback (translation has no
        // per-utterance done; a model session waits a bounded time for its slower second stream).
        let idle = self.last_event.map_or(Duration::ZERO, |t| t.elapsed());
        if let Some(result) = self.acc.finalize(idle) {
            self.last_event = None;
            return result;
        }

        self.acc.partial()
    }

    /// The rate the protocol wants — the pipeline resamples the capture device to it natively (e.g.
    /// OpenAI 24 kHz keeps the 8–12 kHz fricative band a 16 kHz pass would discard).
    fn input_sample_rate(&self) -> u32 {
        self.sample_rate
    }

    fn reset(&mut self) {
        self.acc.reset();
        self.last_event = None;
    }

    /// Queues an authoritative text item for the session (the worker turns it into a
    /// `conversation.item.create`). A no-op on the wire for vendors without the concept.
    fn inject_text(&mut self, text: &str) {
        let _ = self.out_tx.send(Outgoing::Text(text.to_owned()));
    }

    /// Queues an explicit response trigger (`response.create`) — the assist drives replies on a
    /// controlled cadence instead of auto-replying each turn.
    fn request_response(&mut self) {
        let _ = self.out_tx.send(Outgoing::CreateResponse);
    }
}

impl Drop for RealtimeEngine {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

/// How an utterance is finalised, derived from the session kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FinalizeMode {
    /// One stream with an explicit completion event (a transcription session) — finalise on it.
    Done,
    /// One append-only stream with no per-utterance event (a translation session) — finalise when
    /// its deltas pause past the gap.
    Gap,
    /// Two parallel streams (a model session: the verbatim transcript plus the model's text output)
    /// — finalise once both complete, or when one completes and the other stalls past the gap.
    Dual,
}

impl FinalizeMode {
    fn for_kind(kind: SessionKind) -> Self {
        match kind {
            // Assist has one stream (the model's text output) with an explicit `.done`, like a plain
            // transcription — finalise on that event. No parallel verbatim stream to wait on.
            SessionKind::Transcription | SessionKind::Assist => FinalizeMode::Done,
            SessionKind::Translation => FinalizeMode::Gap,
            SessionKind::Model => FinalizeMode::Dual,
        }
    }

    /// How long one stream may stall before the utterance is finalised with what we have (unused for
    /// `Done`, which finalises on its completion event).
    fn gap(self) -> Duration {
        match self {
            FinalizeMode::Dual => DUAL_FINALIZE_GAP,
            _ => TRANSLATION_FINALIZE_GAP,
        }
    }

    /// Whether this session carries a parallel secondary stream alongside the primary text — a model
    /// session's translated output, or a translation session's source-language original. A plain
    /// transcription (`Done`) has only one stream.
    fn carries_aux(self) -> bool {
        !matches!(self, FinalizeMode::Done)
    }
}

/// Folds [`ServerEvent`]s into the in-progress utterance's verbatim hypothesis and (for a model
/// session) a parallel secondary hypothesis, and decides when the utterance is final.
///
/// Clock-injected: [`finalize`](Self::finalize) takes the idle time since the last event rather than
/// reading a clock, so the whole finalisation state machine is unit-testable without a socket.
struct Accumulator {
    mode: FinalizeMode,
    /// The growing verbatim transcript (the primary stream).
    primary: String,
    /// The growing secondary text (a model session's output, e.g. a translation); stays empty for
    /// single-stream sessions.
    aux: String,
    /// Set once the primary stream reports completion (its authoritative final text).
    primary_done: Option<String>,
    /// Set once the secondary stream reports completion.
    aux_done: Option<String>,
    /// A surfaced error forces the utterance to finalise with whatever has accumulated.
    errored: bool,
}

impl Accumulator {
    fn new(mode: FinalizeMode) -> Self {
        Self {
            mode,
            primary: String::new(),
            aux: String::new(),
            primary_done: None,
            aux_done: None,
            errored: false,
        }
    }

    /// Folds one server event into the hypotheses. Returns `true` if it was a *content* event (one
    /// that advanced a stream or closed the utterance) — the caller resets the gap clock only on
    /// those, so the translation session's continuous output-*audio* frames (which arrive even during
    /// silence) don't keep the gap from ever elapsing and leave the line forever unfinalised.
    fn apply(&mut self, event: ServerEvent) -> bool {
        match event {
            ServerEvent::Delta(d) => self.primary.push_str(&d),
            ServerEvent::Completed(text) => {
                self.primary = text.clone();
                self.primary_done = Some(text);
            }
            ServerEvent::AuxDelta(d) => self.aux.push_str(&d),
            ServerEvent::AuxDone(text) => {
                self.aux = text.clone();
                self.aux_done = Some(text);
            }
            ServerEvent::Failed(_) | ServerEvent::Error(_) => self.errored = true,
            ServerEvent::Other => return false,
        }
        true
    }

    /// The current partial: the growing primary text, plus the secondary stream when this session
    /// carries one and it has content.
    fn partial(&self) -> StreamingResult {
        let aux = (self.mode.carries_aux() && !self.aux.is_empty()).then(|| self.aux.clone());
        StreamingResult {
            text: self.primary.clone(),
            is_endpoint: false,
            aux,
        }
    }

    /// Returns the final result and resets for the next utterance once this one has completed; `idle`
    /// is the time since the last server event, used for the gap fallbacks. `None` while in progress.
    fn finalize(&mut self, idle: Duration) -> Option<StreamingResult> {
        if !self.should_finalize(idle) {
            return None;
        }

        let text = self
            .primary_done
            .take()
            .unwrap_or_else(|| std::mem::take(&mut self.primary));
        let aux = if self.mode.carries_aux() {
            let aux = self
                .aux_done
                .take()
                .unwrap_or_else(|| std::mem::take(&mut self.aux));
            (!aux.is_empty()).then_some(aux)
        } else {
            None
        };

        self.reset();
        Some(StreamingResult {
            text,
            is_endpoint: true,
            aux,
        })
    }

    /// Whether the in-progress utterance is complete.
    fn should_finalize(&self, idle: Duration) -> bool {
        if self.errored {
            return true;
        }
        match self.mode {
            // Finalise on the verbatim completion event.
            FinalizeMode::Done => self.primary_done.is_some(),
            // No completion events: finalise once the append-only stream(s) pause. Translation runs
            // two — the translated text (primary) and the source-language original (aux).
            FinalizeMode::Gap => {
                (!self.primary.is_empty() || !self.aux.is_empty()) && idle >= self.mode.gap()
            }
            // Two streams: both done, or one done and the other stalled past the gap.
            FinalizeMode::Dual => {
                let either = self.primary_done.is_some() || self.aux_done.is_some();
                let both = self.primary_done.is_some() && self.aux_done.is_some();
                both || (either && idle >= self.mode.gap())
            }
        }
    }

    /// Clears all in-progress state for the next utterance.
    fn reset(&mut self) {
        self.primary.clear();
        self.aux.clear();
        self.primary_done = None;
        self.aux_done = None;
        self.errored = false;
    }
}

/// One transcript event from the Realtime socket, mapped from the server event stream.
#[derive(Debug, Clone, PartialEq, Eq)]
enum ServerEvent {
    /// An incremental hypothesis update for the in-progress utterance (the verbatim transcript).
    Delta(String),
    /// The final transcript for a completed utterance (authoritative, punctuated).
    Completed(String),
    /// An incremental update for the *secondary* stream — a model session's instruction-steered text
    /// output (e.g. a translation), running in parallel with the verbatim `Delta` stream.
    AuxDelta(String),
    /// The final secondary text for the in-progress utterance.
    AuxDone(String),
    /// Transcription failed for an item.
    Failed(String),
    /// A session-level error (bad key/model, quota, …).
    Error(String),
    /// Any other event we don't act on (session lifecycle, VAD markers, …).
    Other,
}

/// Opens the Realtime WebSocket at `url`, authenticating with `key`. Fails fast on connect/handshake
/// errors.
fn connect(url: &str, auth: (HeaderName, String)) -> Result<WebSocket<MaybeTlsStream<TcpStream>>> {
    let mut request = url
        .into_client_request()
        .map_err(|e| engine_err("bad realtime url", e))?;

    let (name, value) = auth;
    let value = HeaderValue::from_str(&value).map_err(|e| engine_err("auth header", e))?;
    request.headers_mut().insert(name, value);

    let (ws, _response) = tungstenite::connect(request).map_err(|e| engine_err("connect", e))?;
    Ok(ws)
}

/// Opens and configures a realtime session for `protocol`: connect, send the config message while
/// still blocking (so it lands before the worker streams audio), then switch to non-blocking. Shared by
/// the initial connect and the assist reconnect path.
fn connect_and_configure(
    protocol: &Protocol,
    key: &str,
    params: &ParamValues,
) -> Result<WebSocket<MaybeTlsStream<TcpStream>>> {
    let mut ws = connect(&protocol.url(), protocol.auth(key))?;

    if let Some(config) = protocol.config_message(params) {
        ws.send(Message::text(config))
            .map_err(|e| engine_err("session config", e))?;
    }
    set_nonblocking(&mut ws);

    Ok(ws)
}

/// What the socket worker needs to rebuild a dropped assist connection: the key + params to re-auth and
/// re-send the session config. (The protocol — URL, auth scheme, config shape — is already owned.)
struct ReconnectCtx {
    key: String,
    params: ParamValues,
}

/// Backoff schedule (ms) for reconnecting a dropped assist socket — a few escalating waits, then give
/// up. Bounded so a permanently-down endpoint surfaces an error instead of retrying forever.
const RECONNECT_BACKOFF_MS: &[u64] = &[500, 1_000, 2_000, 4_000, 8_000];

/// Sleeps `ms`, but in small chunks so a Stop request is honoured within ~100 ms instead of after the
/// full wait. Returns false if Stop fired (so the caller bails).
fn sleep_unless_stopped(ms: u64, stop: &AtomicBool) -> bool {
    let mut left = ms;
    while left > 0 {
        if stop.load(Ordering::Relaxed) {
            return false;
        }
        let chunk = left.min(100);
        std::thread::sleep(Duration::from_millis(chunk));
        left -= chunk;
    }
    !stop.load(Ordering::Relaxed)
}

/// Tries to reopen + reconfigure a dropped session, waiting between attempts per
/// [`RECONNECT_BACKOFF_MS`] and bailing the moment Stop is requested. `Some(ws)` on success, `None` if
/// every attempt failed (or Stop fired). Reconnecting loses the server-side conversation history; the
/// assist re-seeds context from the transcript finals it keeps injecting.
fn reconnect_with_backoff(
    protocol: &Protocol,
    ctx: &ReconnectCtx,
    stop: &AtomicBool,
) -> Option<WebSocket<MaybeTlsStream<TcpStream>>> {
    for &delay in RECONNECT_BACKOFF_MS {
        if !sleep_unless_stopped(delay, stop) {
            return None;
        }

        match connect_and_configure(protocol, &ctx.key, &ctx.params) {
            Ok(ws) => {
                eprintln!("wisp: realtime reconnected ({})", protocol.label());
                return Some(ws);
            }
            Err(e) => eprintln!("wisp: realtime reconnect attempt failed: {e}"),
        }
    }

    None
}

/// Puts the socket's underlying TCP stream in non-blocking mode, so the worker can poll reads
/// without blocking the sends it interleaves.
fn set_nonblocking(ws: &mut WebSocket<MaybeTlsStream<TcpStream>>) {
    let stream = match ws.get_mut() {
        MaybeTlsStream::Plain(s) => Some(s),
        MaybeTlsStream::Rustls(s) => Some(&mut s.sock),
        _ => None,
    };
    if let Some(s) = stream {
        let _ = s.set_nonblocking(true);
    }
}

/// The background loop owning the socket: send queued outgoing messages (audio frames, assist text
/// items, response triggers), drain incoming transcript events, until asked to stop or the connection
/// drops.
fn run_socket(
    mut ws: WebSocket<MaybeTlsStream<TcpStream>>,
    out_rx: &Receiver<Outgoing>,
    event_tx: &Sender<ServerEvent>,
    stop: &AtomicBool,
    protocol: Protocol,
    reconnect: Option<ReconnectCtx>,
) {
    loop {
        match pump_socket(&mut ws, out_rx, event_tx, stop, &protocol) {
            PumpOutcome::Stopped => break,
            // The connection dropped. An assist session reconnects with backoff (a long meeting survives
            // a blip); every other session surfaces the error and ends — its caller restarts — unchanged.
            PumpOutcome::Dropped(msg) => {
                match reconnect
                    .as_ref()
                    .and_then(|ctx| reconnect_with_backoff(&protocol, ctx, stop))
                {
                    Some(fresh) => ws = fresh,
                    None => {
                        let _ = event_tx.send(ServerEvent::Error(msg));
                        break;
                    }
                }
            }
        }
    }

    let _ = ws.close(None);
    let _ = ws.flush();
}

/// Why [`pump_socket`] returned: Stop was requested, or the connection dropped (with the reason).
enum PumpOutcome {
    Stopped,
    Dropped(String),
}

/// Drives one live socket connection: send queued outgoing messages, drain incoming events, until Stop
/// is requested ([`PumpOutcome::Stopped`]) or the connection drops ([`PumpOutcome::Dropped`]). The
/// caller decides whether a drop ends the session or reconnects.
fn pump_socket(
    ws: &mut WebSocket<MaybeTlsStream<TcpStream>>,
    out_rx: &Receiver<Outgoing>,
    event_tx: &Sender<ServerEvent>,
    stop: &AtomicBool,
    protocol: &Protocol,
) -> PumpOutcome {
    while !stop.load(Ordering::Relaxed) {
        while let Ok(out) = out_rx.try_recv() {
            // Map each queued message to its wire frame; text/response triggers are `None` (skipped)
            // for vendors that don't support them, so only OpenAI assist sessions emit them.
            let message = match out {
                Outgoing::Audio(pcm) => Some(protocol.audio_message(&pcm)),
                Outgoing::Text(text) => protocol.text_item_message(&text),
                Outgoing::CreateResponse => protocol.create_response_message(),
            };
            let Some(message) = message else { continue };

            if let Err(e) = ws.send(message) {
                if !is_would_block(&e) {
                    return PumpOutcome::Dropped(format!("send failed: {e}"));
                }
            }
        }

        loop {
            // Re-check stop inside the read loop: a translation session floods translated-audio
            // frames, so there's always another message buffered and the loop would never hit
            // `WouldBlock` to break — leaving Stop (which joins this thread) hung until the flood ends.
            if stop.load(Ordering::Relaxed) {
                break;
            }
            match ws.read() {
                Ok(Message::Text(text)) => {
                    // A translation session streams translated *audio* frames we don't use; skip them
                    // cheaply (no log, no parse) so the flood doesn't drown the loop.
                    if text.contains("output_audio.delta") {
                        continue;
                    }
                    // Diagnostic: surface exactly what the server sends (truncated) to the dev log,
                    // so a "connected but nothing transcribed" session is debuggable.
                    let snippet: String = text.chars().take(220).collect();
                    eprintln!("wisp realtime ◀ {snippet}");
                    let _ = event_tx.send(protocol.parse(text.as_str()));
                }
                Ok(Message::Binary(bytes)) => {
                    // Some vendors (Gemini Live) send their JSON `serverContent` as binary frames, not
                    // text — decode and parse the same way so the transcript isn't silently dropped.
                    if let Ok(text) = std::str::from_utf8(&bytes) {
                        let snippet: String = text.chars().take(220).collect();
                        eprintln!("wisp realtime ◀ {snippet}");
                        let _ = event_tx.send(protocol.parse(text));
                    }
                }
                Ok(Message::Close(frame)) => {
                    eprintln!("wisp: realtime closed: {frame:?}");
                    return PumpOutcome::Dropped("connection closed".to_owned());
                }
                Ok(_) => {} // ping/pong — ignore
                Err(e) if is_would_block(&e) => break,
                Err(e) => {
                    return PumpOutcome::Dropped(format!("read failed: {e}"));
                }
            }
        }

        std::thread::sleep(Duration::from_millis(5));
    }

    PumpOutcome::Stopped
}

/// Whether a tungstenite error is just "no data yet" on the non-blocking socket (not fatal).
fn is_would_block(e: &tungstenite::Error) -> bool {
    matches!(e, tungstenite::Error::Io(io) if io.kind() == ErrorKind::WouldBlock)
}

/// The server-side endpointing config, chosen by the `vad_type` param: `server_vad` (energy
/// threshold + fixed silence window — fast, predictable) or `semantic_vad` (a model judges when the
/// turn is semantically complete, tuned by `eagerness` — fewer mid-sentence cuts). Each mode reads
/// only its own knobs, so the irrelevant ones are simply ignored.
fn turn_detection(params: &ParamValues) -> serde_json::Value {
    match params.text("vad_type", "server_vad").as_str() {
        "semantic_vad" => serde_json::json!({
            "type": "semantic_vad",
            "eagerness": params.text("vad_eagerness", "auto")
        }),
        _ => serde_json::json!({
            "type": "server_vad",
            "threshold": params.float("vad_threshold", 0.5),
            "prefix_padding_ms": params.int("vad_prefix_padding_ms", 300),
            "silence_duration_ms": params.int("vad_silence_ms", 500)
        }),
    }
}

/// The shared `audio.input` config: 24 kHz PCM, endpointing (`server_vad`/`semantic_vad`) tuned by
/// `params`, and (unless `off`) server-side noise reduction. `create_response` is set for a model
/// session so the model replies on each turn; a transcription session leaves it off.
fn audio_input(params: &ParamValues, create_response: bool) -> serde_json::Value {
    let mut turn_detection = turn_detection(params);
    if create_response {
        turn_detection["create_response"] = serde_json::json!(true);
        // Don't let a new utterance cancel the in-progress response — by default a new turn barges in
        // and cancels it, which drops the previous utterance's transcript on fast back-to-back speech.
        turn_detection["interrupt_response"] = serde_json::json!(false);
    }

    let mut input = serde_json::json!({
        "format": { "type": "audio/pcm", "rate": REALTIME_SAMPLE_RATE },
        "turn_detection": turn_detection
    });

    let noise = params.text("noise_reduction", "near_field");
    if noise != "off" {
        input["noise_reduction"] = serde_json::json!({ "type": noise });
    }
    input
}

/// The GA `session.update` for a transcription session (dedicated ASR): the model plus optional
/// language/prompt under `audio.input.transcription`, server-VAD endpointing, optional noise reduction.
fn build_transcription_update(model: &str, params: &ParamValues) -> String {
    let mut transcription = serde_json::json!({ "model": model });
    for key in ["language", "prompt"] {
        let value = params.text(key, "");
        if !value.is_empty() {
            transcription[key] = serde_json::json!(value);
        }
    }

    let mut input = audio_input(params, false);
    input["transcription"] = transcription;

    serde_json::json!({
        "type": "session.update",
        "session": { "type": "transcription", "audio": { "input": input } }
    })
    .to_string()
}

/// The `session.update` for a realtime model session with text output: the model is steered by the
/// user's free-form `instructions` and replies in text on each server-VAD turn.
fn build_model_update(params: &ParamValues) -> String {
    let instructions = params.text("instructions", DEFAULT_TRANSCRIBE_INSTRUCTIONS);

    // Run a dedicated input-audio transcription alongside the model's text output. The transcription
    // is the gap-free verbatim stream that backs the transcript; the model's response — which can
    // drop an utterance under rapid back-to-back speech — becomes the secondary (translation) stream.
    // A language hint sharply improves CJK accuracy (Cantonese otherwise romanises into gibberish).
    let mut transcription = serde_json::json!({ "model": MODEL_SESSION_TRANSCRIBE_MODEL });
    let language = params.text("language", "");
    if !language.is_empty() {
        transcription["language"] = serde_json::json!(language);
    }

    let mut input = audio_input(params, true);
    input["transcription"] = transcription;

    serde_json::json!({
        "type": "session.update",
        "session": {
            "type": "realtime",
            "output_modalities": ["text"],
            "instructions": instructions,
            "audio": { "input": input }
        }
    })
    .to_string()
}

/// The `session.update` for an assist session: a realtime model steered by the user's `instructions`,
/// hearing the live audio and answering in text. Unlike [`build_model_update`] it runs **no** parallel
/// `transcription` (the live pipeline already transcribes this audio — a second ASR would only add cost
/// and latency), and it sets `create_response: false` so replies fire only on an explicit
/// `response.create` (the assist drives a controlled cadence — throttle + manual — instead of answering
/// every turn). Server VAD still segments the input so the model has turn-structured context.
fn build_assist_update(params: &ParamValues) -> String {
    let instructions = params.text("instructions", DEFAULT_TRANSCRIBE_INSTRUCTIONS);

    serde_json::json!({
        "type": "session.update",
        "session": {
            "type": "realtime",
            "output_modalities": ["text"],
            "instructions": instructions,
            "audio": { "input": audio_input(params, false) }
        }
    })
    .to_string()
}

/// The `session.update` for a translation session: the input is transcribed by gpt-realtime-whisper
/// and translated to the target `language`; the translated text streams as `session.output_transcript`
/// deltas. The session handles its own turn detection, so no server-VAD knobs.
///
/// The session `type` and `model` are fixed at connection (the `/translations` path + `?model=` query)
/// and are *not* sent here — including them is rejected ("Unknown parameter: 'session.model'"). Only
/// `audio.{output.language, input.transcription, input.noise_reduction}` are updatable.
fn build_translation_update(params: &ParamValues) -> String {
    let target = params.text("target_language", "en");

    let noise = params.text("noise_reduction", "near_field");
    let noise_reduction = if noise == "off" {
        serde_json::Value::Null
    } else {
        serde_json::json!({ "type": noise })
    };

    serde_json::json!({
        "type": "session.update",
        "session": {
            "audio": {
                "input": {
                    "transcription": { "model": "gpt-realtime-whisper" },
                    "noise_reduction": noise_reduction
                },
                "output": { "language": target }
            }
        }
    })
    .to_string()
}

/// A `conversation.item.create` carrying an authoritative user text item — the app's diarized,
/// speaker-attributed transcript final — injected so the model reasons over clean text alongside the
/// audio it hears, correcting its own ASR (especially for CJK) and recovering who said what.
fn build_text_item(text: &str) -> String {
    serde_json::json!({
        "type": "conversation.item.create",
        "item": {
            "type": "message",
            "role": "user",
            "content": [{ "type": "input_text", "text": text }]
        }
    })
    .to_string()
}

/// A `response.create` — explicitly asks the model to reply now. The assist triggers this on a
/// controlled cadence (throttle + manual) instead of letting the server auto-reply every turn.
fn build_response_create() -> String {
    serde_json::json!({ "type": "response.create" }).to_string()
}

/// An audio-append frame carrying base64 PCM16 audio. The event `type` differs by session: the
/// translation endpoint namespaces it as `session.input_audio_buffer.append`, the rest use the plain
/// `input_audio_buffer.append` — see [`audio_append_type`].
fn build_audio_append(pcm16_le: &[u8], append_type: &str) -> String {
    serde_json::json!({
        "type": append_type,
        "audio": b64(pcm16_le),
    })
    .to_string()
}

/// Maps a server event JSON to a [`ServerEvent`]; anything unrecognised is [`ServerEvent::Other`].
///
/// `kind` decides where the model's `response.output_text.*` lands: a [`Model`](SessionKind::Model)
/// session runs a parallel verbatim transcript as the primary stream, so the model's output is the
/// *secondary* (`Aux*`); an [`Assist`](SessionKind::Assist) session has no transcript, so the model's
/// output IS the primary (`Delta`/`Completed`).
fn parse_server_event(json: &str, kind: SessionKind) -> ServerEvent {
    #[derive(serde::Deserialize)]
    struct Err_ {
        #[serde(default)]
        message: String,
    }
    #[derive(serde::Deserialize)]
    struct Event {
        #[serde(rename = "type")]
        kind: String,
        #[serde(default)]
        delta: String,
        #[serde(default)]
        transcript: String,
        /// `response.output_text.done` carries the final text here (not in `transcript`).
        #[serde(default)]
        text: String,
        #[serde(default)]
        error: Option<Err_>,
    }

    let Ok(event) = serde_json::from_str::<Event>(json) else {
        return ServerEvent::Other;
    };
    let message = || {
        event
            .error
            .as_ref()
            .map(|e| e.message.clone())
            .unwrap_or_default()
    };

    match event.kind.as_str() {
        // Transcription session.
        "conversation.item.input_audio_transcription.delta" => ServerEvent::Delta(event.delta),
        "conversation.item.input_audio_transcription.completed" => {
            ServerEvent::Completed(event.transcript)
        }
        "conversation.item.input_audio_transcription.failed" => ServerEvent::Failed(message()),
        // The model's text output. In a model session it's the *secondary* stream (the verbatim
        // transcript above is primary); in an assist session there is no transcript, so it's primary.
        "response.output_text.delta" if kind == SessionKind::Assist => {
            ServerEvent::Delta(event.delta)
        }
        "response.output_text.done" if kind == SessionKind::Assist => {
            ServerEvent::Completed(event.text)
        }
        "response.output_text.delta" => ServerEvent::AuxDelta(event.delta),
        "response.output_text.done" => ServerEvent::AuxDone(event.text),
        // Translation session: the translated text is the primary stream; the source-language
        // transcript is the secondary (original) stream shown alongside it. (The session also streams
        // translated *audio* — `session.output_audio.delta` — which we don't use and let fall through.)
        "session.output_transcript.delta" => ServerEvent::Delta(event.delta),
        "session.input_transcript.delta" => ServerEvent::AuxDelta(event.delta),
        "error" => ServerEvent::Error(message()),
        _ => ServerEvent::Other,
    }
}

/// Resamples `samples` from `src_rate` to 24 kHz and encodes them as little-endian PCM16 bytes —
/// the wire format the Realtime API's `pcm16` input expects.
fn to_pcm16(samples: &[f32], src_rate: u32, target_rate: u32) -> Vec<u8> {
    let resampled = resample_linear(samples, src_rate, target_rate);

    let mut bytes = Vec::with_capacity(resampled.len() * 2);
    for sample in resampled {
        let pcm = (sample.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
        bytes.extend_from_slice(&pcm.to_le_bytes());
    }
    bytes
}

/// Linear-interpolation resample of `input` from `from` to `to` Hz. Simple upsampling for the cloud
/// send path (no anti-alias filter needed when upsampling 16 kHz → 24 kHz).
fn resample_linear(input: &[f32], from: u32, to: u32) -> Vec<f32> {
    if from == to || input.is_empty() || from == 0 {
        return input.to_vec();
    }

    let ratio = f64::from(to) / f64::from(from);
    let out_len = (input.len() as f64 * ratio).round() as usize;
    let last = input.len() - 1;

    let mut out = Vec::with_capacity(out_len);
    for i in 0..out_len {
        let src = i as f64 / ratio;
        let j = src.floor() as usize;
        let frac = (src - j as f64) as f32;
        let a = input[j.min(last)];
        let b = input[(j + 1).min(last)];
        out.push(a + (b - a) * frac);
    }
    out
}

/// Maps a streaming-path failure to a [`WispError`].
fn engine_err(context: &str, e: impl std::fmt::Display) -> WispError {
    WispError::Engine(format!("OpenAI realtime {context}: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn realtime_url_env_constant_is_pinned() {
        // Renaming this breaks any operator who pinned a realtime gateway via env — hard-pin it.
        assert_eq!(REALTIME_URL_ENV, "WISP_OPENAI_REALTIME_URL");
    }

    #[test]
    fn realtime_url_routes_each_session_kind_to_its_endpoint() {
        // The path/intent identifies the session type; the model rides in the session config except
        // for a model session, where it must be in the query so the server provisions it pre-config.
        assert!(
            realtime_url("gpt-4o-transcribe", SessionKind::Transcription)
                .ends_with("/realtime?intent=transcription")
        );
        assert!(realtime_url("gpt-realtime", SessionKind::Model)
            .ends_with("/realtime?model=gpt-realtime"));
        // An assist session connects with `?model=…` just like a model session.
        assert!(realtime_url("gpt-4o-realtime-preview", SessionKind::Assist)
            .ends_with("/realtime?model=gpt-4o-realtime-preview"));
        // Translation lives on its own sub-path AND needs the model query (else `missing_model`).
        assert!(
            realtime_url("gpt-realtime-translate", SessionKind::Translation)
                .ends_with("/realtime/translations?model=gpt-realtime-translate")
        );
    }

    #[test]
    fn session_update_uses_param_defaults_for_model_vad_and_noise() {
        let params = ParamValues::from_specs(&openai_param_specs("gpt-4o-transcribe"));
        let json = build_transcription_update("gpt-4o-transcribe", &params);
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();

        assert_eq!(v["type"], "session.update");
        assert_eq!(v["session"]["type"], "transcription");

        let input = &v["session"]["audio"]["input"];
        assert_eq!(input["format"]["type"], "audio/pcm");
        assert_eq!(input["format"]["rate"], 24_000);
        assert_eq!(input["transcription"]["model"], "gpt-4o-transcribe");
        assert!(
            input["transcription"]["language"].is_null(),
            "language defaults to auto (empty) → omitted"
        );

        let vad = &input["turn_detection"];
        assert_eq!(vad["type"], "server_vad");
        assert_eq!(vad["threshold"], 0.5);
        assert_eq!(vad["silence_duration_ms"], 500);
        assert_eq!(vad["prefix_padding_ms"], 300);
        assert_eq!(input["noise_reduction"]["type"], "near_field");
    }

    #[test]
    fn session_update_applies_language_prompt_and_omits_noise_when_off() {
        use wisp_core::params::ParamValue;

        let mut params = ParamValues::from_specs(&openai_param_specs("gpt-4o-transcribe"));
        params.set("language", ParamValue::Text("yue".to_owned()));
        params.set("prompt", ParamValue::Text("kubectl".to_owned()));
        params.set("vad_threshold", ParamValue::Float(0.8));
        params.set("noise_reduction", ParamValue::Text("off".to_owned()));

        let json = build_transcription_update("gpt-4o-transcribe", &params);
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();

        let input = &v["session"]["audio"]["input"];
        assert_eq!(input["transcription"]["language"], "yue");
        assert_eq!(input["transcription"]["prompt"], "kubectl");
        assert_eq!(input["turn_detection"]["threshold"], 0.8);
        assert!(
            input["noise_reduction"].is_null(),
            "noise_reduction omitted when off"
        );
    }

    #[test]
    fn param_specs_expose_language_prompt_and_vad() {
        let keys = openai_param_specs("gpt-4o-transcribe")
            .into_iter()
            .map(|s| s.key)
            .collect::<Vec<_>>();

        for expected in [
            "language",
            "prompt",
            "vad_type",
            "vad_threshold",
            "vad_silence_ms",
            "vad_prefix_padding_ms",
            "vad_eagerness",
            "noise_reduction",
        ] {
            assert!(keys.iter().any(|k| k == expected), "missing {expected}");
        }
    }

    #[test]
    fn assist_realtime_specs_are_endpointing_and_noise_only() {
        // The realtime assist exposes turn-detection + noise reduction — exactly the knobs
        // build_assist_update consumes — and NOT transcription-only concepts (language/prompt/
        // instructions), since the assist's behaviour rides in the prompt instructions instead.
        let keys = assist_realtime_param_specs()
            .into_iter()
            .map(|s| s.key)
            .collect::<Vec<_>>();

        for expected in [
            "vad_type",
            "vad_threshold",
            "vad_silence_ms",
            "vad_prefix_padding_ms",
            "vad_eagerness",
            "noise_reduction",
        ] {
            assert!(keys.iter().any(|k| k == expected), "missing {expected}");
        }
        for absent in ["language", "prompt", "instructions", "target_language"] {
            assert!(
                !keys.iter().any(|k| k == absent),
                "should not expose {absent}"
            );
        }
    }

    #[test]
    fn turn_detection_supports_server_and_semantic_vad() {
        use wisp_core::params::ParamValue;

        // Default is server_vad, carrying the energy threshold + silence/prefix windows.
        let params = ParamValues::from_specs(&openai_param_specs("gpt-4o-transcribe"));
        let td = turn_detection(&params);
        assert_eq!(td["type"], "server_vad");
        assert_eq!(td["silence_duration_ms"], 500);
        assert_eq!(td["threshold"], 0.5);
        assert!(td["eagerness"].is_null(), "server_vad has no eagerness");

        // Switching vad_type to semantic_vad yields the model-judged shape tuned by eagerness, with
        // none of the server_vad timers.
        let mut semantic = params.clone();
        semantic.set("vad_type", ParamValue::Text("semantic_vad".to_owned()));
        semantic.set("vad_eagerness", ParamValue::Text("high".to_owned()));
        let td = turn_detection(&semantic);
        assert_eq!(td["type"], "semantic_vad");
        assert_eq!(td["eagerness"], "high");
        assert!(td["threshold"].is_null(), "semantic_vad has no threshold");
        assert!(td["silence_duration_ms"].is_null());
    }

    #[test]
    fn model_session_uses_instructions_text_output_and_parses_output_text() {
        use wisp_core::params::ParamValue;

        // A gpt-realtime model is steered by `instructions`, and also takes a source `language` hint
        // (for its verbatim transcription) — but no `prompt` (the instructions cover biasing).
        let keys = openai_param_specs("gpt-realtime")
            .into_iter()
            .map(|s| s.key)
            .collect::<Vec<_>>();
        assert!(keys.iter().any(|k| k == "instructions"));
        assert!(keys.iter().any(|k| k == "language"));
        assert!(!keys.iter().any(|k| k == "prompt"));

        // The session.update is a model session: text output, instructions, server VAD that creates
        // a response each turn.
        let mut params = ParamValues::from_specs(&openai_param_specs("gpt-realtime"));
        params.set(
            "instructions",
            ParamValue::Text("Translate to English.".to_owned()),
        );
        params.set("language", ParamValue::Text("yue".to_owned()));
        let v: serde_json::Value = serde_json::from_str(&build_model_update(&params)).unwrap();
        assert_eq!(v["session"]["type"], "realtime");
        assert_eq!(v["session"]["output_modalities"][0], "text");
        assert_eq!(v["session"]["instructions"], "Translate to English.");

        // A dedicated input-audio transcription runs alongside the model — the gap-free verbatim
        // stream that backs the transcript (the model's own output is only the secondary stream). The
        // language hint rides on it, so Cantonese transcribes as proper characters, not romanised.
        let transcription = &v["session"]["audio"]["input"]["transcription"];
        assert_eq!(transcription["model"], "gpt-4o-transcribe");
        assert_eq!(transcription["language"], "yue");

        let vad = &v["session"]["audio"]["input"]["turn_detection"];
        assert_eq!(vad["create_response"], true);
        assert_eq!(
            vad["interrupt_response"], false,
            "a new utterance must not cancel the in-progress response (else fast speech drops a turn)"
        );

        // The verbatim transcript rides input_audio_transcription (the primary Delta/Completed); the
        // model's own text output is the *secondary* stream (AuxDelta/AuxDone), e.g. the translation.
        assert_eq!(
            parse_server_event(
                r#"{"type":"conversation.item.input_audio_transcription.delta","delta":"你"}"#,
                SessionKind::Model
            ),
            ServerEvent::Delta("你".to_owned())
        );
        assert_eq!(
            parse_server_event(
                r#"{"type":"response.output_text.delta","delta":"Hel"}"#,
                SessionKind::Model
            ),
            ServerEvent::AuxDelta("Hel".to_owned())
        );
        assert_eq!(
            parse_server_event(
                r#"{"type":"response.output_text.done","text":"Hello."}"#,
                SessionKind::Model
            ),
            ServerEvent::AuxDone("Hello.".to_owned())
        );
    }

    #[test]
    fn assist_session_drops_transcription_and_maps_output_text_to_primary() {
        // An assist session is a realtime model steered by `instructions`, with text output and a
        // response created each turn — but NO parallel transcription (the live pipeline already
        // transcribes; a second ASR would only add cost and latency).
        let mut params = ParamValues::new();
        params.set(
            "instructions",
            ParamValue::Text("You are a meeting copilot.".to_owned()),
        );
        let v: serde_json::Value = serde_json::from_str(&build_assist_update(&params)).unwrap();
        assert_eq!(v["session"]["type"], "realtime");
        assert_eq!(v["session"]["output_modalities"][0], "text");
        assert_eq!(v["session"]["instructions"], "You are a meeting copilot.");
        assert!(
            v["session"]["audio"]["input"]["turn_detection"]["create_response"].is_null(),
            "assist replies on an explicit response.create, not auto every turn"
        );
        assert!(
            v["session"]["audio"]["input"]["transcription"].is_null(),
            "assist runs no parallel transcription"
        );

        // With no transcript stream, the model's own output_text IS the primary (Delta/Completed),
        // finalised on its `.done` — not the secondary Aux* a model session would use.
        assert_eq!(
            parse_server_event(
                r#"{"type":"response.output_text.delta","delta":"Ask"}"#,
                SessionKind::Assist
            ),
            ServerEvent::Delta("Ask".to_owned())
        );
        assert_eq!(
            parse_server_event(
                r#"{"type":"response.output_text.done","text":"Ask about scope."}"#,
                SessionKind::Assist
            ),
            ServerEvent::Completed("Ask about scope.".to_owned())
        );
    }

    #[test]
    fn reconnect_backoff_is_bounded_and_non_decreasing() {
        assert!(
            !RECONNECT_BACKOFF_MS.is_empty(),
            "must attempt at least one reconnect"
        );
        assert!(
            RECONNECT_BACKOFF_MS.windows(2).all(|w| w[0] <= w[1]),
            "backoff waits must be non-decreasing"
        );
        // Pinned so a careless edit can't make it retry forever (unbounded) or hammer (no waits).
        assert_eq!(RECONNECT_BACKOFF_MS, &[500, 1_000, 2_000, 4_000, 8_000]);
    }

    #[test]
    fn assist_control_messages_inject_text_and_trigger_response_openai_only() {
        // An injected transcript final becomes a conversation.item.create user text item.
        let item: serde_json::Value =
            serde_json::from_str(&build_text_item("Them (Speaker 2): ship it Friday")).unwrap();
        assert_eq!(item["type"], "conversation.item.create");
        assert_eq!(item["item"]["role"], "user");
        assert_eq!(item["item"]["content"][0]["type"], "input_text");
        assert_eq!(
            item["item"]["content"][0]["text"],
            "Them (Speaker 2): ship it Friday"
        );

        // The explicit response trigger.
        let resp: serde_json::Value = serde_json::from_str(&build_response_create()).unwrap();
        assert_eq!(resp["type"], "response.create");

        // Only the OpenAI sessions emit these frames; other vendors have no equivalent and skip them.
        let openai = Protocol::OpenAi {
            model: "gpt-realtime".to_owned(),
            kind: SessionKind::Assist,
        };
        assert!(openai.text_item_message("hi").is_some());
        assert!(openai.create_response_message().is_some());

        let speechmatics = Protocol::Speechmatics {
            operating_point: "enhanced".to_owned(),
        };
        assert!(speechmatics.text_item_message("hi").is_none());
        assert!(speechmatics.create_response_message().is_none());
    }

    // ── Accumulator: the per-session-kind finalisation state machine (unit-tested without a socket) ──

    #[test]
    fn accumulator_done_mode_finalises_on_completed_with_no_aux() {
        let mut acc = Accumulator::new(FinalizeMode::Done);

        acc.apply(ServerEvent::Delta("he".to_owned()));
        assert_eq!(acc.partial(), StreamingResult::new("he", false));
        assert!(acc.finalize(Duration::ZERO).is_none(), "still in progress");

        acc.apply(ServerEvent::Completed("hello.".to_owned()));
        let result = acc.finalize(Duration::ZERO).expect("completed → final");
        assert_eq!(result, StreamingResult::new("hello.", true));
        assert_eq!(
            result.aux, None,
            "a transcription session has no secondary stream"
        );

        assert!(
            acc.finalize(Duration::ZERO).is_none(),
            "reset after finalising"
        );
        assert_eq!(acc.partial().text, "");
    }

    #[test]
    fn accumulator_gap_mode_finalises_only_after_a_pause() {
        let mut acc = Accumulator::new(FinalizeMode::Gap);
        acc.apply(ServerEvent::Delta("hola".to_owned()));

        assert!(
            acc.finalize(Duration::from_millis(500)).is_none(),
            "deltas still flowing — keep the line open"
        );
        let result = acc
            .finalize(TRANSLATION_FINALIZE_GAP)
            .expect("paused past the gap → final");
        assert_eq!(result, StreamingResult::new("hola", true));
    }

    #[test]
    fn accumulator_gap_mode_carries_a_secondary_stream() {
        // A translation session has no completion events and runs two append-only streams: the
        // translated text (primary) and the source-language original (aux). Both accumulate and
        // finalise together once they pause.
        let mut acc = Accumulator::new(FinalizeMode::Gap);
        acc.apply(ServerEvent::Delta("Bonjour".to_owned())); // translation (primary)
        acc.apply(ServerEvent::AuxDelta("Hello".to_owned())); // original (aux)

        assert_eq!(
            acc.partial(),
            StreamingResult {
                text: "Bonjour".to_owned(),
                is_endpoint: false,
                aux: Some("Hello".to_owned()),
            }
        );
        assert!(acc.finalize(Duration::from_millis(500)).is_none());

        let result = acc
            .finalize(TRANSLATION_FINALIZE_GAP)
            .expect("paused past the gap → final");
        assert_eq!(
            result,
            StreamingResult {
                text: "Bonjour".to_owned(),
                is_endpoint: true,
                aux: Some("Hello".to_owned()),
            }
        );
    }

    #[test]
    fn accumulator_dual_mode_finalises_when_both_streams_complete() {
        let mut acc = Accumulator::new(FinalizeMode::Dual);

        acc.apply(ServerEvent::Delta("你好".to_owned()));
        acc.apply(ServerEvent::AuxDelta("Hel".to_owned()));
        assert_eq!(
            acc.partial(),
            StreamingResult {
                text: "你好".to_owned(),
                is_endpoint: false,
                aux: Some("Hel".to_owned()),
            },
            "partial shows verbatim primary + the growing translation"
        );

        // Primary completes first; wait (small idle) for the slower secondary stream.
        acc.apply(ServerEvent::Completed("你好。".to_owned()));
        assert!(
            acc.finalize(Duration::ZERO).is_none(),
            "verbatim done but wait for the model's output"
        );

        acc.apply(ServerEvent::AuxDone("Hello.".to_owned()));
        let result = acc.finalize(Duration::ZERO).expect("both done → final");
        assert_eq!(
            result,
            StreamingResult {
                text: "你好。".to_owned(),
                is_endpoint: true,
                aux: Some("Hello.".to_owned()),
            }
        );
    }

    #[test]
    fn accumulator_dual_mode_falls_back_when_one_stream_stalls() {
        // The verbatim transcript finishes but the model's translation hangs: finalise with the
        // partial translation once the gap elapses, rather than holding the reliable transcript hostage.
        let mut acc = Accumulator::new(FinalizeMode::Dual);
        acc.apply(ServerEvent::Completed("你好。".to_owned()));
        acc.apply(ServerEvent::AuxDelta("Hel".to_owned()));

        assert!(
            acc.finalize(Duration::from_millis(500)).is_none(),
            "still within the wait window"
        );
        let result = acc
            .finalize(DUAL_FINALIZE_GAP)
            .expect("stalled second stream → fallback final");
        assert_eq!(
            result,
            StreamingResult {
                text: "你好。".to_owned(),
                is_endpoint: true,
                aux: Some("Hel".to_owned()),
            }
        );
    }

    #[test]
    fn accumulator_finalises_on_error_with_whatever_accumulated() {
        let mut acc = Accumulator::new(FinalizeMode::Dual);
        acc.apply(ServerEvent::Delta("partial".to_owned()));
        acc.apply(ServerEvent::Error("quota exceeded".to_owned()));

        let result = acc
            .finalize(Duration::ZERO)
            .expect("an error closes the utterance immediately");
        assert!(result.is_endpoint);
        assert_eq!(result.text, "partial");
    }

    #[test]
    fn apply_reports_content_events_so_audio_frames_dont_block_finalisation() {
        let mut acc = Accumulator::new(FinalizeMode::Gap);
        // Content events advance a stream → true, so the caller resets the gap clock.
        assert!(acc.apply(ServerEvent::Delta("Bon".to_owned())));
        assert!(acc.apply(ServerEvent::AuxDelta("Hi".to_owned())));
        // A non-content event (e.g. the translation session's constant output-audio frames) → false,
        // so it never resets the gap clock; silence can still elapse and finalise the line.
        assert!(!acc.apply(ServerEvent::Other));
    }

    #[test]
    fn finalize_mode_maps_each_session_kind() {
        assert_eq!(
            FinalizeMode::for_kind(SessionKind::Transcription),
            FinalizeMode::Done
        );
        assert_eq!(
            FinalizeMode::for_kind(SessionKind::Translation),
            FinalizeMode::Gap
        );
        assert_eq!(
            FinalizeMode::for_kind(SessionKind::Model),
            FinalizeMode::Dual
        );
        // Assist has one stream with an explicit done, like a transcription.
        assert_eq!(
            FinalizeMode::for_kind(SessionKind::Assist),
            FinalizeMode::Done
        );
    }

    #[test]
    fn translation_session_targets_a_language_and_parses_output_transcript() {
        use wisp_core::params::ParamValue;

        // A translation model exposes a target language, not instructions or server-VAD knobs.
        let keys = openai_param_specs("gpt-realtime-translate")
            .into_iter()
            .map(|s| s.key)
            .collect::<Vec<_>>();
        assert!(keys.iter().any(|k| k == "target_language"));
        assert!(
            !keys.iter().any(|k| k == "vad_threshold"),
            "no server-VAD knobs"
        );

        let mut params = ParamValues::from_specs(&openai_param_specs("gpt-realtime-translate"));
        params.set("target_language", ParamValue::Text("es".to_owned()));
        let v: serde_json::Value =
            serde_json::from_str(&build_translation_update(&params)).unwrap();

        // The session `type` and `model` are fixed at connection (URL), so the update carries neither
        // — only the updatable `audio` fields. Sending `session.model` is rejected by the server.
        assert!(
            v["session"]["type"].is_null(),
            "type is set by the URL, not the update"
        );
        assert!(
            v["session"]["model"].is_null(),
            "model is set by ?model=, not the update"
        );
        assert_eq!(v["session"]["audio"]["output"]["language"], "es");
        assert_eq!(
            v["session"]["audio"]["input"]["transcription"]["model"],
            "gpt-realtime-whisper"
        );

        // The translated output streams as session.output_transcript deltas.
        let delta = r#"{"type":"session.output_transcript.delta","delta":"Hola"}"#;
        assert_eq!(
            parse_server_event(delta, SessionKind::Translation),
            ServerEvent::Delta("Hola".to_owned())
        );
    }

    #[test]
    fn audio_append_carries_base64_pcm16() {
        let json = build_audio_append(&[0x01, 0x02, 0x03], audio_append_type(SessionKind::Model));
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();

        assert_eq!(v["type"], "input_audio_buffer.append");
        assert_eq!(v["audio"], b64(&[0x01, 0x02, 0x03]));
    }

    #[test]
    fn translation_namespaces_its_audio_append_type() {
        // The translation endpoint only accepts `session.`-prefixed client events; the rest use the
        // plain name. Sending the wrong one is rejected ("Supported values are: 'session.update',
        // 'session.input_audio_buffer.append', 'session.close'").
        assert_eq!(
            audio_append_type(SessionKind::Translation),
            "session.input_audio_buffer.append"
        );
        assert_eq!(
            audio_append_type(SessionKind::Transcription),
            "input_audio_buffer.append"
        );
        assert_eq!(
            audio_append_type(SessionKind::Model),
            "input_audio_buffer.append"
        );

        let json = build_audio_append(&[0xAA], audio_append_type(SessionKind::Translation));
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(v["type"], "session.input_audio_buffer.append");
    }

    #[test]
    fn parses_delta_completed_error_and_unknown_events() {
        let delta = r#"{"type":"conversation.item.input_audio_transcription.delta","delta":"Hel"}"#;
        assert_eq!(
            parse_server_event(delta, SessionKind::Transcription),
            ServerEvent::Delta("Hel".to_owned())
        );

        let done = r#"{"type":"conversation.item.input_audio_transcription.completed","transcript":"Hello."}"#;
        assert_eq!(
            parse_server_event(done, SessionKind::Transcription),
            ServerEvent::Completed("Hello.".to_owned())
        );

        let err = r#"{"type":"error","error":{"message":"bad key"}}"#;
        assert_eq!(
            parse_server_event(err, SessionKind::Transcription),
            ServerEvent::Error("bad key".to_owned())
        );

        let other = r#"{"type":"input_audio_buffer.speech_started"}"#;
        assert_eq!(
            parse_server_event(other, SessionKind::Transcription),
            ServerEvent::Other
        );

        assert_eq!(
            parse_server_event("not json", SessionKind::Transcription),
            ServerEvent::Other
        );
    }

    #[test]
    fn to_pcm16_upsamples_16k_to_24k_and_encodes_little_endian() {
        // 16 samples at 16 kHz → ~24 samples at 24 kHz → twice as many bytes.
        let samples = vec![0.0f32; 16];
        let bytes = to_pcm16(&samples, 16_000, 24_000);
        assert_eq!(bytes.len(), 24 * 2, "1.5× upsample, 2 bytes/sample");

        // Full-scale clamps to i16::MAX, little-endian.
        let hot = to_pcm16(&[1.0, -1.0], 24_000, 24_000); // same rate → no resample, 2 samples
        assert_eq!(&hot[0..2], &i16::MAX.to_le_bytes());
        assert_eq!(&hot[2..4], &(-i16::MAX).to_le_bytes());
    }

    #[test]
    fn resample_linear_is_identity_at_the_same_rate() {
        let input = vec![0.1, 0.2, 0.3];
        assert_eq!(resample_linear(&input, 16_000, 16_000), input);
        assert!(resample_linear(&[], 16_000, 24_000).is_empty());
    }

    #[test]
    fn speechmatics_start_is_raw_16k_with_language_and_operating_point() {
        use wisp_core::params::ParamValue;
        let mut params = ParamValues::from_specs(&speechmatics_param_specs());
        params.set("language", ParamValue::Text("yue".to_owned()));

        let v: serde_json::Value =
            serde_json::from_str(&speechmatics_start("enhanced", &params)).unwrap();
        assert_eq!(v["message"], "StartRecognition");
        assert_eq!(v["audio_format"]["type"], "raw");
        assert_eq!(v["audio_format"]["encoding"], "pcm_s16le");
        assert_eq!(v["audio_format"]["sample_rate"], 16_000);
        assert_eq!(v["transcription_config"]["language"], "yue");
        assert_eq!(v["transcription_config"]["operating_point"], "enhanced");
        assert_eq!(v["transcription_config"]["enable_partials"], false);
    }

    #[test]
    fn speechmatics_parse_appends_finals_and_surfaces_errors() {
        // Each AddTranscript is a final segment we append (Delta); the gap commits the line.
        assert_eq!(
            speechmatics_parse(r#"{"message":"AddTranscript","metadata":{"transcript":" Hello"}}"#),
            ServerEvent::Delta(" Hello".to_owned())
        );
        assert_eq!(
            speechmatics_parse(r#"{"message":"RecognitionStarted","id":"x"}"#),
            ServerEvent::Other
        );
        assert_eq!(
            speechmatics_parse(r#"{"message":"Error","type":"not_authorised","reason":"bad key"}"#),
            ServerEvent::Error("bad key".to_owned())
        );
    }

    #[test]
    fn streaming_param_specs_dispatches_per_protocol() {
        let openai: Vec<_> =
            streaming_param_specs(StreamingProtocol::OpenAiRealtime, "gpt-4o-transcribe")
                .into_iter()
                .map(|s| s.key)
                .collect();
        assert!(openai.iter().any(|k| k == "language"));

        let speechmatics: Vec<_> =
            streaming_param_specs(StreamingProtocol::Speechmatics, "enhanced")
                .into_iter()
                .map(|s| s.key)
                .collect();
        assert!(speechmatics.iter().any(|k| k == "language"));
        assert!(speechmatics.iter().any(|k| k == "max_delay"));
    }

    #[test]
    fn gemini_setup_audio_parse_and_auth() {
        use wisp_core::params::ParamValue;
        let mut params = ParamValues::from_specs(&gemini_param_specs());
        params.set("language", ParamValue::Text("yue".to_owned()));
        let v: serde_json::Value =
            serde_json::from_str(&gemini_setup("gemini-3.1-flash-live-preview", &params)).unwrap();
        assert_eq!(v["setup"]["model"], "models/gemini-3.1-flash-live-preview");
        // The Cantonese hint rides in a system instruction.
        assert!(v["setup"]["systemInstruction"]["parts"][0]["text"]
            .as_str()
            .unwrap()
            .contains("Cantonese"));
        assert!(v["setup"]["inputAudioTranscription"].is_object());
        assert_eq!(
            v["setup"]["generationConfig"]["responseModalities"][0],
            "AUDIO"
        );

        let a: serde_json::Value = serde_json::from_str(&gemini_audio(&[1, 2, 3, 4])).unwrap();
        assert_eq!(
            a["realtimeInput"]["audio"]["mimeType"],
            "audio/pcm;rate=16000"
        );
        assert!(a["realtimeInput"]["audio"]["data"].is_string());

        // The input transcription is surfaced as a Delta; the model's own reply (modelTurn) is ignored.
        assert_eq!(
            gemini_parse(r#"{"serverContent":{"inputTranscription":{"text":"你好"}}}"#),
            ServerEvent::Delta("你好".to_owned())
        );
        assert_eq!(
            gemini_parse(r#"{"serverContent":{"modelTurn":{"parts":[{"text":"hi"}]}}}"#),
            ServerEvent::Other
        );
        assert_eq!(
            gemini_parse(r#"{"error":{"message":"bad key"}}"#),
            ServerEvent::Error("bad key".to_owned())
        );

        // Gemini authenticates with the x-goog-api-key header (raw key), not Bearer.
        let p = Protocol::Gemini {
            model: "x".to_owned(),
        };
        let (name, value) = p.auth("KEY");
        assert_eq!(name.as_str(), "x-goog-api-key");
        assert_eq!(value, "KEY");
        assert_eq!(p.sample_rate(), 16_000);
    }

    #[test]
    fn protocol_speechmatics_uses_raw_binary_audio_and_16k() {
        let p = Protocol::Speechmatics {
            operating_point: "enhanced".to_owned(),
        };
        assert_eq!(p.sample_rate(), 16_000);
        assert_eq!(p.finalize_mode(), FinalizeMode::Gap);
        assert!(matches!(p.audio_message(&[1, 2, 3, 4]), Message::Binary(_)));

        // OpenAI frames the same audio as base64-in-JSON text instead.
        let oa = Protocol::OpenAi {
            model: "gpt-4o-transcribe".to_owned(),
            kind: SessionKind::Transcription,
        };
        assert_eq!(oa.sample_rate(), 24_000);
        assert!(matches!(oa.audio_message(&[1, 2, 3, 4]), Message::Text(_)));
    }
}
