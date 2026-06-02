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
//! Protocol (OpenAI GA Realtime, verified against `openapi.yaml`): connect to
//! `wss://api.openai.com/v1/realtime?intent=transcription` with just the bearer token (the old
//! `OpenAI-Beta: realtime=v1` header selects the retired beta shape, now rejected), send a
//! `session.update` configuring a `transcription` session (model + server VAD nested under
//! `session.audio.input`), stream `input_audio_buffer.append` frames, and read
//! `conversation.item.input_audio_transcription.delta` (partial) and `.completed` (final). Input is
//! 16-bit PCM, **24 kHz**, mono, little-endian — so we resample the pipeline's 16 kHz.

use std::io::ErrorKind;
use std::net::TcpStream;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::Duration;

use tungstenite::client::IntoClientRequest;
use tungstenite::http::{header::AUTHORIZATION, HeaderValue};
use tungstenite::stream::MaybeTlsStream;
use tungstenite::{Message, WebSocket};

use wisp_core::engine::{StreamingAsrEngine, StreamingResult};
use wisp_core::error::{Result, WispError};
use wisp_core::params::{ParamSpec, ParamValues};

use crate::b64;

/// OpenAI Realtime requires 24 kHz PCM16 input.
const REALTIME_SAMPLE_RATE: u32 = 24_000;

/// Default Realtime transcription WebSocket endpoint (overridable via [`OpenAiRealtimeEngine::REALTIME_URL_ENV`]).
const DEFAULT_REALTIME_URL: &str = "wss://api.openai.com/v1/realtime?intent=transcription";

/// Surfaces a session error the user should see (bad config, server-side error, dropped connection)
/// — wired by the app to show a notice, so a failure is never silent. Called off the audio thread.
pub type ErrorSink = Box<dyn Fn(&str) + Send>;

/// A realtime, streaming [`StreamingAsrEngine`] backed by the OpenAI Realtime WebSocket API.
///
/// Construction connects the socket and configures the transcription session (failing fast on a bad
/// key or unreachable host); a background thread then streams audio and collects transcript events.
pub struct OpenAiRealtimeEngine {
    /// PCM16 (24 kHz, LE) audio frames to send, handed to the worker.
    audio_tx: Sender<Vec<u8>>,
    /// Parsed transcript events from the worker.
    event_rx: Receiver<ServerEvent>,
    /// Signals the worker to close the socket and exit.
    stop: Arc<AtomicBool>,
    worker: Option<JoinHandle<()>>,
    /// The in-progress utterance's accumulated partial text.
    hypothesis: String,
    /// Reports errors to the app so they surface in the UI instead of failing silently.
    on_error: ErrorSink,
}

impl OpenAiRealtimeEngine {
    /// Env var overriding the Realtime WebSocket URL — for an air-gapped deployment or an
    /// OpenAI-compatible realtime gateway. Unset uses [`DEFAULT_REALTIME_URL`].
    pub const REALTIME_URL_ENV: &'static str = "WISP_OPENAI_REALTIME_URL";

    /// The tunable knobs this engine exposes **for `model`**, each with a smart default — rendered by
    /// the generic advanced-parameters UI and read back in `new` via [`ParamValues`]. Per-model
    /// because OpenAI's transcription params differ by model (`prompt` vs `delay`); a new knob is
    /// added here and needs no UI change.
    pub fn param_specs(model: &str) -> Vec<ParamSpec> {
        let mut specs = vec![ParamSpec::enumerated(
            "language",
            "Language",
            "The spoken language, or auto-detect. Setting it improves accuracy and latency.",
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
        .basic()];

        // Model-specific: gpt-realtime-whisper takes a decode `delay`; the gpt-4o-transcribe family
        // takes a biasing `prompt`. OpenAI rejects each knob on the other model.
        if model == "gpt-realtime-whisper" {
            specs.push(
                ParamSpec::enumerated(
                    "delay",
                    "Delay",
                    "How long to wait before emitting text — higher is more accurate but higher \
                     latency.",
                    &[
                        ("minimal", "Minimal"),
                        ("low", "Low"),
                        ("medium", "Medium"),
                        ("high", "High"),
                        ("xhigh", "Extra high"),
                    ],
                    "low",
                )
                .basic(),
            );
        } else {
            specs.push(
                ParamSpec::text(
                    "prompt",
                    "Hints",
                    "Names, jargon, or acronyms to bias the transcription (e.g. \"Acme, kubectl\").",
                    "",
                )
                .basic(),
            );
        }

        specs.extend([
            ParamSpec::float(
                "vad_threshold",
                "Voice sensitivity",
                "How clearly speech must stand out for the server to treat it as voice. Higher = \
                 stricter (fewer false starts in noise).",
                0.0,
                1.0,
                0.05,
                0.5,
            ),
            ParamSpec::int(
                "vad_silence_ms",
                "End-of-speech silence",
                "How long a pause (ms) ends an utterance and finalises the line. Lower feels \
                 snappier; higher avoids cutting mid-sentence.",
                100,
                2000,
                500,
            ),
            ParamSpec::int(
                "vad_prefix_padding_ms",
                "Lead-in padding",
                "Audio kept before speech starts (ms), so the first word isn't clipped.",
                0,
                1000,
                300,
            ),
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
            ),
        ]);

        specs
    }

    /// Connects and configures a Realtime transcription session for `model`, authenticating with
    /// `api_key`. `params` carries the advanced knobs — including the language — with missing keys
    /// falling back to [`Self::param_specs`] defaults. Errors on a blank key, a failed connection, or
    /// a rejected handshake.
    pub fn new(
        model: &str,
        api_key: &str,
        params: &ParamValues,
        on_error: ErrorSink,
    ) -> Result<Self> {
        let key = api_key.trim();
        if key.is_empty() {
            return Err(WispError::Engine("OpenAI needs an API key".to_owned()));
        }

        let mut ws = connect(key)?;
        eprintln!("wisp: OpenAI Realtime connected (model {model})");

        // Configure the session while still blocking, so the update is delivered before the worker
        // flips the socket to non-blocking and starts streaming audio.
        ws.send(Message::text(build_session_update(model, params)))
            .map_err(|e| engine_err("session config", e))?;
        set_nonblocking(&mut ws);

        let (audio_tx, audio_rx) = mpsc::channel::<Vec<u8>>();
        let (event_tx, event_rx) = mpsc::channel::<ServerEvent>();
        let stop = Arc::new(AtomicBool::new(false));
        let stop_worker = Arc::clone(&stop);

        let worker = std::thread::spawn(move || run_socket(ws, &audio_rx, &event_tx, &stop_worker));

        Ok(Self {
            audio_tx,
            event_rx,
            stop,
            worker: Some(worker),
            hypothesis: String::new(),
            on_error,
        })
    }

    /// Applies every transcript event received so far, returning the result to surface this call:
    /// `Some(final)` once an utterance completed (text cleared for the next), else the growing
    /// partial, or `None` when nothing has changed.
    fn drain_events(&mut self) -> Option<StreamingResult> {
        let mut final_text: Option<String> = None;
        let mut endpoint = false;

        while let Ok(event) = self.event_rx.try_recv() {
            match event {
                ServerEvent::Delta(delta) => self.hypothesis.push_str(&delta),
                ServerEvent::Completed(transcript) => {
                    final_text = Some(transcript);
                    endpoint = true;
                }
                ServerEvent::Failed(msg) | ServerEvent::Error(msg) => {
                    eprintln!("wisp: OpenAI Realtime error: {msg}");
                    (self.on_error)(&msg);
                    endpoint = true;
                }
                ServerEvent::Other => {}
            }
        }

        if let Some(text) = final_text {
            self.hypothesis.clear();
            return Some(StreamingResult {
                text,
                is_endpoint: true,
            });
        }

        if endpoint {
            // A failed/errored utterance: close it out with whatever partial we had, then reset.
            let text = std::mem::take(&mut self.hypothesis);
            return Some(StreamingResult {
                text,
                is_endpoint: true,
            });
        }

        None
    }
}

impl StreamingAsrEngine for OpenAiRealtimeEngine {
    fn accept_waveform(&mut self, sample_rate: u32, samples: &[f32]) -> StreamingResult {
        if !samples.is_empty() {
            // A send error means the worker is gone (connection dropped); we surface that through the
            // event channel, so just drop the audio here.
            let _ = self.audio_tx.send(to_pcm16(samples, sample_rate));
        }

        self.drain_events().unwrap_or_else(|| StreamingResult {
            text: self.hypothesis.clone(),
            is_endpoint: false,
        })
    }

    fn reset(&mut self) {
        self.hypothesis.clear();
    }
}

impl Drop for OpenAiRealtimeEngine {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

/// One transcript event from the Realtime socket, mapped from the server event stream.
#[derive(Debug, Clone, PartialEq, Eq)]
enum ServerEvent {
    /// An incremental hypothesis update for the in-progress utterance.
    Delta(String),
    /// The final transcript for a completed utterance (authoritative, punctuated).
    Completed(String),
    /// Transcription failed for an item.
    Failed(String),
    /// A session-level error (bad key/model, quota, …).
    Error(String),
    /// Any other event we don't act on (session lifecycle, VAD markers, …).
    Other,
}

/// Opens the Realtime WebSocket, authenticating with `key`. Fails fast on connect/handshake errors.
fn connect(key: &str) -> Result<WebSocket<MaybeTlsStream<TcpStream>>> {
    let url = std::env::var(OpenAiRealtimeEngine::REALTIME_URL_ENV)
        .unwrap_or_else(|_| DEFAULT_REALTIME_URL.to_owned());

    let mut request = url
        .as_str()
        .into_client_request()
        .map_err(|e| engine_err("bad realtime url", e))?;

    let bearer = HeaderValue::from_str(&format!("Bearer {key}"))
        .map_err(|e| engine_err("auth header", e))?;
    // GA Realtime: just the bearer token. The old `OpenAI-Beta: realtime=v1` header selects the
    // retired beta API shape, which the server now rejects ("beta_api_shape_disabled").
    request.headers_mut().insert(AUTHORIZATION, bearer);

    let (ws, _response) = tungstenite::connect(request).map_err(|e| engine_err("connect", e))?;
    Ok(ws)
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

/// The background loop owning the socket: send queued audio, drain incoming transcript events, until
/// asked to stop or the connection drops.
fn run_socket(
    mut ws: WebSocket<MaybeTlsStream<TcpStream>>,
    audio_rx: &Receiver<Vec<u8>>,
    event_tx: &Sender<ServerEvent>,
    stop: &AtomicBool,
) {
    while !stop.load(Ordering::Relaxed) {
        while let Ok(pcm) = audio_rx.try_recv() {
            if let Err(e) = ws.send(Message::text(build_audio_append(&pcm))) {
                if !is_would_block(&e) {
                    let _ = event_tx.send(ServerEvent::Error(format!("send failed: {e}")));
                    return;
                }
            }
        }

        loop {
            match ws.read() {
                Ok(Message::Text(text)) => {
                    // Diagnostic: surface exactly what the server sends (truncated) to the dev log,
                    // so a "connected but nothing transcribed" session is debuggable.
                    let snippet: String = text.chars().take(220).collect();
                    eprintln!("wisp realtime ◀ {snippet}");
                    let _ = event_tx.send(parse_server_event(text.as_str()));
                }
                Ok(Message::Close(frame)) => {
                    eprintln!("wisp: OpenAI Realtime closed: {frame:?}");
                    let _ = event_tx.send(ServerEvent::Error("connection closed".to_owned()));
                    return;
                }
                Ok(_) => {} // ping/pong/binary — ignore
                Err(e) if is_would_block(&e) => break,
                Err(e) => {
                    let _ = event_tx.send(ServerEvent::Error(format!("read failed: {e}")));
                    return;
                }
            }
        }

        std::thread::sleep(Duration::from_millis(5));
    }

    let _ = ws.close(None);
    let _ = ws.flush();
}

/// Whether a tungstenite error is just "no data yet" on the non-blocking socket (not fatal).
fn is_would_block(e: &tungstenite::Error) -> bool {
    matches!(e, tungstenite::Error::Io(io) if io.kind() == ErrorKind::WouldBlock)
}

/// The GA `session.update` payload for a transcription session: 24 kHz PCM input, the chosen model,
/// server-VAD endpointing tuned by `params`, and (unless `off`) server-side noise reduction. GA
/// nests everything under `session.audio.input`. Language, prompt, and delay all ride in `params`
/// and are sent only when set (each is model-specific — see [`Self::param_specs`]).
fn build_session_update(model: &str, params: &ParamValues) -> String {
    let mut transcription = serde_json::json!({ "model": model });
    for key in ["language", "prompt", "delay"] {
        let value = params.text(key, "");
        if !value.is_empty() {
            transcription[key] = serde_json::json!(value);
        }
    }

    let mut input = serde_json::json!({
        "format": { "type": "audio/pcm", "rate": REALTIME_SAMPLE_RATE },
        "transcription": transcription,
        "turn_detection": {
            "type": "server_vad",
            "threshold": params.float("vad_threshold", 0.5),
            "prefix_padding_ms": params.int("vad_prefix_padding_ms", 300),
            "silence_duration_ms": params.int("vad_silence_ms", 500)
        }
    });

    let noise = params.text("noise_reduction", "near_field");
    if noise != "off" {
        input["noise_reduction"] = serde_json::json!({ "type": noise });
    }

    serde_json::json!({
        "type": "session.update",
        "session": { "type": "transcription", "audio": { "input": input } }
    })
    .to_string()
}

/// An `input_audio_buffer.append` frame carrying base64 PCM16 audio.
fn build_audio_append(pcm16_le: &[u8]) -> String {
    serde_json::json!({
        "type": "input_audio_buffer.append",
        "audio": b64(pcm16_le),
    })
    .to_string()
}

/// Maps a server event JSON to a [`ServerEvent`]; anything unrecognised is [`ServerEvent::Other`].
fn parse_server_event(json: &str) -> ServerEvent {
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
        "conversation.item.input_audio_transcription.delta" => ServerEvent::Delta(event.delta),
        "conversation.item.input_audio_transcription.completed" => {
            ServerEvent::Completed(event.transcript)
        }
        "conversation.item.input_audio_transcription.failed" => ServerEvent::Failed(message()),
        "error" => ServerEvent::Error(message()),
        _ => ServerEvent::Other,
    }
}

/// Resamples `samples` from `src_rate` to 24 kHz and encodes them as little-endian PCM16 bytes —
/// the wire format the Realtime API's `pcm16` input expects.
fn to_pcm16(samples: &[f32], src_rate: u32) -> Vec<u8> {
    let resampled = resample_linear(samples, src_rate, REALTIME_SAMPLE_RATE);

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
        assert_eq!(
            OpenAiRealtimeEngine::REALTIME_URL_ENV,
            "WISP_OPENAI_REALTIME_URL"
        );
    }

    #[test]
    fn session_update_uses_param_defaults_for_model_vad_and_noise() {
        let params =
            ParamValues::from_specs(&OpenAiRealtimeEngine::param_specs("gpt-4o-transcribe"));
        let json = build_session_update("gpt-4o-transcribe", &params);
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

        let mut params =
            ParamValues::from_specs(&OpenAiRealtimeEngine::param_specs("gpt-4o-transcribe"));
        params.set("language", ParamValue::Text("yue".to_owned()));
        params.set("prompt", ParamValue::Text("kubectl".to_owned()));
        params.set("vad_threshold", ParamValue::Float(0.8));
        params.set("noise_reduction", ParamValue::Text("off".to_owned()));

        let json = build_session_update("gpt-4o-transcribe", &params);
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
    fn param_specs_are_per_model() {
        let keys = |model: &str| {
            OpenAiRealtimeEngine::param_specs(model)
                .into_iter()
                .map(|s| s.key)
                .collect::<Vec<_>>()
        };

        let transcribe = keys("gpt-4o-transcribe");
        assert!(transcribe.iter().any(|k| k == "prompt"), "4o has a prompt");
        assert!(!transcribe.iter().any(|k| k == "delay"), "4o has no delay");

        let whisper = keys("gpt-realtime-whisper");
        assert!(whisper.iter().any(|k| k == "delay"), "whisper has a delay");
        assert!(
            !whisper.iter().any(|k| k == "prompt"),
            "whisper has no prompt"
        );

        // Language and the VAD knobs are common to every model.
        for model in ["gpt-4o-transcribe", "gpt-realtime-whisper"] {
            assert!(keys(model).iter().any(|k| k == "language"));
            assert!(keys(model).iter().any(|k| k == "vad_threshold"));
        }
    }

    #[test]
    fn audio_append_carries_base64_pcm16() {
        let json = build_audio_append(&[0x01, 0x02, 0x03]);
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();

        assert_eq!(v["type"], "input_audio_buffer.append");
        assert_eq!(v["audio"], b64(&[0x01, 0x02, 0x03]));
    }

    #[test]
    fn parses_delta_completed_error_and_unknown_events() {
        let delta = r#"{"type":"conversation.item.input_audio_transcription.delta","delta":"Hel"}"#;
        assert_eq!(
            parse_server_event(delta),
            ServerEvent::Delta("Hel".to_owned())
        );

        let done = r#"{"type":"conversation.item.input_audio_transcription.completed","transcript":"Hello."}"#;
        assert_eq!(
            parse_server_event(done),
            ServerEvent::Completed("Hello.".to_owned())
        );

        let err = r#"{"type":"error","error":{"message":"bad key"}}"#;
        assert_eq!(
            parse_server_event(err),
            ServerEvent::Error("bad key".to_owned())
        );

        let other = r#"{"type":"input_audio_buffer.speech_started"}"#;
        assert_eq!(parse_server_event(other), ServerEvent::Other);

        assert_eq!(parse_server_event("not json"), ServerEvent::Other);
    }

    #[test]
    fn to_pcm16_upsamples_16k_to_24k_and_encodes_little_endian() {
        // 16 samples at 16 kHz → ~24 samples at 24 kHz → twice as many bytes.
        let samples = vec![0.0f32; 16];
        let bytes = to_pcm16(&samples, 16_000);
        assert_eq!(bytes.len(), 24 * 2, "1.5× upsample, 2 bytes/sample");

        // Full-scale clamps to i16::MAX, little-endian.
        let hot = to_pcm16(&[1.0, -1.0], 24_000); // same rate → no resample, 2 samples
        assert_eq!(&hot[0..2], &i16::MAX.to_le_bytes());
        assert_eq!(&hot[2..4], &(-i16::MAX).to_le_bytes());
    }

    #[test]
    fn resample_linear_is_identity_at_the_same_rate() {
        let input = vec![0.1, 0.2, 0.3];
        assert_eq!(resample_linear(&input, 16_000, 16_000), input);
        assert!(resample_linear(&[], 16_000, 24_000).is_empty());
    }
}
