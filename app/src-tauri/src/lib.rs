//! Wisp desktop app — a thin Tauri shell over `wisp_pipeline::Session`.
//!
//! Commands: list/download/select models; list input devices and choose mic + system sources;
//! start/stop transcription. A session is spawned per audio source (microphone = "Me", system
//! loopback = meeting participants), all forwarding transcript segments to the webview.

use std::collections::{HashMap, VecDeque};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use tauri::{path::BaseDirectory, AppHandle, Emitter, Manager, State};

#[cfg(target_os = "macos")]
use wisp_aec::WebrtcEchoCanceller;
use wisp_audio::{
    normalize_for_asr_in_place, tee, to_mono_16k, ChannelSource, EchoCancellingSource, MediaSource,
    MeetingMixer, MicSource, MutedSource, Resampler, RnnoiseDenoiser, Tee, FRAME_CHUNK_MS,
    TARGET_SAMPLE_RATE,
};
use wisp_core::aec::{EchoCanceller, PassthroughEchoCanceller};
use wisp_core::audio::{AudioFrame, AudioSource, AudioSourceInfo};
use wisp_core::channel::FrameReceiver;
use wisp_core::cloud::{CloudAuth, CloudModel, CloudProtocol, CloudProvider, StreamingProtocol};
use wisp_core::dedup::CrossStreamEchoFilter;
use wisp_core::denoise::Denoiser;
use wisp_core::diarize::{attribute_speakers_by_word, ClipDiarizer, SpeakerSpan};
use wisp_core::engine::{AsrEngine, ClipOptions, StreamingAsrEngine};
use wisp_core::error::{Result as WispResult, WispError};
use wisp_core::export::{format_markdown, format_transcript, ExportFormat, MeetingMeta};
use wisp_core::model::{ModelDescriptor, ModelFamily, ModelFile, ModelId, ModelStore, Quant};
use wisp_core::params::{ParamKind, ParamSpec, ParamValue, ParamValues};
use wisp_core::task::run_within;
use wisp_core::transcript::{AudioSourceKind, SegmentStatus, TranscriptEvent, TranscriptSegment};
use wisp_engine_cloud::{
    assist_param_specs, assist_realtime_param_specs, batch_param_specs as cloud_batch_param_specs,
    build_assist_engine, build_realtime_engine, chat_completion, chat_completion_stream,
    streaming_param_specs as cloud_streaming_param_specs, ChatRequest, CloudEngine,
};
use wisp_engine_sherpa::{
    GtcrnDenoiser, ParaformerEngine, ParakeetEngine, SenseVoiceEngine, SherpaDiarizer,
    SherpaLiveDiarizer, SileroSegmenter, StreamingTransducerEngine, WhisperEngine,
};
use wisp_library::{Library, Note, NoteSummary, SearchHit, Segment};
#[cfg(target_os = "windows")]
use wisp_loopback::WasapiLoopbackSource;
use wisp_models::{
    builtin_catalog, cloud_catalog, coreml_asset, denoise_models, diarization_models,
    family_runnable, model_fit, recommended_accurate_model, recommended_default_model, Accelerator,
    FsModelStore, GpuTier, HttpDownloader, MachineProfile, ModelFit,
};
use wisp_pipeline::{
    remap_to_original, transcribe_in_windows, EnergySegmenter, EnergyVad, GatedClip, LiveStream,
    Segmenter, Session, Transcriber, Vad, DEFAULT_SILENCE_HANGOVER,
};
#[cfg(target_os = "macos")]
use wisp_screencapture::ScreenCaptureSource;

mod dictation;
mod permissions;

/// Event channel the UI listens on for transcript segments.
const SEGMENT_EVENT: &str = "transcript://segment";

/// Event channel the UI listens on for a live cloud-streaming error (so a failure isn't silent).
const LIVE_ERROR_EVENT: &str = "live://error";

/// Event channel the UI listens on for realtime AI-assist responses — one finalised reply per turn,
/// payload is the response text. A reply that streamed in via [`ASSIST_DELTA_EVENT`] is closed by this.
const ASSIST_TEXT_EVENT: &str = "assist://text";

/// Event channel for an incremental chunk of the in-progress assist reply (payload is the new text to
/// append) — so a reply streams into the feed as it generates instead of popping in whole at the end.
const ASSIST_DELTA_EVENT: &str = "assist://delta";

/// Event channel for a realtime AI-assist error (bad key/model, server error, dropped socket) — kept
/// separate from [`LIVE_ERROR_EVENT`] so the assist pane shows its own failures, not the transcript's.
const ASSIST_ERROR_EVENT: &str = "assist://error";

/// Event channel the UI listens on for model-download progress.
const DOWNLOAD_PROGRESS_EVENT: &str = "download://progress";

/// Event channels for file transcription: total duration up front, decode progress (0–100), each
/// segment as it's produced, and a completion signal.
const FILE_META_EVENT: &str = "file://meta";
/// The current pipeline phase (`"decoding"`, `"reducing noise"`, `"transcribing"`), so the UI can
/// say what's happening even during phases that report no percentage (decode, denoise, and engines
/// without a native progress callback).
const FILE_STAGE_EVENT: &str = "file://stage";
const FILE_PROGRESS_EVENT: &str = "file://progress";
const FILE_SEGMENT_EVENT: &str = "file://segment";
const FILE_DONE_EVENT: &str = "file://done";

/// Sentinel "device" id selecting one-click system-audio capture (ScreenCaptureKit, no setup).
/// Must match the value used by the UI.
const SYSTEM_CAPTURE_ID: &str = "__wisp_system_audio__";

/// Sentinel mic id meaning "no microphone" — capture system audio only. Matches the UI.
const MIC_OFF_ID: &str = "__wisp_mic_off__";

/// Voice-activity RMS gate for the microphone stream, slightly above the clean system stream's
/// default. Kept modest so the quieter, trailing ends of real speech aren't gated as silence (which
/// would cut sentences short); residual echo/noise is handled by AEC + the cross-stream dedup.
const MIC_VAD_THRESHOLD: f32 = 0.012;

/// Shared application state.
struct AppState {
    store: Arc<FsModelStore>,
    sessions: Mutex<Vec<Session>>,
    /// The push-to-talk dictation session, while the hotkey is held; `None` otherwise.
    dictation: Mutex<Option<dictation::Dictation>>,
    /// The configured dictation hotkey, and whether it's currently registered.
    dictation_hotkey: Mutex<String>,
    dictation_enabled: Mutex<bool>,
    active: Mutex<Option<ModelId>>,
    mic_device: Mutex<Option<String>>,
    system_device: Mutex<Option<String>>,
    /// Transcription language code (`zh`/`yue`/…); empty = auto-detect.
    language: Mutex<String>,
    /// Denoiser id for the live capture (`"rnnoise"`, `"denoise-gtcrn"`), or `None` for off.
    live_denoiser: Mutex<Option<String>>,
    /// Speaker-ID model id for live diarization, or `None` to skip live speaker labels.
    live_diarize_model: Mutex<Option<String>>,
    /// Live biasing prompt (names, jargon); empty for none.
    live_prompt: Mutex<String>,
    /// Live decoding: beam search (accurate) vs greedy (fast).
    live_accurate: Mutex<bool>,
    /// The system-audio fan-out, present only while mic + system run together (AEC active).
    tee: Mutex<Option<Tee>>,
    /// Fan-out tees feeding the realtime AI assist its copy of each transcription stream — kept alive
    /// for the session's duration (dropping them closes the assist branches).
    assist_tees: Mutex<Vec<Tee>>,
    /// The mixed live audio (mic + system) for the realtime assist, available while a live session
    /// runs; the assist's realtime engine consumes it when started. `None` between sessions.
    assist_audio: Mutex<Option<Box<dyn AudioSource>>>,
    /// The running realtime AI-assist worker (its stop flag + thread), or `None` when assist is idle.
    /// The worker owns the mix source while running and hands it back on stop, so assist can restart.
    assist_worker: Mutex<Option<AssistWorker>>,
    /// While the realtime assist runs, the channel the live session pushes each diarized final into, so
    /// the assist injects authoritative, speaker-attributed text alongside the audio it hears. `None`
    /// when no assist is running (the live sink then skips the routing).
    assist_finals_tx: Mutex<Option<std::sync::mpsc::Sender<String>>>,
    /// Segments from the most recent file transcription, kept for export.
    file_segments: Mutex<Vec<TranscriptSegment>>,
    /// Set by `cancel_file_transcription` to stop the running file transcription at the next window
    /// boundary; reset at the start of each `transcribe_file`.
    file_cancel: Arc<AtomicBool>,
    /// True while a file transcription runs, so a second `transcribe_file` is rejected rather than
    /// racing it (both would clobber `file_segments`). Cleared by an RAII guard on every return.
    file_busy: Arc<AtomicBool>,
    /// Committed finals from the current/most-recent live session (both mic and system streams),
    /// retained so the meeting can be exported after it ends. Cleared when a new session starts.
    live_segments: Mutex<Vec<TranscriptSegment>>,
    /// The on-disk meeting knowledge base (SQLite). Finished meetings are saved, listed, and searched
    /// here; a single connection behind a mutex (a personal library has no concurrency needs).
    library: Mutex<Library>,
    /// Per-stream live mute flags shared with the running capture (via `MutedSource`), so the Live bar
    /// can mute/unmute "You" (mic) and "Them" (system) mid-session. Reset to unmuted on each start.
    mic_muted: Arc<AtomicBool>,
    system_muted: Arc<AtomicBool>,
    /// File where the active model id is persisted, so the choice survives a restart.
    active_model_path: PathBuf,
    /// Per-provider cloud API keys, kept purely on-device (persisted to `cloud_keys_path`).
    cloud_keys: Mutex<HashMap<String, String>>,
    /// File the cloud API keys persist to — local app data only, never synced or sent anywhere
    /// except as the auth header to the provider the key belongs to.
    cloud_keys_path: PathBuf,
    /// User-added custom cloud model ids (provider + wire id), so a just-released model is usable
    /// without an app update. Persisted to `cloud_custom_models_path`.
    cloud_custom_models: Mutex<Vec<CloudCustomModel>>,
    /// File the custom cloud model ids persist to — local app data only.
    cloud_custom_models_path: PathBuf,
    /// User-defined OpenAI-compatible cloud endpoints (base URL + protocol + model). Persisted to
    /// `cloud_custom_endpoints_path`; each endpoint's API key lives in `cloud_keys` under its id.
    cloud_custom_endpoints: Mutex<Vec<CustomCloudEndpoint>>,
    /// File the custom endpoints persist to — local app data only.
    cloud_custom_endpoints_path: PathBuf,
    /// User-imported custom models (their files live under `custom_models_dir/<id>/`).
    custom_models: Mutex<Vec<CustomModel>>,
    /// Directory holding each imported model's files, and the registry JSON beside it.
    custom_models_dir: PathBuf,
}

/// Serializable transcript segment delivered to the UI.
#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct SegmentDto {
    id: u64,
    text: String,
    start_ms: u64,
    end_ms: u64,
    source: String,
    speaker: Option<u32>,
    is_final: bool,
    /// A parallel rendering of this segment (a cloud model session's translation), when present.
    aux_text: Option<String>,
}

impl From<&TranscriptSegment> for SegmentDto {
    fn from(segment: &TranscriptSegment) -> Self {
        Self {
            id: segment.id,
            text: segment.text.clone(),
            start_ms: segment.start.as_millis() as u64,
            end_ms: segment.end.as_millis() as u64,
            source: format!("{:?}", segment.source),
            speaker: segment.speaker.map(|s| s.0),
            is_final: matches!(segment.status, SegmentStatus::Final),
            aux_text: segment.aux_text.clone(),
        }
    }
}

/// Download progress for a model, streamed to the UI.
#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct DownloadProgressDto {
    id: String,
    downloaded: u64,
    total: u64,
}

/// Metadata emitted at the start of a file transcription (for the progress bar).
#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct FileMetaDto {
    name: String,
    total_ms: u64,
}

/// A catalog model with its install/active status, for the picker UI.
#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct ModelInfoDto {
    id: String,
    name: String,
    size_bytes: u64,
    languages: Vec<String>,
    description: String,
    installed: bool,
    active: bool,
    /// The engine family (`"Whisper"`, `"WhisperCpp"`, `"SenseVoice"`, `"StreamingTransducer"`), for
    /// the picker to curate — e.g. de-emphasise CPU-ONNX Whisper on a Mac that has the Metal one.
    family: String,
    /// Recommended for this machine's **Live** (real-time) use.
    recommended_live: bool,
    /// Recommended for this machine's **File** (accuracy-first) use.
    recommended_file: bool,
    /// Whether this model has an optional Core ML (Neural Engine) encoder to download.
    coreml_available: bool,
    /// Whether that Core ML encoder is already downloaded next to the model.
    coreml_installed: bool,
    /// Download size of the Core ML encoder, for its progress bar.
    coreml_size_bytes: u64,
    /// How this model fits the host: `"ready"`, `"heavy"` (runs but large for the RAM), or `"blocked"`
    /// (this OS/machine can't run it). The picker greys out `"blocked"` and hints on `"heavy"`.
    fit: String,
    /// The reason behind a `"heavy"` or `"blocked"` fit (e.g. "Needs macOS 26"); `None` when ready.
    fit_reason: Option<String>,
    /// Whether this model has downloaded files on disk that can be deleted to reclaim space — true only
    /// for a catalog model whose files are present. False for OS-provided models (no files of ours) and
    /// before download. Drives the picker's delete affordance.
    deletable: bool,
}

/// Per-file transcription options sent from the UI as one object.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct FileTranscribeOptions {
    /// Emit per-segment timestamps (for SRT/VTT export).
    timestamps: bool,
    /// Beam search (accurate) vs greedy (fast) decoding.
    accurate: bool,
    /// Context primer (names, jargon) biasing the decoder's spelling; empty for none.
    prompt: String,
    /// Speaker-ID model id to diarize with, or `None` to skip speaker labelling.
    diarize_model: Option<String>,
    /// Drop non-speech (silence/music) before decoding so the engine can't hallucinate in it.
    gate_speech: bool,
    /// Denoiser id (`"rnnoise"`, …) to clean the audio before ASR, or `None` to leave it untouched.
    denoiser: Option<String>,
    /// Engine: `None`/`"local"` uses the active on-device model; `"cloud"` uses the provider/model
    /// below (with the API key read from local storage).
    engine: Option<String>,
    /// Cloud provider id, when `engine` is `"cloud"`.
    cloud_provider: Option<String>,
    /// Cloud model id, when `engine` is `"cloud"`.
    cloud_model: Option<String>,
    /// Advanced cloud parameter values (vendor knobs like temperature) from the settings panel, keyed
    /// by spec key. Absent/empty for on-device engines.
    #[serde(default)]
    params: HashMap<String, serde_json::Value>,
}

/// Builds the ASR engine for a downloaded model. `language` is a code (`zh`/`yue`/`en`/…) or empty
/// for auto-detection.
fn build_engine(
    descriptor: &ModelDescriptor,
    dir: &Path,
    language: &str,
) -> WispResult<Box<dyn AsrEngine>> {
    match descriptor.family {
        ModelFamily::SenseVoice => {
            let model_name = descriptor
                .files
                .iter()
                .find(|f| f.name.ends_with(".onnx"))
                .map(|f| f.name.clone())
                .ok_or_else(|| WispError::Model("model descriptor has no .onnx file".to_owned()))?;
            let engine =
                SenseVoiceEngine::new(&dir.join(model_name), &dir.join("tokens.txt"), language)?;
            Ok(Box::new(engine))
        }
        ModelFamily::Whisper => {
            let onnx = |needle: &str| {
                descriptor
                    .files
                    .iter()
                    .find(|f| f.name.contains(needle) && f.name.ends_with(".onnx"))
                    .map(|f| dir.join(&f.name))
            };
            let encoder = onnx("encoder")
                .ok_or_else(|| WispError::Model("whisper model has no encoder".to_owned()))?;
            let decoder = onnx("decoder")
                .ok_or_else(|| WispError::Model("whisper model has no decoder".to_owned()))?;
            let tokens = descriptor
                .files
                .iter()
                .find(|f| f.name.ends_with("tokens.txt"))
                .map(|f| dir.join(&f.name))
                .ok_or_else(|| WispError::Model("whisper model has no tokens file".to_owned()))?;
            // Cantonese (yue) tokens exist only in Whisper large-v3; smaller sizes fall back to zh.
            let language = if language == "yue" && !descriptor.id.as_str().contains("large-v3") {
                "zh"
            } else {
                language
            };
            let engine = WhisperEngine::new(&encoder, &decoder, &tokens, language)?;
            Ok(Box::new(engine))
        }
        ModelFamily::WhisperCpp => build_whisper_cpp_engine(descriptor, dir, language),
        ModelFamily::Paraformer => {
            let model_name = descriptor
                .files
                .iter()
                .find(|f| f.name.ends_with(".onnx"))
                .map(|f| f.name.clone())
                .ok_or_else(|| WispError::Model("paraformer model has no .onnx file".to_owned()))?;
            let engine = ParaformerEngine::new(&dir.join(model_name), &dir.join("tokens.txt"))?;
            Ok(Box::new(engine))
        }
        ModelFamily::Parakeet => {
            let onnx = |needle: &str| {
                descriptor
                    .files
                    .iter()
                    .find(|f| f.name.contains(needle) && f.name.ends_with(".onnx"))
                    .map(|f| dir.join(&f.name))
            };
            let encoder = onnx("encoder")
                .ok_or_else(|| WispError::Model("parakeet model has no encoder".to_owned()))?;
            let decoder = onnx("decoder")
                .ok_or_else(|| WispError::Model("parakeet model has no decoder".to_owned()))?;
            let joiner = onnx("joiner")
                .ok_or_else(|| WispError::Model("parakeet model has no joiner".to_owned()))?;
            let engine = ParakeetEngine::new(&encoder, &decoder, &joiner, &dir.join("tokens.txt"))?;
            Ok(Box::new(engine))
        }
        other => Err(WispError::Engine(format!(
            "no engine for model family {other:?} yet"
        ))),
    }
}

/// Builds the streaming (online) transducer engine from a downloaded model — encoder/decoder/joiner
/// ONNX + tokens. Cross-platform CPU; the thread count scales to the machine inside the engine.
fn build_streaming_engine(
    descriptor: &ModelDescriptor,
    dir: &Path,
) -> WispResult<Box<dyn StreamingAsrEngine>> {
    let onnx = |needle: &str| {
        descriptor
            .files
            .iter()
            .find(|f| f.name.contains(needle) && f.name.ends_with(".onnx"))
            .map(|f| dir.join(&f.name))
    };
    let encoder = onnx("encoder")
        .ok_or_else(|| WispError::Model("streaming model has no encoder".to_owned()))?;
    let decoder = onnx("decoder")
        .ok_or_else(|| WispError::Model("streaming model has no decoder".to_owned()))?;
    let joiner = onnx("joiner")
        .ok_or_else(|| WispError::Model("streaming model has no joiner".to_owned()))?;
    let tokens = descriptor
        .files
        .iter()
        .find(|f| f.name.ends_with("tokens.txt"))
        .map(|f| dir.join(&f.name))
        .ok_or_else(|| WispError::Model("streaming model has no tokens file".to_owned()))?;
    let engine = StreamingTransducerEngine::new(&encoder, &decoder, &joiner, &tokens)?;
    Ok(Box::new(engine))
}

/// Which engine a live session runs: an on-device model, a cloud realtime-streaming provider, or a
/// cloud batch model driven segment-by-segment. Resolved once when a session starts and shared by
/// every captured stream (mic + system).
enum LiveEngine {
    Local {
        descriptor: ModelDescriptor,
        dir: PathBuf,
    },
    CloudStreaming {
        streaming: StreamingProtocol,
        model: String,
        key: String,
        params: ParamValues,
    },
    /// A batch model (built-in or a custom endpoint) run live by segment-batch: the local VAD cuts
    /// each utterance and the cloud transcribes it whole — finalize-only, no mid-sentence partials.
    CloudBatch {
        provider: CloudProvider,
        model: String,
        key: String,
        params: ParamValues,
    },
}

/// The advanced parameter specs a cloud `provider` exposes for live streaming, or none if it can't
/// stream. Dispatches to the engine crate's per-protocol specs.
fn streaming_param_specs(provider: &CloudProvider, model: &str) -> Vec<ParamSpec> {
    match provider.streaming {
        Some(sp) => cloud_streaming_param_specs(sp, model),
        None => vec![],
    }
}

/// The advanced parameter specs a cloud `provider` exposes for File (batch) transcription. Dispatches
/// to the engine crate's per-protocol specs.
fn batch_param_specs(provider: &CloudProvider, model: &str) -> Vec<ParamSpec> {
    cloud_batch_param_specs(provider.protocol, model)
}

/// Builds a [`ParamValues`] from the user's raw JSON `overrides`, coercing each to its spec's kind.
///
/// Only params the user actually set are included — spec defaults are **not** seeded — so an unset
/// param is omitted from the provider request and falls through to that model's own (optimal) default
/// rather than a fabricated value the model might reject. Unknown keys and type mismatches are
/// dropped, so a stale or malformed override never breaks a session.
fn build_param_values(
    specs: &[ParamSpec],
    overrides: &HashMap<String, serde_json::Value>,
) -> ParamValues {
    let mut values = ParamValues::new();

    for spec in specs {
        if let Some(value) = overrides
            .get(&spec.key)
            .and_then(|raw| coerce_param(raw, &spec.kind))
        {
            values.set(&spec.key, value);
        }
    }
    values
}

/// Coerces a raw JSON override to the [`ParamValue`] its [`ParamKind`] expects, or `None` on a type
/// mismatch (so the default stands).
fn coerce_param(raw: &serde_json::Value, kind: &ParamKind) -> Option<ParamValue> {
    match kind {
        ParamKind::Float { .. } => raw.as_f64().map(ParamValue::Float),
        ParamKind::Int { .. } => raw.as_i64().map(ParamValue::Int),
        ParamKind::Bool => raw.as_bool().map(ParamValue::Bool),
        ParamKind::Enum(_) | ParamKind::Text | ParamKind::TextArea => {
            raw.as_str().map(|s| ParamValue::Text(s.to_owned()))
        }
    }
}

/// Builds the GPU whisper.cpp engine from a downloaded GGUF model — Metal on macOS, Vulkan on Windows
/// (under the `whisper-vulkan` feature). Where the engine isn't built, the stub below reports it.
#[cfg(any(
    target_os = "macos",
    all(target_os = "windows", feature = "whisper-vulkan")
))]
fn build_whisper_cpp_engine(
    descriptor: &ModelDescriptor,
    dir: &Path,
    language: &str,
) -> WispResult<Box<dyn AsrEngine>> {
    let model = descriptor
        .files
        .iter()
        .find(|f| f.name.ends_with(".bin"))
        .map(|f| dir.join(&f.name))
        .ok_or_else(|| WispError::Model("whisper.cpp model has no .bin file".to_owned()))?;
    let engine = wisp_engine_whisper_cpp::WhisperCppEngine::new(&model, language)?;
    Ok(Box::new(engine))
}

#[cfg(not(any(
    target_os = "macos",
    all(target_os = "windows", feature = "whisper-vulkan")
)))]
fn build_whisper_cpp_engine(
    _descriptor: &ModelDescriptor,
    _dir: &Path,
    _language: &str,
) -> WispResult<Box<dyn AsrEngine>> {
    Err(WispError::Engine(
        "the whisper.cpp GPU engine is only available on macOS, and on Windows GPU builds"
            .to_owned(),
    ))
}

/// Builds Apple's on-device streaming recogniser (macOS 26 SpeechAnalyzer) for the session language.
/// No download — the OS owns the model; only the BCP-47 locale is resolved here. macOS only.
#[cfg(target_os = "macos")]
fn build_apple_speech_engine(language: &str) -> WispResult<Box<dyn StreamingAsrEngine>> {
    let locale = wisp_applespeech::locale_for_language(language);
    let engine = wisp_applespeech::AppleSpeechEngine::new(&locale)?;
    Ok(Box::new(engine))
}

#[cfg(not(target_os = "macos"))]
fn build_apple_speech_engine(_language: &str) -> WispResult<Box<dyn StreamingAsrEngine>> {
    Err(WispError::Engine(
        "Apple on-device speech is only available on macOS".to_owned(),
    ))
}

/// Whether Apple's on-device recogniser can run on this host right now (macOS 26+). Off macOS, or on an
/// older macOS, it's `false` — the picker then hides the entry even though the catalog lists it.
#[cfg(target_os = "macos")]
fn apple_speech_available() -> bool {
    wisp_applespeech::is_available()
}

#[cfg(not(target_os = "macos"))]
fn apple_speech_available() -> bool {
    false
}

/// Builds the segmenter for a live session: the Silero neural VAD when its bundled model resolves,
/// otherwise the dependency-free energy gate (so capture still works if the asset is missing).
fn build_segmenter(app: &AppHandle, kind: AudioSourceKind) -> Box<dyn Segmenter> {
    if let Some(model) = silero_model_path(app) {
        match SileroSegmenter::new(&model) {
            Ok(segmenter) => return Box::new(segmenter),
            Err(e) => eprintln!("wisp: Silero VAD load failed ({e}); using energy gate"),
        }
    }

    let vad: Box<dyn Vad> = match kind {
        AudioSourceKind::Microphone => Box::new(EnergyVad::new(MIC_VAD_THRESHOLD)),
        _ => Box::new(EnergyVad::default()),
    };
    Box::new(EnergySegmenter::new(vad, DEFAULT_SILENCE_HANGOVER))
}

/// Resolves the bundled `silero_vad.onnx` resource, or `None` if it isn't present.
fn silero_model_path(app: &AppHandle) -> Option<PathBuf> {
    let path = app
        .path()
        .resolve("resources/silero_vad.onnx", BaseDirectory::Resource)
        .ok()?;
    path.exists().then_some(path)
}

/// Per-session live transcription settings, read from app state when a session starts.
struct LiveSettings {
    language: String,
    /// Denoiser id (`"rnnoise"`, `"denoise-gtcrn"`), or `None` for off.
    denoiser: Option<String>,
    /// Downloaded directory for a model-based denoiser (GTCRN); `None` for the built-in / off.
    denoise_dir: Option<PathBuf>,
    /// Downloaded diarization model directory for live speaker labels, or `None` to skip.
    diarize_dir: Option<PathBuf>,
    /// Biasing prompt (names, jargon) for the streaming engine; empty for none.
    prompt: String,
    /// Beam search (accurate) vs greedy (fast) for the streaming engine.
    accurate: bool,
}

/// The mono rate the assist mix runs at — OpenAI Realtime's native input, so the engine passes it
/// through without a second resample. Both tapped streams are converted to it before summing.
const ASSIST_MIX_RATE: u32 = 24_000;

/// The live audio for the realtime AI assist (option B): the same post-processed streams the
/// transcription gets (mic after AEC + system), summed into one mono [`ASSIST_MIX_RATE`] stream.
///
/// The two taps arrive at *different* rates (the AEC mic at 16 kHz, the raw system at the capture
/// rate, e.g. 48 kHz) and in differently-sized frames, so each is resampled to the common rate first;
/// the secondary is accumulated in a small jitter buffer and mixed sample-aligned (never decimated, no
/// rate-mismatched sum). `primary` drives the cadence (`recv`, blocking) — its drop-oldest tee means an
/// idle assist never stalls or backs up the capture.
struct MixSource {
    primary: FrameReceiver,
    primary_rs: Resampler,
    secondary: Option<FrameReceiver>,
    secondary_rs: Resampler,
    /// Resampled secondary samples waiting to be mixed (bounded; oldest dropped past the cap).
    secondary_buf: VecDeque<f32>,
    /// Soft-knee + depth-capped-ducking mixer (no clip distortion, both speakers stay intelligible).
    mixer: MeetingMixer,
    info: AudioSourceInfo,
}

impl AudioSource for MixSource {
    fn info(&self) -> AudioSourceInfo {
        self.info.clone()
    }

    fn next_frame(&mut self) -> wisp_core::error::Result<Option<AudioFrame>> {
        let Some(frame) = self.primary.recv() else {
            return Ok(None);
        };
        let timestamp = frame.timestamp;
        let mut mono = self.primary_rs.process(&frame);

        if let Some(sec) = &self.secondary {
            // Drain every buffered secondary frame (resampled to the mix rate) — never decimate it —
            // capping the jitter buffer so a slightly-faster stream can't grow it without bound.
            while let Some(other) = sec.try_recv() {
                let resampled = self.secondary_rs.process(&other);
                self.secondary_buf.extend(resampled);
            }
            let cap = ASSIST_MIX_RATE as usize / 2; // ~0.5 s
            while self.secondary_buf.len() > cap {
                self.secondary_buf.pop_front();
            }

            // Mix in time order, as many secondary samples as this frame holds — soft-limited so a loud
            // mic+system sum never clips, with light depth-capped ducking so the dominant talker stays
            // clear while the other side is never lost.
            let take = mono.len().min(self.secondary_buf.len());
            let secondary: Vec<f32> = self.secondary_buf.drain(..take).collect();
            self.mixer.mix(&mut mono, &secondary);
        }

        Ok(Some(AudioFrame::new(mono, ASSIST_MIX_RATE, 1, timestamp)))
    }
}

/// Tees a processed transcription `source` so the assist can hear the same audio: returns the source
/// for transcription (a [`ChannelSource`] over one branch) and stashes the other branch + the tee
/// handle. The tee is drop-oldest, so an unread assist branch never stalls or backs up the capture.
///
/// When `want` is false (no real-time assist armed for this session) the source is returned untouched —
/// no tee, no pump thread — so an ordinary live session's audio path is byte-for-byte the original.
fn tap_for_assist(
    source: Box<dyn AudioSource>,
    want: bool,
    branches: &mut Vec<FrameReceiver>,
    tees: &mut Vec<Tee>,
) -> Box<dyn AudioSource> {
    if !want {
        return source;
    }

    let info = source.info();
    let (handle, main_rx, assist_rx) = tee(source);

    tees.push(handle);
    branches.push(assist_rx);
    Box::new(ChannelSource::new(main_rx, info))
}

/// Wraps a live stream's source with its shared mute flag (mic = You, system = Them), so the Live bar
/// can silence it mid-session. Any non-live kind is returned untouched.
fn wrap_muted(
    app: &AppHandle,
    source: Box<dyn AudioSource>,
    kind: AudioSourceKind,
) -> Box<dyn AudioSource> {
    let state = app.state::<AppState>();
    let muted = match kind {
        AudioSourceKind::Microphone => Arc::clone(&state.mic_muted),
        AudioSourceKind::System => Arc::clone(&state.system_muted),
        _ => return source,
    };
    Box::new(MutedSource::new(source, muted))
}

/// Mutes/unmutes a live stream — `"mic"` (You) or `"system"` (Them) — mid-session. The capture keeps
/// running (so an echo far-end reference stays live) but the muted stream is silenced, so it stops
/// producing transcription until unmuted.
#[tauri::command]
fn set_stream_muted(state: State<'_, AppState>, kind: String, muted: bool) -> Result<(), String> {
    let flag = match kind.as_str() {
        "mic" => &state.mic_muted,
        "system" => &state.system_muted,
        other => return Err(format!("unknown stream: {other}")),
    };
    flag.store(muted, Ordering::Relaxed);
    Ok(())
}

/// Spawns one transcription session over `source`, tagging its segments with `kind` and
/// forwarding them to the webview.
///
/// When `dedup` is set (the mic + system case), every segment is routed through the shared
/// [`CrossStreamEchoFilter`] first, so a mic segment that echoes a recent meeting segment is
/// dropped rather than emitted.
fn spawn_session(
    app: &AppHandle,
    engine_source: &LiveEngine,
    source: Box<dyn AudioSource>,
    kind: AudioSourceKind,
    dedup: Option<Arc<Mutex<CrossStreamEchoFilter>>>,
    settings: &LiveSettings,
) -> WispResult<Session> {
    // Gate the stream on its live mute flag, so the Live bar can mute "You"/"Them" mid-session.
    let source = wrap_muted(app, source, kind);

    // The per-session denoiser and the event sink (cross-stream echo dedup on finals) are the same
    // regardless of engine kind, so build them once up front.
    let denoiser = settings
        .denoiser
        .as_deref()
        .and_then(|id| build_denoiser(id, settings.denoise_dir.as_deref()));

    let sink = build_live_sink(app, dedup);

    let (descriptor, dir) = match engine_source {
        // Cloud realtime self-segments and denoises server-side, so it drives the streaming pipeline
        // directly — no local segmenter, diarizer, or RNNoise (the cloud's noise_reduction param
        // handles denoise; doubling it up would hurt).
        LiveEngine::CloudStreaming {
            streaming,
            model,
            key,
            params,
        } => {
            // Surface session errors (bad key/model, server error, dropped connection) to the UI so
            // a cloud failure is never silent. Runs off the audio thread.
            let app_err = app.clone();
            let on_error: Box<dyn Fn(&str) + Send> = Box::new(move |msg: &str| {
                let _ = app_err.emit(LIVE_ERROR_EVENT, msg.to_owned());
            });

            let engine = build_realtime_engine(*streaming, model, key, params, on_error)?;
            return Ok(Session::spawn_streaming(engine, source, sink, None, kind));
        }
        // A batch model (built-in or custom endpoint) run live: the local VAD cuts each utterance
        // and the cloud transcribes it whole. CloudEngine implements AsrEngine, so it drops into the
        // same decoupled live path as local batch models — capture stays real-time while slow calls
        // drain on their own thread. Finalize-only (no mid-sentence partials); a failed call (bad
        // key/model/URL) surfaces to the UI instead of silently producing nothing.
        LiveEngine::CloudBatch {
            provider,
            model,
            key,
            params,
        } => {
            let app_err = app.clone();
            let on_error: Box<dyn Fn(&str) + Send> = Box::new(move |msg: &str| {
                let _ = app_err.emit(LIVE_ERROR_EVENT, msg.to_owned());
            });

            let engine =
                CloudEngine::new(provider, model, key, &settings.language, params.clone())?
                    .with_error_sink(on_error);
            let segmenter = build_segmenter(app, kind);
            let transcriber = Transcriber::new(Box::new(engine), kind);

            return Ok(Session::spawn_live(
                segmenter,
                transcriber,
                source,
                sink,
                denoiser,
            ));
        }
        LiveEngine::Local { descriptor, dir } => (descriptor, dir),
    };

    // A streaming transducer self-segments and decodes cheaply per chunk, so it drives the pipeline
    // directly on one thread. Every other family is VAD-segmented and transcribed whole on the
    // decoupled live path — capture stays real-time while the slow engine drains complete utterances
    // on its own thread, so it never stalls capture or drops audio mid-sentence.
    if descriptor.family == ModelFamily::StreamingTransducer {
        let engine = build_streaming_engine(descriptor, dir)?;
        return Ok(Session::spawn_streaming(
            engine, source, sink, denoiser, kind,
        ));
    }

    // Apple's on-device recogniser self-segments and streams volatile/final results, so it drives the
    // streaming pipeline directly too — no local VAD, decode loop, or model files.
    if descriptor.family == ModelFamily::AppleSpeech {
        let engine = build_apple_speech_engine(&settings.language)?;
        return Ok(Session::spawn_streaming(
            engine, source, sink, denoiser, kind,
        ));
    }

    let transcriber = build_local_transcriber(descriptor, dir, kind, settings)?;
    let segmenter = build_segmenter(app, kind);

    Ok(Session::spawn_live(
        segmenter,
        transcriber,
        source,
        sink,
        denoiser,
    ))
}

/// Builds the event sink shared by every live path: forwards each segment to the webview, dropping
/// cross-stream echo via a shared `dedup` filter, then feeds each admitted final to the realtime
/// assist and the retained meeting transcript. Every segment — partial or final — is routed through
/// the filter, which suppresses a mic echo even while it's still provisional; the filter only
/// *remembers* committed meeting finals, so a mic line never primes itself.
fn build_live_sink(
    app: &AppHandle,
    dedup: Option<Arc<Mutex<CrossStreamEchoFilter>>>,
) -> wisp_pipeline::EventSink {
    let emitter = app.clone();
    Box::new(move |event| {
        if let TranscriptEvent::Segment(segment) = event {
            let emit = match &dedup {
                Some(filter) => filter.lock().map(|mut f| f.admit(&segment)).unwrap_or(true),
                None => true,
            };
            if emit {
                let _ = emitter.emit(SEGMENT_EVENT, SegmentDto::from(&segment));
                if matches!(segment.status, SegmentStatus::Final) {
                    route_assist_final(&emitter, &segment);
                    retain_live_segment(&emitter, &segment);
                }
            }
        }
    })
}

/// Builds the local-batch live transcriber: a configured engine wrapped in a [`Transcriber`], with
/// the per-session live diarizer attached when a speaker model is set. Shared by the single-stream
/// ([`spawn_session`]) and shared-engine ([`spawn_shared_local_session`]) local-batch paths; the
/// latter passes one transcriber across both streams, so its diarizer numbers speakers in one space.
fn build_local_transcriber(
    descriptor: &ModelDescriptor,
    dir: &Path,
    kind: AudioSourceKind,
    settings: &LiveSettings,
) -> WispResult<Transcriber> {
    let mut engine = build_engine(descriptor, dir, &settings.language)?;
    engine.configure_streaming(&settings.prompt, settings.accurate);

    let mut transcriber = Transcriber::new(engine, kind);

    // Live speaker labels via a per-session diarizer. Best-effort — if the model won't load,
    // transcribe without labels rather than failing the session.
    if let Some(diarize_dir) = &settings.diarize_dir {
        match SherpaLiveDiarizer::new(&diarize_dir.join("embedding.onnx")) {
            Ok(diarizer) => transcriber = transcriber.with_diarizer(Box::new(diarizer)),
            Err(e) => eprintln!("wisp: live diarizer load failed ({e}); skipping speaker labels"),
        }
    }

    Ok(transcriber)
}

/// Spawns ONE live session that runs several local streams (mic + system) through a SINGLE batch
/// engine — one model copy in RAM, one rolling context, one unified speaker space — instead of a full
/// engine per stream. Each stream keeps its own segmenter, denoiser, mute gate, and
/// [`AudioSourceKind`] (so segments stay labelled by source); they share the transcriber and sink.
/// Only valid for the decoupled local-batch path (see [`LiveEngine::shared_local_target`]).
fn spawn_shared_local_session(
    app: &AppHandle,
    descriptor: &ModelDescriptor,
    dir: &Path,
    streams: Vec<(Box<dyn AudioSource>, AudioSourceKind)>,
    dedup: Option<Arc<Mutex<CrossStreamEchoFilter>>>,
    settings: &LiveSettings,
) -> WispResult<Session> {
    // The shared transcriber's own kind is only the single-stream default; every utterance is tagged
    // per-stream below, so this value never reaches a segment. Use the first stream's kind.
    let default_kind = streams
        .first()
        .map(|(_, kind)| *kind)
        .unwrap_or(AudioSourceKind::Microphone);
    let transcriber = build_local_transcriber(descriptor, dir, default_kind, settings)?;

    let live_streams = streams
        .into_iter()
        .map(|(source, kind)| LiveStream {
            segmenter: build_segmenter(app, kind),
            source: wrap_muted(app, source, kind),
            denoiser: settings
                .denoiser
                .as_deref()
                .and_then(|id| build_denoiser(id, settings.denoise_dir.as_deref())),
            kind,
        })
        .collect();

    let sink = build_live_sink(app, dedup);

    Ok(Session::spawn_live_multi(live_streams, transcriber, sink))
}

impl LiveEngine {
    /// The `(descriptor, dir)` of a local batch engine that can be SHARED across multiple live streams
    /// — i.e. one that drives the decoupled VAD-segment path. Streaming transducers and Apple Speech
    /// self-segment a single audio timeline (no shared transcriber is possible), and cloud engines run
    /// per stream, so they return `None` and stay one session per stream.
    fn shared_local_target(&self) -> Option<(&ModelDescriptor, &Path)> {
        match self {
            LiveEngine::Local { descriptor, dir }
                if !matches!(
                    descriptor.family,
                    ModelFamily::StreamingTransducer | ModelFamily::AppleSpeech
                ) =>
            {
                Some((descriptor, dir))
            }
            _ => None,
        }
    }
}

/// Total physical memory in bytes, via `sysctl hw.memsize`. Falls back to 16 GiB (treated as
/// "ample") if the query fails, so model recommendation never errors out.
#[cfg(target_os = "macos")]
fn machine_ram_bytes() -> u64 {
    let mut value: u64 = 0;
    let mut size = std::mem::size_of::<u64>();
    // SAFETY: `hw.memsize` is a valid C string; `value`/`size` are correctly-sized out-pointers and
    // the remaining args are the documented null/zero for "no new value".
    let rc = unsafe {
        libc::sysctlbyname(
            c"hw.memsize".as_ptr(),
            &mut value as *mut u64 as *mut libc::c_void,
            &mut size,
            std::ptr::null_mut(),
            0,
        )
    };
    if rc == 0 && value > 0 {
        value
    } else {
        16 * 1024 * 1024 * 1024
    }
}

/// Total physical memory in bytes on Windows, via `GlobalMemoryStatusEx` (`ullTotalPhys`). Falls
/// back to 16 GiB if the call fails, so model recommendation never errors out.
#[cfg(target_os = "windows")]
fn machine_ram_bytes() -> u64 {
    use windows::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};

    // SAFETY: a zeroed MEMORYSTATUSEX with `dwLength` set is exactly the input the API documents.
    let mut status: MEMORYSTATUSEX = unsafe { std::mem::zeroed() };
    status.dwLength = std::mem::size_of::<MEMORYSTATUSEX>() as u32;

    // SAFETY: `status` is a valid, correctly-sized out-pointer for the duration of the call.
    match unsafe { GlobalMemoryStatusEx(&mut status) } {
        Ok(()) if status.ullTotalPhys > 0 => status.ullTotalPhys,
        _ => 16 * 1024 * 1024 * 1024,
    }
}

/// Total physical memory in bytes on Linux, from `/proc/meminfo`'s `MemTotal`. Falls back to 16 GiB
/// if the file can't be read or parsed.
#[cfg(target_os = "linux")]
fn machine_ram_bytes() -> u64 {
    std::fs::read_to_string("/proc/meminfo")
        .ok()
        .and_then(|info| wisp_models::parse_meminfo_total_bytes(&info))
        .unwrap_or(16 * 1024 * 1024 * 1024)
}

/// Any other platform: assume 16 GiB ("ample") so model recommendation never errors out.
#[cfg(not(any(target_os = "macos", target_os = "windows", target_os = "linux")))]
fn machine_ram_bytes() -> u64 {
    16 * 1024 * 1024 * 1024
}

/// The best ASR accelerator on this host. macOS uses the Apple GPU via Metal (whisper.cpp); other
/// platforms run ONNX on the CPU today, so they report `Cpu` until a Windows/Linux GPU engine is
/// wired (then this grows to CUDA/Vulkan/DirectML and the recommender follows automatically).
#[cfg(target_os = "macos")]
fn host_accelerator() -> Accelerator {
    Accelerator::Metal
}

#[cfg(not(target_os = "macos"))]
fn host_accelerator() -> Accelerator {
    Accelerator::Cpu
}

/// Reads a string sysctl by name (e.g. the CPU brand string), or `None` if it isn't available.
#[cfg(target_os = "macos")]
fn sysctl_string(name: &std::ffi::CStr) -> Option<String> {
    let mut size: usize = 0;
    // SAFETY: a null value buffer asks sysctl for the size it needs; `name` is a valid C string.
    let rc = unsafe {
        libc::sysctlbyname(
            name.as_ptr(),
            std::ptr::null_mut(),
            &mut size,
            std::ptr::null_mut(),
            0,
        )
    };
    if rc != 0 || size == 0 {
        return None;
    }

    let mut buf = vec![0u8; size];
    // SAFETY: `buf` holds `size` bytes; `size` is updated in place to the number actually written.
    let rc = unsafe {
        libc::sysctlbyname(
            name.as_ptr(),
            buf.as_mut_ptr() as *mut libc::c_void,
            &mut size,
            std::ptr::null_mut(),
            0,
        )
    };
    if rc != 0 {
        return None;
    }

    // `size` counts the trailing NUL — trim it before decoding.
    Some(String::from_utf8_lossy(&buf[..size.saturating_sub(1)]).into_owned())
}

/// The GPU power tier. On Apple Silicon it's the chip class parsed from the CPU brand string
/// ("Apple M2 Max" → High, "… Ultra" → Ultra, "… Pro" → Standard, plain "Apple M_" → Entry), the
/// signal that decides which Whisper this Mac can run live. Other platforms report `None` until a
/// GPU ASR engine is wired for them.
#[cfg(target_os = "macos")]
fn host_gpu_tier() -> GpuTier {
    let brand = sysctl_string(c"machdep.cpu.brand_string").unwrap_or_default();
    if brand.contains("Ultra") {
        GpuTier::Ultra
    } else if brand.contains("Max") {
        GpuTier::High
    } else if brand.contains("Pro") {
        GpuTier::Standard
    } else {
        GpuTier::Entry
    }
}

#[cfg(not(target_os = "macos"))]
fn host_gpu_tier() -> GpuTier {
    GpuTier::None
}

/// Auto-detects what this machine can run, for picking the default model — accelerator, memory, and
/// the GPU power tier so the recommendation tracks the actual chip rather than a fixed per-OS guess.
fn detect_machine() -> MachineProfile {
    MachineProfile::detailed(host_accelerator(), machine_ram_bytes(), host_gpu_tier())
}

/// Maps a catalog descriptor to its UI DTO, resolving install/active status against the store.
/// `recommended` is the machine-appropriate default; `None` for non-ASR lists that don't recommend.
/// A model is deletable when it has files of ours present on disk — i.e. a downloaded catalog model.
/// OS-provided models (no files) and not-yet-downloaded ones have nothing to reclaim.
fn model_deletable(has_files: bool, on_disk: bool) -> bool {
    has_files && on_disk
}

fn to_model_info(
    d: ModelDescriptor,
    store: &FsModelStore,
    active: Option<&ModelId>,
    live_rec: Option<&ModelId>,
    file_rec: Option<&ModelId>,
) -> ModelInfoDto {
    // An OS-provided model (no files of ours, e.g. Apple on-device speech) is always "installed" —
    // there's nothing to download, so the picker shows it ready rather than gating it behind a fetch.
    let installed = d.files.is_empty() || store.local_path(&d.id).is_some();
    let is_active = active == Some(&d.id);
    let recommended_live = live_rec == Some(&d.id);
    let recommended_file = file_rec == Some(&d.id);
    let family = format!("{:?}", d.family);
    let size_bytes = d.total_size_bytes();
    let deletable = model_deletable(!d.files.is_empty(), store.local_path(&d.id).is_some());

    let coreml = coreml_asset(&d);
    let coreml_installed = coreml
        .as_ref()
        .is_some_and(|a| store.coreml_installed(&d.id, a));
    let coreml_size_bytes = coreml.as_ref().map(|a| a.size_bytes).unwrap_or(0);

    ModelInfoDto {
        id: d.id.0,
        name: d.display_name,
        size_bytes,
        languages: d.languages,
        description: d.description,
        installed,
        active: is_active,
        family,
        recommended_live,
        recommended_file,
        coreml_available: coreml.is_some(),
        coreml_installed,
        coreml_size_bytes,
        // Defaults to a comfortable fit; the ASR picker overwrites these per machine (support-model
        // lists — diarization/denoise — are CPU-ONNX and always ready, so they keep the default).
        fit: "ready".to_owned(),
        fit_reason: None,
        deletable,
    }
}

/// Overwrites a DTO's `fit`/`fit_reason` from a [`ModelFit`] verdict — the bridge between the pure
/// machine assessment and the picker's UI fields.
fn apply_fit(dto: &mut ModelInfoDto, fit: ModelFit) {
    let (kind, reason) = match fit {
        ModelFit::Ready => ("ready", None),
        ModelFit::Heavy(reason) => ("heavy", Some(reason)),
        ModelFit::Blocked(reason) => ("blocked", Some(reason)),
    };

    dto.fit = kind.to_owned();
    dto.fit_reason = reason;

    // A model the host can't run is never "ready to start" — surface it greyed, not as installed.
    if dto.fit == "blocked" {
        dto.installed = false;
    }
}

/// The host fit for a catalog `descriptor`, layering the Apple-Speech macOS-26 runtime check (which the
/// pure assessment can't see) on top of [`model_fit`]: on a Mac too old for the API, mark it blocked.
fn assess_fit(descriptor: &ModelDescriptor, machine: &MachineProfile) -> ModelFit {
    if descriptor.family == ModelFamily::AppleSpeech
        && family_runnable(descriptor.family, machine.accelerator)
        && !apple_speech_available()
    {
        return ModelFit::Blocked("Needs macOS 26 or newer".to_owned());
    }

    model_fit(descriptor, machine)
}

#[tauri::command]
fn list_models(state: State<'_, AppState>) -> Result<Vec<ModelInfoDto>, String> {
    let active = state
        .active
        .lock()
        .map_err(|_| "state lock poisoned".to_owned())?
        .clone();
    let machine = detect_machine();
    let live_rec = recommended_default_model(&machine, &builtin_catalog());
    let file_rec = recommended_accurate_model(&machine, &builtin_catalog());
    let mut models: Vec<ModelInfoDto> = state
        .store
        .available()
        .into_iter()
        // Only transcription models belong in the ASR picker — diarization and denoise are support
        // models with their own pickers.
        .filter(|d| d.family.is_asr())
        // Don't hide what this host can't run — surface every ASR model, tagging each with how it fits
        // (ready / heavy / blocked) so the picker greys out the unrunnable ones with a reason.
        .map(|d| {
            let fit = assess_fit(&d, &machine);
            let mut dto = to_model_info(
                d,
                &state.store,
                active.as_ref(),
                Some(&live_rec),
                Some(&file_rec),
            );
            apply_fit(&mut dto, fit);
            dto
        })
        .collect();

    // Append the user's imported custom models — always installed, tagged with the same fit (a
    // Metal-only `.bin` shows greyed "Needs a macOS Metal GPU" off macOS rather than silently vanishing).
    let custom = state
        .custom_models
        .lock()
        .map_err(|_| "state lock poisoned".to_owned())?;
    for cm in custom.iter() {
        let Some(mut info) = custom_model_info(cm, active.as_ref()) else {
            continue;
        };
        if let Some(descriptor) = custom_descriptor(cm) {
            apply_fit(&mut info, assess_fit(&descriptor, &machine));
        }
        models.push(info);
    }

    Ok(models)
}

/// The diarization (speaker-ID) models, with install status — sourced separately so they never
/// appear in the ASR model picker.
#[tauri::command]
fn list_diarization_models(state: State<'_, AppState>) -> Result<Vec<ModelInfoDto>, String> {
    let models = state
        .store
        .available()
        .into_iter()
        .filter(|d| d.family == ModelFamily::Diarization)
        .map(|d| to_model_info(d, &state.store, None, None, None))
        .collect();
    Ok(models)
}

/// The downloadable denoiser models (e.g. GTCRN), with install status — for the "Reduce noise"
/// strength picker. The light built-in RNNoise needs no download and isn't listed here.
#[tauri::command]
fn list_denoise_models(state: State<'_, AppState>) -> Result<Vec<ModelInfoDto>, String> {
    let models = state
        .store
        .available()
        .into_iter()
        .filter(|d| d.family == ModelFamily::Denoise)
        .map(|d| to_model_info(d, &state.store, None, None, None))
        .collect();
    Ok(models)
}

/// A cloud transcription model, for the picker.
#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct CloudModelDto {
    id: String,
    name: String,
    streaming: bool,
    batch: bool,
    description: String,
    /// The provider's recommended default for its capability — the picker pre-selects it.
    recommended: bool,
    /// The model returns speaker labels itself — the UI hides local diarization when it's picked.
    diarizes: bool,
    /// User-added (not in the built-in catalog) — the UI tags it and offers removal.
    custom: bool,
}

/// A cloud provider with its models and whether its API key is already saved on this device.
#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct CloudProviderDto {
    id: String,
    name: String,
    key_set: bool,
    /// A masked form of the saved key (e.g. `sk-…a1b2`) for display, or `None` when no key is saved.
    /// The full key never leaves the backend; only this masked hint does.
    key_hint: Option<String>,
    /// The provider's "API keys" console page, for the "Get a key" link.
    keys_url: String,
    /// User-added custom endpoint (not a built-in catalog provider) — the UI offers edit + removal.
    custom: bool,
    /// Base URL and normalized protocol (`"openai"`/`"chat"`/`"gemini"`) — used to pre-fill the edit
    /// form for custom endpoints. Harmless for catalog providers (not secret).
    base_url: String,
    protocol: String,
    /// AI notes/assist tuning — populated for custom endpoints (so the edit form pre-fills it),
    /// default for catalog providers.
    assist: AssistParams,
    models: Vec<CloudModelDto>,
}

/// The cloud provider with `id` from the built-in catalog, if any.
fn cloud_provider_by_id(id: &str) -> Option<CloudProvider> {
    cloud_catalog().into_iter().find(|p| p.id == id)
}

/// A short, non-secret hint for a saved key — a leading `sk-`-style prefix and the last four chars
/// (e.g. `sk-…a1b2`), matching how provider consoles display stored keys. Returns `None` for a blank
/// key. Only the last four characters are revealed; the rest is never shown.
fn mask_key(key: &str) -> Option<String> {
    let key = key.trim();
    if key.is_empty() {
        return None;
    }

    let n = key.chars().count();
    let last4: String = key.chars().skip(n.saturating_sub(4)).collect();

    if n <= 8 {
        return Some(format!("…{last4}"));
    }

    let prefix: String = key.chars().take(3).collect();
    Some(format!("{prefix}…{last4}"))
}

/// Loads the on-device cloud API keys (provider → key); an absent or unreadable file yields none.
fn load_cloud_keys(path: &Path) -> HashMap<String, String> {
    fs::read_to_string(path)
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_default()
}

/// Persists `keys` to `path` (local app data only). Best-effort — a write failure is logged.
fn save_cloud_keys(path: &Path, keys: &HashMap<String, String>) {
    match serde_json::to_string_pretty(keys) {
        Ok(json) => {
            if let Err(e) = fs::write(path, json) {
                eprintln!("wisp: could not persist cloud keys: {e}");
            }
        }
        Err(e) => eprintln!("wisp: could not serialize cloud keys: {e}"),
    }
}

/// The cloud providers and their models, each flagged with whether its API key is already saved on
/// this device — for the Cloud picker and key manager.
#[tauri::command]
fn list_cloud_providers(state: State<'_, AppState>) -> Result<Vec<CloudProviderDto>, String> {
    let keys = state
        .cloud_keys
        .lock()
        .map_err(|_| "state lock poisoned".to_owned())?;
    let customs = state
        .cloud_custom_models
        .lock()
        .map_err(|_| "state lock poisoned".to_owned())?;
    let endpoints = state
        .cloud_custom_endpoints
        .lock()
        .map_err(|_| "state lock poisoned".to_owned())?;

    let mut providers: Vec<CloudProviderDto> = cloud_catalog()
        .into_iter()
        .map(|p| provider_to_dto(with_custom_models(p, &customs), &keys, &customs, false))
        .collect();
    providers.extend(endpoints.iter().map(|e| {
        let mut dto = provider_to_dto(e.to_provider(), &keys, &customs, true);
        dto.assist = e.assist.clone();
        dto
    }));
    Ok(providers)
}

/// Builds the DTO for one provider: its masked-key state plus each model, flagged custom or not.
fn provider_to_dto(
    p: CloudProvider,
    keys: &HashMap<String, String>,
    customs: &[CloudCustomModel],
    custom: bool,
) -> CloudProviderDto {
    let key_hint = keys.get(&p.id).and_then(|k| mask_key(k));
    let pid = p.id.clone();
    let protocol = match p.protocol {
        CloudProtocol::OpenAiChatAudio => "chat",
        CloudProtocol::Gemini => "gemini",
        _ => "openai",
    }
    .to_owned();
    CloudProviderDto {
        key_set: key_hint.is_some(),
        key_hint,
        keys_url: p.keys_url,
        custom,
        base_url: p.base_url,
        protocol,
        assist: AssistParams::default(),
        models: p
            .models
            .into_iter()
            .map(|m| CloudModelDto {
                custom: customs
                    .iter()
                    .any(|c| c.provider == pid && c.id.trim() == m.id),
                id: m.id,
                name: m.display_name,
                streaming: m.streaming,
                batch: m.batch,
                description: m.description,
                recommended: m.recommended,
                diarizes: m.diarizes,
            })
            .collect(),
        id: p.id,
        name: p.display_name,
    }
}

/// Stores (or, with an empty `key`, clears) the API key for `provider`, on this device only.
#[tauri::command]
fn set_cloud_key(state: State<'_, AppState>, provider: String, key: String) -> Result<(), String> {
    let mut keys = state
        .cloud_keys
        .lock()
        .map_err(|_| "state lock poisoned".to_owned())?;
    if key.trim().is_empty() {
        keys.remove(&provider);
    } else {
        keys.insert(provider, key.trim().to_owned());
    }
    save_cloud_keys(&state.cloud_keys_path, &keys);
    Ok(())
}

/// Whether an API key is saved for `provider`. Never returns the key itself.
#[tauri::command]
fn cloud_key_set(state: State<'_, AppState>, provider: String) -> Result<bool, String> {
    let keys = state
        .cloud_keys
        .lock()
        .map_err(|_| "state lock poisoned".to_owned())?;
    Ok(keys.get(&provider).is_some_and(|k| !k.trim().is_empty()))
}

/// A user-added cloud model id — a `(provider, id)` the built-in catalog doesn't list yet. Because
/// the cloud adapter routes by the provider's `protocol` (not by model id), any new id of a known
/// provider runs through the existing engine — the zero-maintenance way to use a model the day it
/// ships, with no app update. File/batch only (streaming would need a realtime client).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
struct CloudCustomModel {
    /// The built-in provider this model is reached through, e.g. `"openai"` / `"google"`.
    provider: String,
    /// The wire model id sent to the API, e.g. `"gpt-4o-transcribe-2027"`.
    id: String,
    /// Display name for the picker; falls back to the id when blank.
    name: String,
}

/// Loads the custom cloud model registry from `path`; an absent/garbage file yields none.
fn load_cloud_custom_models(path: &Path) -> Vec<CloudCustomModel> {
    fs::read_to_string(path)
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_default()
}

/// Persists the custom cloud model registry to `path` (local app data only). Best-effort.
fn save_cloud_custom_models(path: &Path, models: &[CloudCustomModel]) {
    match serde_json::to_string_pretty(models) {
        Ok(json) => {
            if let Err(e) = fs::write(path, json) {
                eprintln!("wisp: could not persist custom cloud models: {e}");
            }
        }
        Err(e) => eprintln!("wisp: could not serialize custom cloud models: {e}"),
    }
}

/// The AI notes/assist tuning a custom endpoint carries (its chat model's knobs). All optional —
/// an empty/`None` field falls back to a built-in default and is never sent to the provider. Used
/// only by [`run_llm_task`]; transcription has its own per-model parameter panel.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct AssistParams {
    /// Sampling temperature for the chat model (omitted from the request when unset, so a model that
    /// only accepts its default temperature — e.g. a reasoning model — isn't sent one).
    #[serde(default)]
    temperature: Option<f64>,
    /// Cap on the reply length, in tokens (omitted from the request when unset).
    #[serde(default)]
    max_tokens: Option<u32>,
    /// The model's context window, in tokens. When set and a transcript would exceed it, the assist
    /// runs map-reduce (summarize chunks, then combine) instead of one over-long request.
    #[serde(default)]
    context_tokens: Option<u32>,
    /// Nucleus sampling cutoff (omitted from the request when unset).
    #[serde(default)]
    top_p: Option<f64>,
    /// Repetition penalty: positive discourages reusing the same words (omitted when unset).
    #[serde(default)]
    frequency_penalty: Option<f64>,
    /// Topic-novelty penalty: positive pushes toward new subjects (omitted when unset).
    #[serde(default)]
    presence_penalty: Option<f64>,
    /// A standing instruction prepended to every assist task on this endpoint (persona, language,
    /// style). Empty for none.
    #[serde(default)]
    system_prompt: String,
}

/// A user-defined OpenAI-compatible cloud endpoint: a base URL, the API shape it speaks, one model
/// id, and its assist tuning. The API key is stored separately in `cloud_keys` under `id`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
struct CustomCloudEndpoint {
    /// Stable slug used as the provider id and key handle, e.g. `"custom-my-gateway"`.
    id: String,
    /// Display name shown in the picker.
    name: String,
    /// Base URL, e.g. `http://host:40000/v1` (a trailing slash is trimmed by the engine).
    base_url: String,
    /// `"chat"` selects the `/chat/completions` audio shape; anything else = `/audio/transcriptions`.
    protocol: String,
    /// Wire model id sent to the API.
    model: String,
    /// AI notes/assist tuning (defaults when absent, so older saved files load unchanged).
    #[serde(default)]
    assist: AssistParams,
}

/// The payload the add/update endpoint commands accept — the editable fields of a custom endpoint,
/// as one object so the command stays within a sane parameter count.
#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct EndpointInput {
    name: String,
    base_url: String,
    protocol: String,
    model: String,
    #[serde(default)]
    assist: AssistParams,
}

impl CustomCloudEndpoint {
    /// The wire protocol this endpoint speaks (defaults to the transcription shape).
    fn cloud_protocol(&self) -> CloudProtocol {
        match self.protocol.as_str() {
            "chat" => CloudProtocol::OpenAiChatAudio,
            _ => CloudProtocol::OpenAi,
        }
    }

    /// The vendor-agnostic [`CloudProvider`] this endpoint maps to — one file-only model.
    fn to_provider(&self) -> CloudProvider {
        CloudProvider {
            id: self.id.clone(),
            display_name: self.name.clone(),
            protocol: self.cloud_protocol(),
            base_url: self.base_url.clone(),
            keys_url: String::new(),
            auth: CloudAuth::bearer(),
            streaming: None,
            models: vec![CloudModel {
                id: self.model.clone(),
                display_name: self.model.clone(),
                streaming: false,
                batch: true,
                languages: vec![],
                description: "Custom OpenAI-compatible endpoint.".to_owned(),
                recommended: true,
                diarizes: false,
            }],
        }
    }
}

/// Loads the user's custom endpoints; an absent or unreadable file yields none.
fn load_cloud_endpoints(path: &Path) -> Vec<CustomCloudEndpoint> {
    fs::read_to_string(path)
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_default()
}

/// Persists `endpoints` to `path` (local app data only). Best-effort — a write failure is logged.
fn save_cloud_endpoints(path: &Path, endpoints: &[CustomCloudEndpoint]) {
    match serde_json::to_string_pretty(endpoints) {
        Ok(json) => {
            if let Err(e) = fs::write(path, json) {
                eprintln!("wisp: could not persist custom cloud endpoints: {e}");
            }
        }
        Err(e) => eprintln!("wisp: could not serialize custom cloud endpoints: {e}"),
    }
}

/// Resolves a provider id — the built-in catalog first, then the user's custom endpoints.
fn resolve_cloud_provider(id: &str, endpoints: &[CustomCloudEndpoint]) -> Option<CloudProvider> {
    cloud_provider_by_id(id).or_else(|| {
        endpoints
            .iter()
            .find(|e| e.id == id)
            .map(|e| e.to_provider())
    })
}

/// A filesystem/url-safe slug for a custom endpoint id, unique against the catalog and `existing`
/// endpoints. Falls back to `"custom-endpoint"` for an all-symbol name, then disambiguates with `-N`.
fn unique_endpoint_id(name: &str, existing: &[CustomCloudEndpoint]) -> String {
    let mut slug = String::new();
    let mut prev_dash = false;
    for ch in name.trim().to_lowercase().chars() {
        if ch.is_ascii_alphanumeric() {
            slug.push(ch);
            prev_dash = false;
        } else if !prev_dash {
            slug.push('-');
            prev_dash = true;
        }
    }
    let base = format!("custom-{}", slug.trim_matches('-'));
    let base = if base == "custom-" {
        "custom-endpoint".to_owned()
    } else {
        base
    };

    let taken =
        |id: &str| cloud_provider_by_id(id).is_some() || existing.iter().any(|e| e.id == id);
    if !taken(&base) {
        return base;
    }
    (2..)
        .map(|n| format!("{base}-{n}"))
        .find(|id| !taken(id))
        .unwrap()
}

/// `provider` with the user's matching custom ids appended as batch-only models, so the picker and
/// the engine both see them. Blank ids and ones already in the catalog are skipped (no duplicates).
fn with_custom_models(mut provider: CloudProvider, customs: &[CloudCustomModel]) -> CloudProvider {
    for cm in customs.iter().filter(|c| c.provider == provider.id) {
        let id = cm.id.trim();

        if id.is_empty() || provider.models.iter().any(|m| m.id == id) {
            continue;
        }

        let name = cm.name.trim();

        provider.models.push(CloudModel {
            id: id.to_owned(),
            display_name: if name.is_empty() {
                id.to_owned()
            } else {
                name.to_owned()
            },
            streaming: false,
            batch: true,
            languages: vec![],
            description: "Custom model — added by you.".to_owned(),
            recommended: false,
            diarizes: false,
        });
    }
    provider
}

/// Adds a custom cloud model id for `provider`, so it appears in the picker and is usable at once.
/// Rejects an unknown provider, a blank id, or an id that already exists (catalog or custom).
#[tauri::command]
fn add_cloud_custom_model(
    state: State<'_, AppState>,
    provider: String,
    model_id: String,
    name: String,
) -> Result<(), String> {
    let base = cloud_provider_by_id(&provider)
        .ok_or_else(|| format!("unknown cloud provider {provider}"))?;

    let id = model_id.trim().to_owned();
    if id.is_empty() {
        return Err("a model id is required".to_owned());
    }

    let mut customs = state
        .cloud_custom_models
        .lock()
        .map_err(|_| "state lock poisoned".to_owned())?;

    if with_custom_models(base, &customs).model(&id).is_some() {
        return Err(format!("model '{id}' already exists for {provider}"));
    }

    customs.push(CloudCustomModel {
        provider,
        id,
        name: name.trim().to_owned(),
    });
    save_cloud_custom_models(&state.cloud_custom_models_path, &customs);
    Ok(())
}

/// Removes a previously added custom cloud model id; a no-op if it wasn't present.
#[tauri::command]
fn remove_cloud_custom_model(
    state: State<'_, AppState>,
    provider: String,
    model_id: String,
) -> Result<(), String> {
    let id = model_id.trim();

    let mut customs = state
        .cloud_custom_models
        .lock()
        .map_err(|_| "state lock poisoned".to_owned())?;

    customs.retain(|c| !(c.provider == provider && c.id == id));
    save_cloud_custom_models(&state.cloud_custom_models_path, &customs);
    Ok(())
}

/// Adds a custom OpenAI-compatible endpoint and returns its generated id (used as the provider id and
/// the key handle). Validates a non-blank name, an `http(s)` base URL, and a model id; the API key is
/// saved separately via `set_cloud_key` under the returned id.
#[tauri::command]
fn add_cloud_endpoint(state: State<'_, AppState>, input: EndpointInput) -> Result<String, String> {
    let (name, base_url, protocol, model) =
        clean_endpoint_fields(&input.name, &input.base_url, &input.protocol, &input.model)?;

    let mut endpoints = state
        .cloud_custom_endpoints
        .lock()
        .map_err(|_| "state lock poisoned".to_owned())?;

    let id = unique_endpoint_id(&name, &endpoints);
    endpoints.push(CustomCloudEndpoint {
        id: id.clone(),
        name,
        base_url,
        protocol,
        model,
        assist: normalize_assist(input.assist),
    });
    save_cloud_endpoints(&state.cloud_custom_endpoints_path, &endpoints);
    Ok(id)
}

/// Clamps assist params to sane ranges (and drops zero token caps to "unset") so a hand-edited or
/// stale value can never reach the provider as something invalid.
fn normalize_assist(mut assist: AssistParams) -> AssistParams {
    assist.temperature = assist.temperature.map(|t| t.clamp(0.0, 2.0));
    assist.top_p = assist.top_p.map(|p| p.clamp(0.0, 1.0));
    assist.frequency_penalty = assist.frequency_penalty.map(|p| p.clamp(-2.0, 2.0));
    assist.presence_penalty = assist.presence_penalty.map(|p| p.clamp(-2.0, 2.0));
    assist.max_tokens = assist.max_tokens.filter(|&n| n > 0);
    assist.context_tokens = assist.context_tokens.filter(|&n| n > 0);
    assist.system_prompt = assist.system_prompt.trim().to_owned();
    assist
}

/// Trims + validates a custom endpoint's fields, returning `(name, base_url, protocol, model)` or an
/// error message. `protocol` is normalized to `"chat"` or `"openai"`.
fn clean_endpoint_fields(
    name: &str,
    base_url: &str,
    protocol: &str,
    model: &str,
) -> Result<(String, String, String, String), String> {
    let name = name.trim();
    let base_url = base_url.trim().trim_end_matches('/');
    let model = model.trim();

    if name.is_empty() || base_url.is_empty() || model.is_empty() {
        return Err("name, base URL, and model id are all required".to_owned());
    }
    if !base_url.starts_with("http://") && !base_url.starts_with("https://") {
        return Err("base URL must start with http:// or https://".to_owned());
    }

    let protocol = if protocol == "chat" { "chat" } else { "openai" }.to_owned();
    Ok((
        name.to_owned(),
        base_url.to_owned(),
        protocol,
        model.to_owned(),
    ))
}

/// Updates an existing custom endpoint in place, keeping its id and stored key. Errors if the id is
/// unknown or a field is invalid.
#[tauri::command]
fn update_cloud_endpoint(
    state: State<'_, AppState>,
    id: String,
    input: EndpointInput,
) -> Result<(), String> {
    let (name, base_url, protocol, model) =
        clean_endpoint_fields(&input.name, &input.base_url, &input.protocol, &input.model)?;

    let mut endpoints = state
        .cloud_custom_endpoints
        .lock()
        .map_err(|_| "state lock poisoned".to_owned())?;
    let ep = endpoints
        .iter_mut()
        .find(|e| e.id == id)
        .ok_or_else(|| format!("unknown endpoint {id}"))?;

    ep.name = name;
    ep.base_url = base_url;
    ep.protocol = protocol;
    ep.model = model;
    ep.assist = normalize_assist(input.assist);
    save_cloud_endpoints(&state.cloud_custom_endpoints_path, &endpoints);
    Ok(())
}

/// Removes a custom endpoint and its stored key; a no-op if it wasn't present.
#[tauri::command]
fn remove_cloud_endpoint(state: State<'_, AppState>, id: String) -> Result<(), String> {
    {
        let mut endpoints = state
            .cloud_custom_endpoints
            .lock()
            .map_err(|_| "state lock poisoned".to_owned())?;
        endpoints.retain(|e| e.id != id);
        save_cloud_endpoints(&state.cloud_custom_endpoints_path, &endpoints);
    }

    let mut keys = state
        .cloud_keys
        .lock()
        .map_err(|_| "state lock poisoned".to_owned())?;
    if keys.remove(&id).is_some() {
        save_cloud_keys(&state.cloud_keys_path, &keys);
    }
    Ok(())
}

/// One tunable parameter spec, flattened for the UI's generic advanced-settings panel: `kind`
/// selects the control, with `min`/`max`/`step` for sliders and `options` for a dropdown.
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ParamSpecDto {
    key: String,
    label: String,
    help: String,
    /// `"float"` | `"int"` | `"bool"` | `"enum"` | `"text"`.
    kind: String,
    min: f64,
    max: f64,
    step: f64,
    options: Vec<EnumOptionDto>,
    /// Smart default as raw JSON (number/bool/string) — the control's initial value.
    default: serde_json::Value,
    advanced: bool,
}

/// One dropdown choice: the wire `value` and its display `label`.
#[derive(Serialize)]
struct EnumOptionDto {
    value: String,
    label: String,
}

/// Flattens a [`ParamSpec`] into its UI DTO.
fn param_spec_dto(spec: &ParamSpec) -> ParamSpecDto {
    let options = match &spec.kind {
        ParamKind::Enum(opts) => opts
            .iter()
            .map(|o| EnumOptionDto {
                value: o.value.clone(),
                label: o.label.clone(),
            })
            .collect(),
        _ => vec![],
    };
    let (kind, min, max, step) = match &spec.kind {
        ParamKind::Float { min, max, step } => ("float", *min, *max, *step),
        ParamKind::Int { min, max } => ("int", *min as f64, *max as f64, 1.0),
        ParamKind::Bool => ("bool", 0.0, 0.0, 0.0),
        ParamKind::Enum(_) => ("enum", 0.0, 0.0, 0.0),
        ParamKind::Text => ("text", 0.0, 0.0, 0.0),
        ParamKind::TextArea => ("textarea", 0.0, 0.0, 0.0),
    };

    ParamSpecDto {
        key: spec.key.clone(),
        label: spec.label.clone(),
        help: spec.help.clone(),
        kind: kind.to_owned(),
        min,
        max,
        step,
        options,
        default: param_value_json(&spec.default),
        advanced: spec.advanced,
    }
}

/// A [`ParamValue`] as raw JSON, for the control's initial value.
fn param_value_json(value: &ParamValue) -> serde_json::Value {
    match value {
        ParamValue::Float(v) => serde_json::json!(v),
        ParamValue::Int(v) => serde_json::json!(v),
        ParamValue::Bool(v) => serde_json::json!(v),
        ParamValue::Text(v) => serde_json::json!(v),
    }
}

/// The advanced live-streaming parameter specs a cloud `provider` exposes, for the generic settings
/// panel. Empty when the provider can't stream (so the panel simply shows nothing).
#[tauri::command]
fn streaming_params(provider: String, model: String) -> Result<Vec<ParamSpecDto>, String> {
    let provider = cloud_provider_by_id(&provider)
        .ok_or_else(|| format!("unknown cloud provider {provider}"))?;
    Ok(streaming_param_specs(&provider, &model)
        .iter()
        .map(param_spec_dto)
        .collect())
}

/// The advanced File (batch) parameter specs a cloud `provider` exposes for `model`, for the generic
/// settings panel. Empty when the provider/model has no extra knobs.
#[tauri::command]
fn batch_params(
    state: State<'_, AppState>,
    provider: String,
    model: String,
) -> Result<Vec<ParamSpecDto>, String> {
    let endpoints = state
        .cloud_custom_endpoints
        .lock()
        .map_err(|_| "state lock poisoned".to_owned())?;
    let provider = resolve_cloud_provider(&provider, &endpoints)
        .ok_or_else(|| format!("unknown cloud provider {provider}"))?;
    Ok(batch_param_specs(&provider, &model)
        .iter()
        .map(param_spec_dto)
        .collect())
}

/// The advanced parameter specs the chat assist exposes (temperature / top_p / max reply tokens), for
/// the generic settings panel — the assist-side counterpart of [`batch_params`]. Vendor-agnostic (every
/// assist provider speaks the same OpenAI-compatible chat tuning), so it takes no provider/model.
#[tauri::command]
fn assist_params() -> Vec<ParamSpecDto> {
    assist_param_specs().iter().map(param_spec_dto).collect()
}

/// The advanced parameter specs the **realtime** assist exposes (turn-detection endpointing + noise
/// reduction) — the realtime counterpart of [`assist_params`]. OpenAI-realtime-only, like the realtime
/// assist itself, so it takes no provider/model.
#[tauri::command]
fn assist_realtime_params() -> Vec<ParamSpecDto> {
    assist_realtime_param_specs()
        .iter()
        .map(param_spec_dto)
        .collect()
}

/// Runs a one-shot LLM task (summary, action items, or a custom prompt) over `transcript` using the
/// chat model of cloud `provider` — typically a user's custom OpenAI-compatible endpoint (their
/// gateway, a local Ollama, …). Runs off the main thread (the call is a slow HTTP round-trip).
#[tauri::command]
async fn run_llm_task(
    app: AppHandle,
    provider: String,
    model: String,
    system_prompt: String,
    transcript: String,
    params: HashMap<String, serde_json::Value>,
) -> Result<String, String> {
    tauri::async_runtime::spawn_blocking(move || {
        run_llm_task_blocking(app, &provider, &model, &system_prompt, &transcript, &params)
    })
    .await
    .map_err(|e| format!("LLM task failed: {e}"))?
}

/// The blocking body of [`run_llm_task`]: resolve the provider (catalog or custom endpoint) and its
/// key, then call the chat model with the task's system prompt and the transcript.
fn run_llm_task_blocking(
    app: AppHandle,
    provider_id: &str,
    model: &str,
    system_prompt: &str,
    transcript: &str,
    params: &HashMap<String, serde_json::Value>,
) -> Result<String, String> {
    if transcript.trim().is_empty() {
        return Err("there's no transcript to work on yet".to_owned());
    }

    let state = app.state::<AppState>();
    let (provider, key, assist) = resolve_assist_target(&state, provider_id)?;
    let assist = overlay_assist_params(assist, &build_param_values(&assist_param_specs(), params));

    // Prepend the endpoint's standing instruction (persona / language / style) to the task prompt.
    let system = combine_system(&assist.system_prompt, system_prompt);

    run_assist(&provider, model, &key, &system, transcript, &assist)
}

/// Resolves the cloud `provider` (catalog or custom endpoint), its on-device key, and its assist tuning
/// — the common preamble for every assist call. A catalog provider uses default tuning; a custom
/// endpoint carries its own (temperature, context size, system prompt, …).
fn resolve_assist_target(
    state: &AppState,
    provider_id: &str,
) -> Result<(CloudProvider, String, AssistParams), String> {
    let endpoints = state
        .cloud_custom_endpoints
        .lock()
        .map_err(|_| "state lock poisoned".to_owned())?;

    let provider = resolve_cloud_provider(provider_id, &endpoints)
        .ok_or_else(|| format!("unknown provider {provider_id}"))?;

    let key = state
        .cloud_keys
        .lock()
        .map_err(|_| "state lock poisoned".to_owned())?
        .get(provider_id)
        .filter(|k| !k.trim().is_empty())
        .cloned()
        .ok_or_else(|| format!("no API key saved for {provider_id}"))?;

    let assist = endpoints
        .iter()
        .find(|e| e.id == provider_id)
        .map(|e| e.assist.clone())
        .unwrap_or_default();

    Ok((provider, key, assist))
}

/// Overlays the user's advanced assist params (from the settings panel) onto the resolved endpoint
/// tuning: a knob the user moved off its default is present in `params` and overrides; one left at its
/// default isn't present, so the endpoint's own value (often "unset" → the model's own optimum) stands.
/// Each override is clamped here, since these bypass the save-time [`normalize_assist`]. A `max_tokens`
/// of 0 means "no cap" (its default), so it's treated as unset.
fn overlay_assist_params(mut assist: AssistParams, params: &ParamValues) -> AssistParams {
    if params.contains("temperature") {
        assist.temperature = Some(params.float("temperature", 1.0).clamp(0.0, 2.0));
    }

    if params.contains("top_p") {
        assist.top_p = Some(params.float("top_p", 1.0).clamp(0.0, 1.0));
    }

    if params.contains("frequency_penalty") {
        assist.frequency_penalty = Some(params.float("frequency_penalty", 0.0).clamp(-2.0, 2.0));
    }

    if params.contains("presence_penalty") {
        assist.presence_penalty = Some(params.float("presence_penalty", 0.0).clamp(-2.0, 2.0));
    }

    if params.contains("max_tokens") {
        let n = params.int("max_tokens", 0).clamp(0, i64::from(u32::MAX));
        if n > 0 {
            assist.max_tokens = Some(n as u32);
        }
    }

    assist
}

/// Streams a chat assist task into the feed: resolve the provider, then run a single chat call with
/// `stream: true`, emitting each chunk as [`ASSIST_DELTA_EVENT`] and the full reply as
/// [`ASSIST_TEXT_EVENT`]. A transcript too long for one call falls back to (non-streamed) map-reduce and
/// emits the combined result whole. Off the main thread — the call is a slow streamed round-trip.
#[tauri::command]
async fn run_assist_stream(
    app: AppHandle,
    provider: String,
    model: String,
    system_prompt: String,
    transcript: String,
    params: HashMap<String, serde_json::Value>,
) -> Result<(), String> {
    tauri::async_runtime::spawn_blocking(move || {
        run_assist_stream_blocking(app, &provider, &model, &system_prompt, &transcript, &params)
    })
    .await
    .map_err(|e| format!("assist stream task failed: {e}"))?
}

fn run_assist_stream_blocking(
    app: AppHandle,
    provider_id: &str,
    model: &str,
    system_prompt: &str,
    transcript: &str,
    params: &HashMap<String, serde_json::Value>,
) -> Result<(), String> {
    if transcript.trim().is_empty() {
        return Err("there's no transcript to work on yet".to_owned());
    }

    let state = app.state::<AppState>();
    let (provider, key, assist) = resolve_assist_target(&state, provider_id)?;
    let assist = overlay_assist_params(assist, &build_param_values(&assist_param_specs(), params));

    let system = combine_system(&assist.system_prompt, system_prompt);

    // A transcript that fits one call streams token-by-token; one too long map-reduces (each chunk
    // whole, no per-token stream) and the combined result is emitted as the single final reply.
    let text = match input_char_budget(assist.context_tokens) {
        Some(budget) if transcript.chars().count() > budget => {
            map_reduce_assist(&provider, model, &key, &system, transcript, &assist, budget)?
        }
        _ => {
            let app_delta = app.clone();
            let req = ChatRequest {
                system: &system,
                user: transcript,
                temperature: assist.temperature,
                max_tokens: assist.max_tokens,
                top_p: assist.top_p,
                frequency_penalty: assist.frequency_penalty,
                presence_penalty: assist.presence_penalty,
            };
            chat_completion_stream(&provider, model, &key, &req, |chunk| {
                let _ = app_delta.emit(ASSIST_DELTA_EVENT, chunk.to_owned());
            })
            .map_err(|e| e.to_string())?
        }
    };

    let _ = app.emit(ASSIST_TEXT_EVENT, text.trim().to_owned());
    Ok(())
}

/// Approximate characters per token — used only to decide when a transcript needs map-reduce. A
/// rough cross-language middle (English ~4, CJK ~1–2); 3 errs toward chunking, which is the safe way
/// to be wrong (an extra round-trip beats overflowing the model's context).
const ASSIST_CHARS_PER_TOKEN: usize = 3;

/// The character budget for one assist request given a context window in tokens, or `None` when no
/// window is set (0 counts as unset). Reserves ~40% of the window for the system prompt and reply.
fn input_char_budget(context_tokens: Option<u32>) -> Option<usize> {
    context_tokens
        .filter(|&t| t > 0)
        .map(|t| (t as usize * ASSIST_CHARS_PER_TOKEN * 3) / 5)
}

/// Splits `text` into chunks of at most `budget` characters, breaking only at line boundaries (one
/// transcript turn per line) so an utterance is never cut mid-sentence. A single over-long line
/// becomes its own chunk.
fn chunk_by_chars(text: &str, budget: usize) -> Vec<String> {
    let mut chunks = Vec::new();
    let mut cur = String::new();

    for line in text.lines() {
        if !cur.is_empty() && cur.chars().count() + 1 + line.chars().count() > budget {
            chunks.push(std::mem::take(&mut cur));
        }
        if !cur.is_empty() {
            cur.push('\n');
        }
        cur.push_str(line);
    }

    if !cur.is_empty() {
        chunks.push(cur);
    }
    chunks
}

/// Prepends an endpoint's standing instruction to a task's system prompt (blank-safe).
fn combine_system(standing: &str, task: &str) -> String {
    let standing = standing.trim();

    if standing.is_empty() {
        task.to_owned()
    } else {
        format!("{standing}\n\n{task}")
    }
}

/// Runs an assist task, transparently map-reducing when `context_tokens` is set and the transcript
/// would overflow it: split into chunks, run the task on each, then combine the partial results.
fn run_assist(
    provider: &CloudProvider,
    model: &str,
    key: &str,
    system: &str,
    transcript: &str,
    assist: &AssistParams,
) -> Result<String, String> {
    match input_char_budget(assist.context_tokens) {
        Some(budget) if transcript.chars().count() > budget => {
            map_reduce_assist(provider, model, key, system, transcript, assist, budget)
        }
        _ => chat_once(provider, model, key, system, transcript, assist),
    }
}

/// One chat-completion call with the endpoint's tuning — temperature / max_tokens / top_p are each
/// sent only when set, so a model that rejects a non-default temperature isn't sent one.
fn chat_once(
    provider: &CloudProvider,
    model: &str,
    key: &str,
    system: &str,
    user: &str,
    assist: &AssistParams,
) -> Result<String, String> {
    chat_completion(
        provider,
        model,
        key,
        &ChatRequest {
            system,
            user,
            temperature: assist.temperature,
            max_tokens: assist.max_tokens,
            top_p: assist.top_p,
            frequency_penalty: assist.frequency_penalty,
            presence_penalty: assist.presence_penalty,
        },
    )
    .map_err(|e| e.to_string())
}

/// Map-reduce for a transcript that exceeds the context window: run the task on each chunk (map),
/// then combine the partials under the original instruction (reduce). Covers the whole transcript
/// rather than truncating it.
fn map_reduce_assist(
    provider: &CloudProvider,
    model: &str,
    key: &str,
    system: &str,
    transcript: &str,
    assist: &AssistParams,
    budget: usize,
) -> Result<String, String> {
    let chunks = chunk_by_chars(transcript, budget);

    let mut partials = Vec::with_capacity(chunks.len());
    for (i, chunk) in chunks.iter().enumerate() {
        let part = chat_once(provider, model, key, system, chunk, assist)?;
        partials.push(format!(
            "=== Part {}/{} ===\n{}",
            i + 1,
            chunks.len(),
            part.trim()
        ));
    }

    let reduce_system = format!(
        "Below are results from running the same instruction on consecutive parts of one long \
         transcript. Combine them into a single coherent result that follows this instruction, \
         merging duplicates and keeping it faithful:\n\n{system}"
    );

    chat_once(
        provider,
        model,
        key,
        &reduce_system,
        &partials.join("\n\n"),
        assist,
    )
}

/// A user-imported model. Its files live under `custom_models_dir/<id>/`; `kind` selects the engine
/// adapter — today only `"whisper-cpp"` (a GGML/GGUF `.bin`); ONNX kinds slot in here later.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
struct CustomModel {
    id: String,
    name: String,
    kind: String,
    files: Vec<String>,
}

/// Loads the imported-model registry from `dir/registry.json`; an absent/garbage file yields none.
fn load_custom_models(dir: &Path) -> Vec<CustomModel> {
    fs::read_to_string(dir.join("registry.json"))
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_default()
}

/// Persists the registry beside the model files. Best-effort — a write failure is logged.
fn save_custom_models(dir: &Path, models: &[CustomModel]) {
    match serde_json::to_string_pretty(models) {
        Ok(json) => {
            if let Err(e) = fs::write(dir.join("registry.json"), json) {
                eprintln!("wisp: could not persist custom models: {e}");
            }
        }
        Err(e) => eprintln!("wisp: could not serialize custom models: {e}"),
    }
}

/// The engine family a custom `kind` maps to, or `None` if the kind isn't recognised.
fn custom_family(kind: &str) -> Option<ModelFamily> {
    match kind {
        "whisper-cpp" => Some(ModelFamily::WhisperCpp),
        _ => None,
    }
}

/// A `ModelDescriptor` for a custom model, so it flows through `build_engine` exactly like a catalog
/// model. Its files carry no URL/checksum — they're already on disk.
fn custom_descriptor(cm: &CustomModel) -> Option<ModelDescriptor> {
    let family = custom_family(&cm.kind)?;
    Some(ModelDescriptor {
        id: ModelId(cm.id.clone()),
        family,
        quant: Quant::F32,
        display_name: cm.name.clone(),
        files: cm
            .files
            .iter()
            .map(|name| ModelFile {
                name: name.clone(),
                url: String::new(),
                sha256: String::new(),
                size_bytes: 0,
            })
            .collect(),
        languages: ["yue", "zh", "en", "ja"]
            .iter()
            .map(|s| (*s).to_owned())
            .collect(),
        description: "Imported custom model.".to_owned(),
    })
}

/// The picker DTO for a custom model — always installed (its files are on disk).
fn custom_model_info(cm: &CustomModel, active: Option<&ModelId>) -> Option<ModelInfoDto> {
    let descriptor = custom_descriptor(cm)?;
    let is_active = active == Some(&descriptor.id);
    Some(ModelInfoDto {
        id: descriptor.id.0,
        name: descriptor.display_name,
        size_bytes: 0,
        languages: descriptor.languages,
        description: descriptor.description,
        installed: true,
        active: is_active,
        family: format!("{:?}", descriptor.family),
        recommended_live: false,
        recommended_file: false,
        coreml_available: false,
        coreml_installed: false,
        coreml_size_bytes: 0,
        // The caller (`list_models`) overwrites these from the machine fit; a comfortable default here.
        fit: "ready".to_owned(),
        fit_reason: None,
        deletable: false,
    })
}

/// Resolves a model id to its descriptor + directory, checking the built-in catalog first and then
/// the user's imported custom models. The single resolution path for both transcribe and live start.
fn resolve_local_model(
    state: &AppState,
    id: &ModelId,
) -> Result<(ModelDescriptor, PathBuf), String> {
    if let Some(descriptor) = state.store.available().into_iter().find(|d| d.id == *id) {
        // A zero-file, OS-provided model (Apple on-device speech) has nothing to download — materialize
        // its dir on first use rather than gating it behind a "not downloaded yet" error.
        let dir = if descriptor.files.is_empty() {
            state.store.ensure(id).map_err(|e| e.to_string())?
        } else {
            state
                .store
                .local_path(id)
                .ok_or_else(|| format!("model '{}' is not downloaded yet", id.as_str()))?
        };
        return Ok((descriptor, dir));
    }

    let custom = state
        .custom_models
        .lock()
        .map_err(|_| "state lock poisoned".to_owned())?;
    let cm = custom
        .iter()
        .find(|c| c.id == id.0)
        .ok_or("selected model is not in the catalog")?;
    let descriptor = custom_descriptor(cm).ok_or("custom model has an unknown kind")?;
    Ok((descriptor, state.custom_models_dir.join(&cm.id)))
}

/// Whether `bytes` begin with a GGML/GGUF magic — the marker of a whisper.cpp model file.
fn is_ggml(bytes: &[u8]) -> bool {
    matches!(&bytes[..bytes.len().min(4)], b"GGUF" | b"ggml" | b"lmgg")
}

/// A filesystem-safe `custom-<slug>` id, suffixed `-2`, `-3`, … on collision.
fn unique_custom_id(name: &str, existing: &[CustomModel]) -> String {
    let slug: String = name
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() {
                c.to_ascii_lowercase()
            } else {
                '-'
            }
        })
        .collect();
    let base = format!("custom-{}", slug.trim_matches('-'));
    let mut id = base.clone();
    let mut n = 2;
    while existing.iter().any(|c| c.id == id) {
        id = format!("{base}-{n}");
        n += 1;
    }
    id
}

/// Imports the model file the user picked: validates it, copies it into the app's custom-models
/// directory, registers it, and returns its picker entry. Today it accepts a whisper.cpp GGML/GGUF
/// `.bin`/`.gguf` (runs on the Metal GPU); ONNX models for the CPU are a later addition.
#[tauri::command]
fn import_custom_model(state: State<'_, AppState>, path: String) -> Result<ModelInfoDto, String> {
    let src = PathBuf::from(&path);
    let ext = src
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase();
    if ext != "bin" && ext != "gguf" {
        return Err("Import a Whisper model file (.bin or .gguf, GGML/GGUF format).".to_owned());
    }

    let mut head = [0u8; 4];
    {
        use std::io::Read;
        let mut f = fs::File::open(&src).map_err(|e| format!("cannot open file: {e}"))?;
        let _ = f.read(&mut head);
    }
    if !is_ggml(&head) {
        return Err("That file isn't a GGML/GGUF Whisper model.".to_owned());
    }

    let file_name = src
        .file_name()
        .and_then(|n| n.to_str())
        .ok_or("file has no name")?
        .to_owned();
    let name = src
        .file_stem()
        .and_then(|n| n.to_str())
        .unwrap_or("Custom model")
        .to_owned();

    let mut models = state
        .custom_models
        .lock()
        .map_err(|_| "state lock poisoned".to_owned())?;
    let id = unique_custom_id(&name, &models);

    let dest_dir = state.custom_models_dir.join(&id);
    fs::create_dir_all(&dest_dir).map_err(|e| format!("cannot create model dir: {e}"))?;
    fs::copy(&src, dest_dir.join(&file_name)).map_err(|e| format!("cannot copy model: {e}"))?;

    let cm = CustomModel {
        id,
        name,
        kind: "whisper-cpp".to_owned(),
        files: vec![file_name],
    };
    models.push(cm.clone());
    save_custom_models(&state.custom_models_dir, &models);

    custom_model_info(&cm, None).ok_or_else(|| "could not register the model".to_owned())
}

#[tauri::command]
async fn download_model(
    app: AppHandle,
    state: State<'_, AppState>,
    id: String,
) -> Result<(), String> {
    let store = Arc::clone(&state.store);
    let model_id = ModelId(id.clone());
    let emitter = app.clone();

    tauri::async_runtime::spawn_blocking(move || {
        // Throttle progress events to ~8/s; always send the final 100%.
        let mut last: Option<Instant> = None;
        let mut on_progress = |downloaded: u64, total: u64| {
            let now = Instant::now();
            let throttle_ok = match last {
                Some(t) => now.duration_since(t) >= Duration::from_millis(120),
                None => true,
            };
            if downloaded >= total || throttle_ok {
                last = Some(now);
                let _ = emitter.emit(
                    DOWNLOAD_PROGRESS_EVENT,
                    DownloadProgressDto {
                        id: id.clone(),
                        downloaded,
                        total,
                    },
                );
            }
        };
        store.ensure_with_progress(&model_id, &mut on_progress)
    })
    .await
    .map_err(|e| e.to_string())?
    .map_err(|e| e.to_string())?;
    Ok(())
}

/// Downloads the optional Core ML (Neural Engine) encoder for an installed whisper.cpp model and
/// unpacks it next to the model. Progress is emitted under the id `coreml:<model-id>`.
#[tauri::command]
async fn download_coreml(
    app: AppHandle,
    state: State<'_, AppState>,
    id: String,
) -> Result<(), String> {
    let store = Arc::clone(&state.store);
    let model_id = ModelId(id.clone());
    let asset = state
        .store
        .available()
        .into_iter()
        .find(|d| d.id == model_id)
        .and_then(|d| coreml_asset(&d))
        .ok_or_else(|| format!("no Core ML encoder for {id}"))?;
    let emitter = app.clone();
    let progress_id = format!("coreml:{id}");

    tauri::async_runtime::spawn_blocking(move || {
        // Throttle progress events to ~8/s; always send the final 100%.
        let mut last: Option<Instant> = None;
        let mut on_progress = |downloaded: u64, total: u64| {
            let now = Instant::now();
            let throttle_ok = match last {
                Some(t) => now.duration_since(t) >= Duration::from_millis(120),
                None => true,
            };
            if downloaded >= total || throttle_ok {
                last = Some(now);
                let _ = emitter.emit(
                    DOWNLOAD_PROGRESS_EVENT,
                    DownloadProgressDto {
                        id: progress_id.clone(),
                        downloaded,
                        total,
                    },
                );
            }
        };
        store.ensure_coreml_with_progress(&model_id, &asset, &mut on_progress)
    })
    .await
    .map_err(|e| e.to_string())?
    .map_err(|e| e.to_string())?;
    Ok(())
}

#[tauri::command]
fn select_model(state: State<'_, AppState>, id: String) -> Result<(), String> {
    let model_id = ModelId(id);
    let in_catalog = state.store.available().iter().any(|d| d.id == model_id);
    let in_custom = state
        .custom_models
        .lock()
        .map_err(|_| "state lock poisoned".to_owned())?
        .iter()
        .any(|c| c.id == model_id.0);
    if !in_catalog && !in_custom {
        return Err("unknown model".to_owned());
    }
    // Persist the choice so it survives a restart (best-effort — not fatal if it fails).
    let _ = fs::write(&state.active_model_path, model_id.as_str());
    *state
        .active
        .lock()
        .map_err(|_| "state lock poisoned".to_owned())? = Some(model_id);
    Ok(())
}

/// Deletes an installed local model's downloaded files to reclaim disk space. If it was the active
/// model the active selection is cleared (in memory and on disk), so the app honestly shows "no model"
/// rather than pointing at files that are gone. The model stays in the catalog — re-downloadable anytime.
#[tauri::command]
fn remove_model(state: State<'_, AppState>, id: String) -> Result<(), String> {
    let model_id = ModelId(id);

    let descriptor = state
        .store
        .available()
        .into_iter()
        .find(|d| d.id == model_id)
        .ok_or_else(|| "only downloaded catalog models can be deleted".to_owned())?;

    if descriptor.files.is_empty() {
        return Err("this model has no files to delete".to_owned());
    }

    state.store.remove(&model_id).map_err(|e| e.to_string())?;

    clear_active_if_removed(state.inner(), &model_id)?;

    Ok(())
}

/// Clears the active selection — in memory and the persisted file — when the model just removed was the
/// active one, so the app never holds a dangling pointer to deleted files.
fn clear_active_if_removed(state: &AppState, removed: &ModelId) -> Result<(), String> {
    let mut active = state
        .active
        .lock()
        .map_err(|_| "state lock poisoned".to_owned())?;

    if active.as_ref() == Some(removed) {
        *active = None;
        let _ = fs::remove_file(&state.active_model_path);
    }

    Ok(())
}

#[tauri::command]
fn list_input_devices() -> Vec<String> {
    wisp_audio::list_input_devices()
}

#[tauri::command]
fn system_audio_id() -> &'static str {
    SYSTEM_CAPTURE_ID
}

#[tauri::command]
fn mic_off_id() -> &'static str {
    MIC_OFF_ID
}

#[tauri::command]
fn screen_recording_authorized() -> bool {
    permissions::screen_recording_authorized()
}

#[tauri::command]
fn request_screen_recording() -> bool {
    permissions::request_screen_recording()
}

#[tauri::command]
fn open_privacy_settings(pane: String) -> Result<(), String> {
    permissions::open_privacy_settings(&pane)
}

/// Relaunches the app. macOS applies a newly granted Screen Recording permission only to a fresh
/// process, so after the user enables Wisp in System Settings the running app must restart to pick
/// it up — otherwise `CGPreflightScreenCaptureAccess` keeps reporting "not authorized" and the
/// permission banner never clears.
#[tauri::command]
fn restart_app(app: AppHandle) {
    app.restart()
}

#[tauri::command]
fn microphone_blocked() -> bool {
    permissions::microphone_blocked()
}

#[tauri::command]
fn session_running(state: State<'_, AppState>) -> Result<bool, String> {
    Ok(!state
        .sessions
        .lock()
        .map_err(|_| "state lock poisoned".to_owned())?
        .is_empty())
}

#[tauri::command]
fn set_language(state: State<'_, AppState>, language: String) -> Result<(), String> {
    *state
        .language
        .lock()
        .map_err(|_| "state lock poisoned".to_owned())? = language;
    Ok(())
}

/// Sets the live denoiser id (`"rnnoise"`/`"denoise-gtcrn"`/none), applied to the next session.
#[tauri::command]
fn set_denoise(state: State<'_, AppState>, denoiser: Option<String>) -> Result<(), String> {
    *state
        .live_denoiser
        .lock()
        .map_err(|_| "state lock poisoned".to_owned())? = denoiser;
    Ok(())
}

/// Sets the live diarization model id (or clears it), applied to the next session that starts.
#[tauri::command]
fn set_live_diarize(state: State<'_, AppState>, model: Option<String>) -> Result<(), String> {
    *state
        .live_diarize_model
        .lock()
        .map_err(|_| "state lock poisoned".to_owned())? = model;
    Ok(())
}

/// Sets the live decoding options — biasing prompt (hints) and accurate/fast — for the next session.
#[tauri::command]
fn set_live_decode(
    state: State<'_, AppState>,
    prompt: String,
    accurate: bool,
) -> Result<(), String> {
    *state
        .live_prompt
        .lock()
        .map_err(|_| "state lock poisoned".to_owned())? = prompt;
    *state
        .live_accurate
        .lock()
        .map_err(|_| "state lock poisoned".to_owned())? = accurate;
    Ok(())
}

#[tauri::command]
fn set_devices(
    state: State<'_, AppState>,
    mic: Option<String>,
    system: Option<String>,
) -> Result<(), String> {
    *state
        .mic_device
        .lock()
        .map_err(|_| "state lock poisoned".to_owned())? = mic;
    *state
        .system_device
        .lock()
        .map_err(|_| "state lock poisoned".to_owned())? = system;
    Ok(())
}

/// Opens the system-audio capture source for this platform, or an error so the caller degrades to
/// mic-only. macOS uses ScreenCaptureKit and Windows uses WASAPI loopback — both one-click, no
/// virtual device; other platforms report unavailable until their own is added.
#[cfg(target_os = "macos")]
fn open_system_capture() -> Result<Box<dyn AudioSource>, String> {
    ScreenCaptureSource::new()
        .map(|s| Box::new(s) as Box<dyn AudioSource>)
        .map_err(|e| e.to_string())
}

#[cfg(target_os = "windows")]
fn open_system_capture() -> Result<Box<dyn AudioSource>, String> {
    WasapiLoopbackSource::new()
        .map(|s| Box::new(s) as Box<dyn AudioSource>)
        .map_err(|e| e.to_string())
}

#[cfg(not(any(target_os = "macos", target_os = "windows")))]
fn open_system_capture() -> Result<Box<dyn AudioSource>, String> {
    Err("system-audio capture isn't available on this platform yet".to_owned())
}

/// How long [`open_mic_within`] waits for the microphone to come up before failing Start. A healthy
/// device opens in well under a second; the budget covers first-use permission + a busy HAL. cpal's
/// open can wedge indefinitely on a CoreAudio HAL left confused by a prior hard-kill, so bounding it
/// turns "Start hangs forever" into a fast, actionable error.
const MIC_STARTUP_TIMEOUT: Duration = Duration::from_secs(5);

/// Opens a microphone source (a named device, or the default) within [`MIC_STARTUP_TIMEOUT`] so a
/// wedged audio system fails Start fast with a clear message instead of hanging it forever. On
/// timeout the in-flight open is detached; it self-cleans when the HAL recovers.
fn open_mic_within(device: Option<String>) -> Result<Box<dyn AudioSource>, String> {
    run_within(
        MIC_STARTUP_TIMEOUT,
        move || -> Result<Box<dyn AudioSource>, String> {
            let mic = match device {
                Some(name) => MicSource::from_device(&name).map_err(|e| e.to_string())?,
                None => MicSource::from_default().map_err(|e| e.to_string())?,
            };
            Ok(Box::new(mic))
        },
    )
    .unwrap_or_else(|| {
        Err(format!(
            "microphone didn't start within {MIC_STARTUP_TIMEOUT:?} — it may be held by another app or the audio system is wedged. Restart the app, or reset audio with: sudo killall coreaudiod"
        ))
    })
}

/// The echo canceller for this platform: WebRTC AEC on macOS (falling back to passthrough if it
/// won't init), passthrough elsewhere. Keeping it a `Box<dyn EchoCanceller>` lets the dual-stream
/// path stay identical on every platform — the cross-stream dedup handles residual echo where there
/// is no real AEC.
#[cfg(target_os = "macos")]
fn echo_canceller() -> Box<dyn EchoCanceller> {
    match WebrtcEchoCanceller::new() {
        Ok(c) => Box::new(c),
        Err(e) => {
            eprintln!("wisp: AEC unavailable ({e}); passing the microphone through");
            Box::new(PassthroughEchoCanceller)
        }
    }
}

#[cfg(not(target_os = "macos"))]
fn echo_canceller() -> Box<dyn EchoCanceller> {
    Box::new(PassthroughEchoCanceller)
}

/// How a live session is engined: the active on-device model (default), or a cloud realtime provider
/// with its advanced parameter overrides. Mirrors the File path's `engine`/`cloudProvider` options.
#[derive(Default, Deserialize)]
#[serde(rename_all = "camelCase")]
struct LiveOptions {
    /// `"cloud"` for a realtime cloud provider; anything else (or absent) uses the on-device model.
    engine: Option<String>,
    cloud_provider: Option<String>,
    cloud_model: Option<String>,
    /// Advanced parameter overrides keyed by `ParamSpec::key` (raw JSON, coerced per spec).
    #[serde(default)]
    params: HashMap<String, serde_json::Value>,
    /// Whether to tap the live audio for the realtime AI assist. The frontend sets it only when a
    /// real-time assist model is armed, so an ordinary session that never uses the assist pays nothing
    /// (no extra tee pump threads): the live audio path is then byte-for-byte the original.
    #[serde(default)]
    assist: bool,
}

/// Resolves the engine a live session will run from `options` + app state: an on-device model, or a
/// cloud realtime provider/model with its key (from local storage) and merged parameter overrides.
/// The realtime streaming protocol a cloud live session should use, or `None` to fall back to
/// segment-batch. Realtime needs both a streaming-capable provider and a realtime-capable model; a
/// batch-only model (e.g. a custom endpoint) returns `None` and runs segment-batch instead.
fn cloud_live_protocol(provider: &CloudProvider, model_id: &str) -> Option<StreamingProtocol> {
    let model_streams = provider
        .model(model_id)
        .map(|m| m.streaming)
        .unwrap_or(false);

    provider.streaming.filter(|_| model_streams)
}

fn resolve_live_engine(state: &AppState, options: &LiveOptions) -> Result<LiveEngine, String> {
    if options.engine.as_deref() != Some("cloud") {
        let active = state
            .active
            .lock()
            .map_err(|_| "state lock poisoned".to_owned())?
            .clone()
            .ok_or("no model selected")?;
        let (descriptor, dir) = resolve_local_model(state, &active)?;
        return Ok(LiveEngine::Local { descriptor, dir });
    }

    let provider_id = options
        .cloud_provider
        .clone()
        .ok_or("no cloud provider selected")?;
    let model = options
        .cloud_model
        .clone()
        .ok_or("no cloud model selected")?;

    let key = state
        .cloud_keys
        .lock()
        .map_err(|_| "state lock poisoned".to_owned())?
        .get(&provider_id)
        .filter(|k| !k.trim().is_empty())
        .cloned()
        .ok_or_else(|| format!("no API key saved for {provider_id}"))?;

    // Resolve from the catalog OR the user's custom endpoints (the same lookup File uses), then merge
    // any custom model ids — so a custom endpoint / model selected in the picker is found here too.
    let provider = {
        let endpoints = state
            .cloud_custom_endpoints
            .lock()
            .map_err(|_| "state lock poisoned".to_owned())?;
        resolve_cloud_provider(&provider_id, &endpoints)
            .ok_or_else(|| format!("unknown cloud provider {provider_id}"))?
    };
    let provider = {
        let customs = state
            .cloud_custom_models
            .lock()
            .map_err(|_| "state lock poisoned".to_owned())?;
        with_custom_models(provider, &customs)
    };

    // Realtime when the provider speaks a streaming socket and the model is realtime-capable;
    // otherwise segment-batch — the local VAD cuts utterances and the cloud transcribes each whole,
    // which lets any batch model (including custom endpoints) run live (finalize-only).
    if let Some(streaming) = cloud_live_protocol(&provider, &model) {
        let params = build_param_values(&streaming_param_specs(&provider, &model), &options.params);

        return Ok(LiveEngine::CloudStreaming {
            streaming,
            model,
            key,
            params,
        });
    }

    let params = build_param_values(&batch_param_specs(&provider, &model), &options.params);

    Ok(LiveEngine::CloudBatch {
        provider,
        model,
        key,
        params,
    })
}

/// Starts a live session. Returns an optional non-fatal notice (e.g. system audio was unavailable
/// so it fell back to mic-only) for the UI to surface; `None` means everything started as requested.
#[tauri::command]
async fn start_session(app: AppHandle, options: LiveOptions) -> Result<Option<String>, String> {
    // The cloud engine connects its WebSocket synchronously (a ~1-2s blocking handshake), and a
    // sync Tauri command runs on the main thread — which would freeze the UI (the Start spinner
    // can't even animate). Run the whole start on a blocking worker so the UI stays smooth.
    tauri::async_runtime::spawn_blocking(move || start_session_blocking(app, options))
        .await
        .map_err(|e| format!("session start task failed: {e}"))?
}

/// The blocking body of [`start_session`], run off the main thread: it builds the engine(s) —
/// connecting the cloud WebSocket — and spawns the capture/transcription session.
fn start_session_blocking(app: AppHandle, options: LiveOptions) -> Result<Option<String>, String> {
    let state = app.state::<AppState>();
    let mut sessions = state
        .sessions
        .lock()
        .map_err(|_| "state lock poisoned".to_owned())?;
    if !sessions.is_empty() {
        return Err("a session is already running".to_owned());
    }

    // Start a fresh meeting transcript for export — drop the previous session's retained finals.
    if let Ok(mut retained) = state.live_segments.lock() {
        retained.clear();
    }
    // Every stream starts unmuted; the Live bar's You/Them chips flip these mid-session.
    state.mic_muted.store(false, Ordering::Relaxed);
    state.system_muted.store(false, Ordering::Relaxed);

    let live_engine = resolve_live_engine(&state, &options)?;
    let language = state
        .language
        .lock()
        .map_err(|_| "state lock poisoned".to_owned())?
        .clone();
    // Live denoising runs frame-by-frame in real time, but the downloadable model denoisers (GTCRN)
    // are offline/whole-clip — run per frame they emit garbled near-silence the engine then
    // hallucinates over. So live only ever uses the streaming built-in RNNoise: coerce any model
    // denoiser to it. (Those models stay available for File mode, which feeds them a whole clip.)
    let denoiser = match state
        .live_denoiser
        .lock()
        .map_err(|_| "state lock poisoned".to_owned())?
        .as_deref()
    {
        None => None,
        Some("rnnoise") => Some("rnnoise".to_owned()),
        Some(other) => {
            eprintln!("wisp: '{other}' is an offline denoiser unsuitable for live — using RNNoise");
            Some("rnnoise".to_owned())
        }
    };
    // The live denoiser is always the built-in RNNoise, which needs no downloaded model directory.
    let denoise_dir: Option<PathBuf> = None;
    let diarize_dir = match &*state
        .live_diarize_model
        .lock()
        .map_err(|_| "state lock poisoned".to_owned())?
    {
        Some(model) => Some(
            state
                .store
                .local_path(&ModelId(model.clone()))
                .ok_or_else(|| format!("speaker model '{model}' is not downloaded yet"))?,
        ),
        None => None,
    };
    let prompt = state
        .live_prompt
        .lock()
        .map_err(|_| "state lock poisoned".to_owned())?
        .clone();
    let accurate = *state
        .live_accurate
        .lock()
        .map_err(|_| "state lock poisoned".to_owned())?;
    let settings = LiveSettings {
        language,
        denoiser,
        denoise_dir,
        diarize_dir,
        prompt,
        accurate,
    };

    // Open the audio sources first (these can fail: device missing / no mic permission).
    let mic_device = state
        .mic_device
        .lock()
        .map_err(|_| "state lock poisoned".to_owned())?
        .clone();
    let mic_source: Option<Box<dyn AudioSource>> = match mic_device {
        Some(name) if name == MIC_OFF_ID => None,
        device => Some(open_mic_within(device)?),
    };

    let system_device = state
        .system_device
        .lock()
        .map_err(|_| "state lock poisoned".to_owned())?
        .clone();
    // System audio is best-effort: if ScreenCaptureKit can't start (a macOS that doesn't support
    // it, no Screen Recording permission, no display), fall back to mic-only rather than failing
    // the whole session — so live transcription works on every Mac, just without meeting audio.
    let mut degraded_notice: Option<String> = None;
    let system_source: Option<Box<dyn AudioSource>> = match system_device {
        Some(name) if name == SYSTEM_CAPTURE_ID => match open_system_capture() {
            Ok(source) => Some(source),
            Err(e) => {
                degraded_notice = Some(format!(
                    "System audio unavailable ({e}) — capturing the microphone only."
                ));
                None
            }
        },
        Some(name) => Some(open_mic_within(Some(name))?),
        None => None,
    };

    // Tap each processed transcription stream so the realtime assist can hear the same audio (mixed).
    let mut assist_branches: Vec<FrameReceiver> = Vec::new();
    let mut assist_tees: Vec<Tee> = Vec::new();

    match (mic_source, system_source) {
        // Both on → the mic re-hears the system audio on speakers (echo). Tee the system capture
        // into the meeting pipeline and the AEC far-end reference, and clean the mic against it.
        (Some(mic), Some(system)) => {
            let system_info = system.info();
            let (tee_handle, meeting_rx, reference_rx) = tee(system);

            // Shared across both sessions: drops any residual echo AEC leaves on the mic stream.
            let dedup = Arc::new(Mutex::new(CrossStreamEchoFilter::new()));

            // Echo-cancel the mic against the system reference (WebRTC on macOS, passthrough where
            // there's no platform AEC); either way the reference is consumed and the dedup mops up.
            let aec_mic: Box<dyn AudioSource> = Box::new(EchoCancellingSource::new(
                mic,
                reference_rx,
                echo_canceller(),
            ));
            let aec_mic = tap_for_assist(
                aec_mic,
                options.assist,
                &mut assist_branches,
                &mut assist_tees,
            );

            let meeting: Box<dyn AudioSource> =
                Box::new(ChannelSource::new(meeting_rx, system_info));
            let meeting = tap_for_assist(
                meeting,
                options.assist,
                &mut assist_branches,
                &mut assist_tees,
            );

            // A shareable local batch engine runs BOTH streams through one model — half the RAM, one
            // rolling context, one unified speaker space. Streaming/Apple/cloud engines can't share a
            // transcriber, so they stay one session per stream.
            match live_engine.shared_local_target() {
                Some((descriptor, dir)) => sessions.push(
                    spawn_shared_local_session(
                        &app,
                        descriptor,
                        dir,
                        vec![
                            (aec_mic, AudioSourceKind::Microphone),
                            (meeting, AudioSourceKind::System),
                        ],
                        Some(dedup),
                        &settings,
                    )
                    .map_err(|e| e.to_string())?,
                ),
                None => {
                    sessions.push(
                        spawn_session(
                            &app,
                            &live_engine,
                            aec_mic,
                            AudioSourceKind::Microphone,
                            Some(Arc::clone(&dedup)),
                            &settings,
                        )
                        .map_err(|e| e.to_string())?,
                    );
                    sessions.push(
                        spawn_session(
                            &app,
                            &live_engine,
                            meeting,
                            AudioSourceKind::System,
                            Some(dedup),
                            &settings,
                        )
                        .map_err(|e| e.to_string())?,
                    );
                }
            }

            *state
                .tee
                .lock()
                .map_err(|_| "state lock poisoned".to_owned())? = Some(tee_handle);
        }

        // Mic only → no playback to echo; capture it directly.
        (Some(mic), None) => {
            let mic = tap_for_assist(mic, options.assist, &mut assist_branches, &mut assist_tees);
            sessions.push(
                spawn_session(
                    &app,
                    &live_engine,
                    mic,
                    AudioSourceKind::Microphone,
                    None,
                    &settings,
                )
                .map_err(|e| e.to_string())?,
            );
        }

        // System only → clean digital capture, no echo path; transcribe it directly.
        (None, Some(system)) => {
            let system = tap_for_assist(
                system,
                options.assist,
                &mut assist_branches,
                &mut assist_tees,
            );
            sessions.push(
                spawn_session(
                    &app,
                    &live_engine,
                    system,
                    AudioSourceKind::System,
                    None,
                    &settings,
                )
                .map_err(|e| e.to_string())?,
            );
        }

        (None, None) => {
            // If system audio was wanted but failed and the mic is off, surface that reason — it's
            // more actionable than a generic "no source" message.
            return Err(degraded_notice.unwrap_or_else(|| {
                "no audio source — enable the microphone or select meeting audio".to_owned()
            }));
        }
    }

    // Mirror the live audio (post-AEC mic + system) into a single mono source the realtime AI
    // assist can subscribe to on demand. The tee is drop-oldest, so an idle assist branch never
    // backs up transcription; the mix is built lazily here and consumed by `start_assist_realtime`.
    store_assist_mix(&state, assist_branches, assist_tees)?;

    Ok(degraded_notice)
}

/// Combines the tapped live branches into one mono [`MixSource`] and parks it (plus the tee handles
/// that keep the taps alive) on [`AppState`] for the realtime assist to pick up. With no branches
/// (no source started) it clears any stale mix instead.
fn store_assist_mix(
    state: &AppState,
    mut branches: Vec<FrameReceiver>,
    tees: Vec<Tee>,
) -> Result<(), String> {
    let mix: Option<Box<dyn AudioSource>> = if branches.is_empty() {
        None
    } else {
        let primary = branches.remove(0);
        let secondary = if branches.is_empty() {
            None
        } else {
            Some(branches.remove(0))
        };
        let info = AudioSourceInfo {
            kind: AudioSourceKind::Microphone,
            name: "Assist mix".to_owned(),
        };

        Some(Box::new(MixSource {
            primary,
            primary_rs: Resampler::new(ASSIST_MIX_RATE),
            secondary,
            secondary_rs: Resampler::new(ASSIST_MIX_RATE),
            secondary_buf: VecDeque::new(),
            mixer: MeetingMixer::new(),
            info,
        }))
    };

    *state
        .assist_audio
        .lock()
        .map_err(|_| "state lock poisoned".to_owned())? = mix;
    *state
        .assist_tees
        .lock()
        .map_err(|_| "state lock poisoned".to_owned())? = tees;

    Ok(())
}

/// Stops the live session off the main thread — joining the capture/transcription threads (which can
/// take a moment to drain) on the UI thread would freeze the app, so run it on a blocking worker.
#[tauri::command]
async fn stop_session(app: AppHandle) -> Result<(), String> {
    tauri::async_runtime::spawn_blocking(move || stop_session_blocking(app))
        .await
        .map_err(|e| format!("session stop task failed: {e}"))?
}

/// How long [`stop_session_blocking`] waits for the capture/assist threads to join before detaching
/// them. State is cleared first, so on timeout the UI returns to idle and a new Start works at once
/// while the stuck thread self-cleans when its OS resource unblocks.
const STOP_TEARDOWN_TIMEOUT: Duration = Duration::from_secs(8);

/// The live resources lifted out of [`AppState`] for teardown — joined off the Stop command so a
/// wedged native handle (a stuck ScreenCaptureKit pump, a blocked engine drop) can never hang it.
struct LiveTeardown {
    worker: Option<AssistWorker>,
    assist_tees: Vec<Tee>,
    assist_audio: Option<Box<dyn AudioSource>>,
    tee: Option<Tee>,
    sessions: Vec<Session>,
}

/// The blocking body of [`stop_session`]: lift every live resource out of state (instant), then join
/// its threads under a deadline so Stop can never hang.
fn stop_session_blocking(app: AppHandle) -> Result<(), String> {
    let state = app.state::<AppState>();

    let teardown = take_live_teardown(&state)?;

    // Join the capture/assist threads off the command, bounded. If a native teardown wedges, detach
    // it (it self-cleans when the OS resource unblocks) rather than hang Stop — state is already
    // clear, so the app stays fully usable and a new Start works immediately.
    match run_within(STOP_TEARDOWN_TIMEOUT, move || teardown_session(teardown)) {
        Some(result) => result,
        None => {
            eprintln!(
                "wisp: session teardown exceeded {STOP_TEARDOWN_TIMEOUT:?} — detaching (state already cleared)"
            );
            Ok(())
        }
    }
}

/// Lifts the running session's resources out of [`AppState`], leaving it empty — fast and
/// non-blocking (quick lock + take/`None`; dropping the finals sender just closes its channel), so
/// the session is "gone" the instant this returns and a wedged teardown can't collide with a Start.
fn take_live_teardown(state: &AppState) -> Result<LiveTeardown, String> {
    let sessions = std::mem::take(
        &mut *state
            .sessions
            .lock()
            .map_err(|_| "state lock poisoned".to_owned())?,
    );
    // Signal every session to stop up front — non-blocking, so even if the bounded join below wedges
    // on a stuck device and detaches, the sessions still wind down and stop emitting segments rather
    // than lingering as a second live transcriber that bleeds into the next session's feed.
    for session in &sessions {
        session.signal_stop();
    }

    // Stop routing finals first so the sink doesn't push into a worker that's about to be joined.
    *state
        .assist_finals_tx
        .lock()
        .map_err(|_| "state lock poisoned".to_owned())? = None;
    let assist_tees = std::mem::take(
        &mut *state
            .assist_tees
            .lock()
            .map_err(|_| "state lock poisoned".to_owned())?,
    );
    let assist_audio = state
        .assist_audio
        .lock()
        .map_err(|_| "state lock poisoned".to_owned())?
        .take();
    let worker = take_assist_worker(state)?;
    let tee = state
        .tee
        .lock()
        .map_err(|_| "state lock poisoned".to_owned())?
        .take();

    Ok(LiveTeardown {
        worker,
        assist_tees,
        assist_audio,
        tee,
        sessions,
    })
}

/// Joins the live session's threads in dependency order, so no join waits on a channel a later step
/// would have closed: close the assist taps (the worker's mixed input ends), join the assist worker,
/// drop the system-audio tee (capture stops and the system session's source channel closes), then
/// stop each transcription session. Returns the first session error, if any.
fn teardown_session(t: LiveTeardown) -> Result<(), String> {
    let LiveTeardown {
        worker,
        assist_tees,
        assist_audio,
        tee,
        sessions,
    } = t;

    // Close the assist taps + drop any parked mix BEFORE joining the worker: it blocks on its mixed
    // `recv` fed by these taps, so closing them lets it observe end-of-stream and exit even if the
    // capture device wedged — otherwise the join could hang on a recv nothing satisfies.
    drop(assist_tees);
    drop(assist_audio);
    if let Some(worker) = worker {
        let _ = worker.stop_join();
    }

    // Dropping the tee stops the capture pump and closes the system session's source channel, so its
    // blocking `recv` returns `None`. Joining the sessions before this would deadlock on that recv.
    drop(tee);

    let mut last_error = None;
    for session in sessions {
        if let Err(e) = session.stop() {
            last_error = Some(e.to_string());
        }
    }

    match last_error {
        Some(e) => Err(e),
        None => Ok(()),
    }
}

/// How often the realtime assist may volunteer an ambient reply, when there's new speech since the
/// last one — so it rides along the conversation without firing on every utterance (noisy + costly).
const ASSIST_THROTTLE: std::time::Duration = std::time::Duration::from_secs(15);

/// If a triggered reply never completes within this long (stalled stream), stop waiting on it so the
/// cadence resumes — a safety valve, not the normal path (replies finish in a second or two).
const ASSIST_RESPONSE_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(30);

/// A running realtime AI-assist worker: the stop flag its loop polls, a one-shot flag the UI sets to
/// pull a reply on demand, and the thread handle — which *returns the mix source* on join, so the
/// assist can be stopped and restarted within one live session without re-tapping the capture.
struct AssistWorker {
    stop: Arc<std::sync::atomic::AtomicBool>,
    hint: Arc<std::sync::atomic::AtomicBool>,
    handle: std::thread::JoinHandle<Box<dyn AudioSource>>,
}

impl AssistWorker {
    /// Signals the loop to stop, joins the thread, and hands back the mix source it owned — `None`
    /// only if the thread panicked (then the source is gone with it).
    fn stop_join(self) -> Option<Box<dyn AudioSource>> {
        self.stop.store(true, std::sync::atomic::Ordering::Relaxed);
        self.handle.join().ok()
    }
}

/// One line of authoritative, speaker-attributed text for the realtime assist — the mic is "Me", the
/// system side is "Them" plus the live diarizer's 1-based speaker number when known. Mirrors the
/// frontend's on-screen attribution so the injected anchor reads the same way the user sees it.
fn assist_final_text(segment: &TranscriptSegment) -> String {
    let who = match segment.source {
        AudioSourceKind::Microphone => "Me".to_owned(),
        AudioSourceKind::System => match segment.speaker {
            Some(id) => format!("Them (Speaker {})", id.0 + 1),
            None => "Them".to_owned(),
        },
        _ => "Speaker".to_owned(),
    };

    format!("{who}: {}", segment.text)
}

/// Routes one diarized final into the running realtime assist, if any — best-effort (a closed channel
/// just means the assist stopped). Called from the live sink for every admitted final.
fn route_assist_final(app: &AppHandle, segment: &TranscriptSegment) {
    let state = app.state::<AppState>();
    let Ok(guard) = state.assist_finals_tx.lock() else {
        return;
    };

    if let Some(tx) = guard.as_ref() {
        let _ = tx.send(assist_final_text(segment));
    }
}

/// Retains one committed live final for export (the meeting transcript), accumulated across both the
/// mic and system streams. Best-effort: a poisoned lock just drops it from the export, never the feed.
fn retain_live_segment(app: &AppHandle, segment: &TranscriptSegment) {
    if let Ok(mut segments) = app.state::<AppState>().live_segments.lock() {
        segments.push(segment.clone());
    }
}

/// Takes the running assist worker out of state, if any — the caller stops + joins it.
fn take_assist_worker(state: &AppState) -> Result<Option<AssistWorker>, String> {
    Ok(state
        .assist_worker
        .lock()
        .map_err(|_| "state lock poisoned".to_owned())?
        .take())
}

/// The on-device API key for cloud `provider`, or an error naming the provider if none is saved.
fn cloud_key(state: &AppState, provider: &str) -> Result<String, String> {
    state
        .cloud_keys
        .lock()
        .map_err(|_| "state lock poisoned".to_owned())?
        .get(provider)
        .filter(|k| !k.trim().is_empty())
        .cloned()
        .ok_or_else(|| format!("no API key saved for {provider}"))
}

/// The realtime assist loop — the hybrid, controlled-cadence engine room. Each iteration:
/// 1. injects any new diarized finals as authoritative text context (`engine.inject_text`),
/// 2. feeds one audio frame (prosody + low latency) and drains any in-flight reply,
/// 3. decides whether to trigger a reply now — on a manual pull, or throttled when there's new speech.
///
/// Replies fire on `response.create` (the session is configured `create_response:false`), so the
/// model answers on this cadence instead of every utterance. Exits when stopped or when the capture
/// closes (the live session ended), returning the source so the assist can restart.
fn spawn_assist_worker(
    app: AppHandle,
    mut engine: Box<dyn StreamingAsrEngine>,
    mut source: Box<dyn AudioSource>,
    finals_rx: std::sync::mpsc::Receiver<String>,
) -> AssistWorker {
    use std::sync::atomic::{AtomicBool, Ordering};

    let stop = Arc::new(AtomicBool::new(false));
    let hint = Arc::new(AtomicBool::new(false));
    let stop_worker = Arc::clone(&stop);
    let hint_worker = Arc::clone(&hint);

    let handle = std::thread::spawn(move || {
        let mut last_trigger = std::time::Instant::now();
        let mut new_speech = false; // a final arrived since the last reply → worth a throttled reply
        let mut awaiting = false; // a reply is in flight (don't stack another)
        let mut awaiting_since = std::time::Instant::now();
        let mut streamed = 0usize; // chars of the in-progress reply already streamed to the UI

        while !stop_worker.load(Ordering::Relaxed) {
            // 1. Inject every authoritative final waiting — the diarized anchor the model trusts.
            while let Ok(text) = finals_rx.try_recv() {
                engine.inject_text(&text);
                new_speech = true;
            }

            // 2. Feed audio + drain any reply. Continuous capture means a frame is always close behind,
            // so the stop flag is honoured within one frame; `None` = the live tee was dropped → stop.
            let frame = match source.next_frame() {
                Ok(Some(frame)) => frame,
                Ok(None) => break,
                Err(_) => break,
            };
            let result = engine.accept_waveform(frame.sample_rate, &frame.samples);
            if result.is_endpoint {
                let text = result.text.trim();
                if !text.is_empty() {
                    let _ = app.emit(ASSIST_TEXT_EVENT, text.to_owned());
                }
                streamed = 0; // the reply closed; the next one starts a fresh stream
                awaiting = false; // the reply completed
            } else {
                // Stream the in-progress reply as it grows: emit only the new suffix (char-safe for CJK),
                // so the feed fills token-by-token instead of waiting for the whole reply.
                let total = result.text.chars().count();
                if total > streamed {
                    let delta: String = result.text.chars().skip(streamed).collect();
                    let _ = app.emit(ASSIST_DELTA_EVENT, delta);
                    streamed = total;
                }
            }

            // A stalled reply must not wedge the cadence forever.
            if awaiting && awaiting_since.elapsed() >= ASSIST_RESPONSE_TIMEOUT {
                awaiting = false;
            }

            // 3. Trigger a reply only when idle: a manual pull always fires; otherwise throttle, and
            // only when there's been new speech since the last one (never reply to silence).
            if !awaiting {
                let manual = hint_worker.swap(false, Ordering::Relaxed);
                let throttled = new_speech && last_trigger.elapsed() >= ASSIST_THROTTLE;
                if manual || throttled {
                    engine.request_response();
                    last_trigger = std::time::Instant::now();
                    awaiting = true;
                    awaiting_since = last_trigger;
                    new_speech = false;
                }
            }
        }

        source
    });

    AssistWorker { stop, hint, handle }
}

/// Starts the realtime AI assist over the live session's audio. Off the main thread — the WebSocket
/// handshake blocks ~1-2s, which would freeze the UI (the Connecting spinner can't animate).
#[tauri::command]
async fn start_assist_realtime(
    app: AppHandle,
    provider: String,
    model: String,
    instructions: String,
    params: HashMap<String, serde_json::Value>,
) -> Result<(), String> {
    tauri::async_runtime::spawn_blocking(move || {
        start_assist_realtime_blocking(app, provider, model, instructions, params)
    })
    .await
    .map_err(|e| format!("assist start task failed: {e}"))?
}

/// Connects an OpenAI realtime `model` that listens to the same mic+system mix as transcription and
/// answers each turn under `instructions` (the user's assist prompt). Requires a live session (for the
/// audio to tap) and that the assist isn't already running. Surfaces a missing key / bad model / failed
/// handshake as an error; on a build failure the mix source is restored so a retry can run.
fn start_assist_realtime_blocking(
    app: AppHandle,
    provider: String,
    model: String,
    instructions: String,
    params: HashMap<String, serde_json::Value>,
) -> Result<(), String> {
    let state = app.state::<AppState>();

    if assist_worker_running(&state)? {
        return Err("the realtime assist is already running".to_owned());
    }

    // Realtime assist speaks OpenAI's `response.output_text` protocol; other providers' live APIs
    // differ, so they aren't wired yet (a chat model runs the polling assist instead).
    if provider != "openai" {
        return Err("realtime assist currently supports OpenAI realtime models".to_owned());
    }

    let key = cloud_key(&state, &provider)?;

    // The model's full instruction is exactly what the user sees + edits in the assist prompt — no
    // hidden backend preamble. (The frontend's realtime prompt carries the grounding/anti-conversational
    // rules, visibly, so every detail is the user's to read and tune.)
    let source = state
        .assist_audio
        .lock()
        .map_err(|_| "state lock poisoned".to_owned())?
        .take()
        .ok_or("start a live session before the realtime assist")?;

    let app_err = app.clone();
    let on_error: Box<dyn Fn(&str) + Send> = Box::new(move |msg: &str| {
        let _ = app_err.emit(ASSIST_ERROR_EVENT, msg.to_owned());
    });

    let assist_params = build_param_values(&assist_realtime_param_specs(), &params);
    let engine = match build_assist_engine(&model, &key, &instructions, &assist_params, on_error) {
        Ok(engine) => engine,
        Err(e) => {
            *state
                .assist_audio
                .lock()
                .map_err(|_| "state lock poisoned".to_owned())? = Some(source);
            return Err(e.to_string());
        }
    };

    // Open the finals channel so the live sink starts feeding the assist its diarized anchors.
    let (finals_tx, finals_rx) = std::sync::mpsc::channel::<String>();
    *state
        .assist_finals_tx
        .lock()
        .map_err(|_| "state lock poisoned".to_owned())? = Some(finals_tx);

    let worker = spawn_assist_worker(app.clone(), engine, source, finals_rx);
    *state
        .assist_worker
        .lock()
        .map_err(|_| "state lock poisoned".to_owned())? = Some(worker);

    Ok(())
}

/// Pulls a realtime-assist reply on demand — the "give me a hint now" button. Sets the worker's
/// one-shot flag, which its loop consumes on the next frame. No-op error if the assist isn't running.
#[tauri::command]
fn assist_hint_now(state: State<'_, AppState>) -> Result<(), String> {
    let guard = state
        .assist_worker
        .lock()
        .map_err(|_| "state lock poisoned".to_owned())?;

    match guard.as_ref() {
        Some(worker) => {
            worker
                .hint
                .store(true, std::sync::atomic::Ordering::Relaxed);
            Ok(())
        }
        None => Err("the realtime assist isn't running".to_owned()),
    }
}

/// Whether an assist worker is currently running (a peek that doesn't take it).
fn assist_worker_running(state: &AppState) -> Result<bool, String> {
    Ok(state
        .assist_worker
        .lock()
        .map_err(|_| "state lock poisoned".to_owned())?
        .is_some())
}

/// Stops the realtime AI assist and restores the mix source so it can be started again within the same
/// live session. No-op when the assist isn't running. Off the main thread — the join may take a moment.
#[tauri::command]
async fn stop_assist_realtime(app: AppHandle) -> Result<(), String> {
    tauri::async_runtime::spawn_blocking(move || stop_assist_realtime_blocking(app))
        .await
        .map_err(|e| format!("assist stop task failed: {e}"))?
}

fn stop_assist_realtime_blocking(app: AppHandle) -> Result<(), String> {
    let state = app.state::<AppState>();

    // Stop the live sink from routing finals before joining the worker (it's about to be gone).
    *state
        .assist_finals_tx
        .lock()
        .map_err(|_| "state lock poisoned".to_owned())? = None;

    if let Some(worker) = take_assist_worker(&state)? {
        // Restore the source so the assist can restart while the session is still live (a session-level
        // Stop clears it separately).
        if let Some(source) = worker.stop_join() {
            *state
                .assist_audio
                .lock()
                .map_err(|_| "state lock poisoned".to_owned())? = Some(source);
        }
    }

    Ok(())
}

/// The transcription backend chosen for a file: the active on-device model, or a cloud
/// provider/model. Resolved before the blocking work so a missing model or key fails fast, then
/// carried into the worker. The surrounding local steps (denoise, gating, diarization) are the same
/// for both — only the engine that turns audio into text differs.
enum FileEngine {
    Local {
        descriptor: ModelDescriptor,
        dir: PathBuf,
    },
    Cloud {
        provider: CloudProvider,
        model: String,
        key: String,
        params: ParamValues,
    },
}

/// Transcribes an audio/video file at `path` with the active model, prioritising accuracy: the
/// whole clip is decoded and run through the engine's native long-form path (no VAD chunking).
/// `timestamps` requests per-segment timings (for SRT/VTT); pass `false` for the most accurate
/// plain-text result. Emits `file://meta`, then the segments and `file://done`; segments are
/// retained for export.
#[tauri::command]
async fn transcribe_file(
    app: AppHandle,
    state: State<'_, AppState>,
    path: String,
    options: FileTranscribeOptions,
) -> Result<(), String> {
    // Reject a second file transcription while one runs (they'd clobber `file_segments`); the guard
    // clears the flag on every return path. Arm a fresh cancel flag for this run.
    struct FileBusyGuard(Arc<AtomicBool>);
    impl Drop for FileBusyGuard {
        fn drop(&mut self) {
            self.0.store(false, Ordering::SeqCst);
        }
    }
    if state.file_busy.swap(true, Ordering::SeqCst) {
        return Err("a file transcription is already in progress".to_owned());
    }
    let _busy = FileBusyGuard(Arc::clone(&state.file_busy));
    state.file_cancel.store(false, Ordering::Relaxed);
    let cancel = Arc::clone(&state.file_cancel);

    let language = state
        .language
        .lock()
        .map_err(|_| "state lock poisoned".to_owned())?
        .clone();

    // The engine for this file: the active on-device model, or a cloud provider/model whose API key
    // is read from local storage. Resolving here, before the blocking work, fails fast on a missing
    // model or an unsaved key.
    let file_engine = match options.engine.as_deref() {
        Some("cloud") => {
            let provider_id = options
                .cloud_provider
                .clone()
                .ok_or("no cloud provider selected")?;
            let model = options
                .cloud_model
                .clone()
                .ok_or("no cloud model selected")?;
            let key = state
                .cloud_keys
                .lock()
                .map_err(|_| "state lock poisoned".to_owned())?
                .get(&provider_id)
                .filter(|k| !k.trim().is_empty())
                .cloned()
                .ok_or_else(|| format!("no API key saved for {provider_id}"))?;

            // Resolve the provider with the user's custom model ids merged in, so a custom id
            // selected in the picker is found by the engine just like a built-in one.
            let provider = {
                let endpoints = state
                    .cloud_custom_endpoints
                    .lock()
                    .map_err(|_| "state lock poisoned".to_owned())?;
                resolve_cloud_provider(&provider_id, &endpoints)
                    .ok_or_else(|| format!("unknown cloud provider {provider_id}"))?
            };
            let provider = {
                let customs = state
                    .cloud_custom_models
                    .lock()
                    .map_err(|_| "state lock poisoned".to_owned())?;
                with_custom_models(provider, &customs)
            };

            // Resolve the vendor knobs (temperature, …) from the settings panel, and fold the shared
            // File biasing prompt into the same bag so the engine applies it to the cloud request.
            let mut params =
                build_param_values(&batch_param_specs(&provider, &model), &options.params);
            params.set("prompt", ParamValue::Text(options.prompt.clone()));

            FileEngine::Cloud {
                provider,
                model,
                key,
                params,
            }
        }
        _ => {
            let active = state
                .active
                .lock()
                .map_err(|_| "state lock poisoned".to_owned())?
                .clone()
                .ok_or("no model selected")?;
            let (descriptor, dir) = resolve_local_model(&state, &active)?;

            FileEngine::Local { descriptor, dir }
        }
    };

    // Resolve the diarization model's directory up front (it must already be downloaded); `None`
    // when speaker ID is off. Diarization needs segment timestamps to map speakers onto lines, so
    // it implies timestamps even if the user didn't ask to display them.
    let diarize_dir = match &options.diarize_model {
        Some(model) => Some(
            state
                .store
                .local_path(&ModelId(model.clone()))
                .ok_or_else(|| format!("speaker model '{model}' is not downloaded yet"))?,
        ),
        None => None,
    };
    let need_timestamps = options.timestamps || diarize_dir.is_some();

    // Resolve a model-based denoiser's directory up front (the built-in "rnnoise" needs none).
    let denoise_dir = match options.denoiser.as_deref() {
        Some(id) if id != "rnnoise" => Some(
            state
                .store
                .local_path(&ModelId(id.to_owned()))
                .ok_or_else(|| format!("denoise model '{id}' is not downloaded yet"))?,
        ),
        _ => None,
    };

    let task_app = app.clone();
    let segments =
        tauri::async_runtime::spawn_blocking(move || -> WispResult<Vec<TranscriptSegment>> {
            let mut source = MediaSource::open(Path::new(&path))?;
            let _ = task_app.emit(
                FILE_META_EVENT,
                FileMetaDto {
                    name: source.info().name,
                    total_ms: source.duration().as_millis() as u64,
                },
            );

            let _ = task_app.emit(FILE_STAGE_EVENT, "decoding");

            // Decode the whole clip to 16 kHz mono — no VAD chunking, so the engine sees the full
            // audio with cross-window context (much more accurate than per-utterance chunks).
            //
            // Pre-size to the reported duration so the buffer rarely reallocate-and-copies as it grows
            // (each doubling transiently holds ~2× the clip). Cap the *speculative* reserve so a bogus
            // duration in a malformed header can't force a huge up-front allocation — `extend` still
            // grows past the hint for genuinely long files.
            const MAX_PRESIZE_SAMPLES: usize = TARGET_SAMPLE_RATE as usize * 3600; // 1 hour
            let hint = (source.duration().as_secs_f64() * TARGET_SAMPLE_RATE as f64) as usize;
            let mut audio: Vec<f32> = Vec::with_capacity(hint.min(MAX_PRESIZE_SAMPLES));
            while let Some(frame) = source.next_frame()? {
                audio.extend_from_slice(&to_mono_16k(&frame));
            }

            // Optionally suppress background noise first, so everything downstream — level
            // normalization, VAD gating, the engine, and diarization — works on the cleaner signal.
            if let Some(id) = options.denoiser.as_deref() {
                if let Some(mut denoiser) = build_denoiser(id, denoise_dir.as_deref()) {
                    let _ = task_app.emit(FILE_STAGE_EVENT, "reducing noise");
                    audio = denoiser.denoise(&audio, TARGET_SAMPLE_RATE);
                }
            }

            // Clean up level and rumble before the engine sees the clip — quiet or hot recordings
            // decode more reliably at a consistent, healthy level. In place, so we never hold a second
            // clip-sized copy of `audio` through the (long) transcription and diarization below.
            normalize_for_asr_in_place(&mut audio, TARGET_SAMPLE_RATE);

            // Optionally drop non-speech (silence/music) so the engine can't hallucinate in the
            // gaps: transcribe the gap-free speech, then map timestamps back to the original
            // timeline. If no speech is found, fall back to the ungated clip rather than emit
            // nothing.
            let gate = if options.gate_speech {
                let gated = gate_clip(&task_app, &audio);
                (!gated.audio.is_empty()).then_some(gated)
            } else {
                None
            };
            let asr_audio: &[f32] = gate
                .as_ref()
                .map_or(audio.as_slice(), |g| g.audio.as_slice());

            let _ = task_app.emit(FILE_STAGE_EVENT, "transcribing");
            let mut engine: Box<dyn AsrEngine> = match &file_engine {
                FileEngine::Local { descriptor, dir } => build_engine(descriptor, dir, &language)?,
                FileEngine::Cloud {
                    provider,
                    model,
                    key,
                    params,
                } => Box::new(CloudEngine::new(
                    provider,
                    model,
                    key,
                    &language,
                    params.clone(),
                )?),
            };

            // Forward 0–100 decode progress to the UI, de-duplicated so we emit only on change.
            let last_pct = std::cell::Cell::new(u8::MAX);
            let progress = |pct: u8| {
                if pct != last_pct.get() {
                    last_pct.set(pct);
                    let _ = task_app.emit(FILE_PROGRESS_EVENT, pct);
                }
            };

            // whisper.cpp drives its own decode progress over one continuous long-form pass. The
            // sherpa engines decode in one opaque call, so window the clip (cut at pauses) to report
            // a percentage there too — timestamps stay in the (gated) clip timeline for the remap.
            let raw_segments = if engine.reports_clip_progress() {
                engine
                    .transcribe_clip(
                        asr_audio,
                        TARGET_SAMPLE_RATE,
                        ClipOptions::new(need_timestamps, options.accurate, &options.prompt)
                            .with_progress(&progress),
                    )?
                    .segments
            } else {
                transcribe_in_windows(
                    engine.as_mut(),
                    asr_audio,
                    TARGET_SAMPLE_RATE,
                    ClipOptions::new(need_timestamps, options.accurate, &options.prompt),
                    &cancel,
                    &progress,
                )?
            };

            // A cancel mid-transcribe stopped the window loop early; skip the (expensive) diarization,
            // per-segment emit, and store below — a cancelled run produces no result.
            if cancel.load(Ordering::Relaxed) {
                return Ok(Vec::new());
            }

            let mut collected: Vec<TranscriptSegment> = raw_segments
                .into_iter()
                .map(|mut segment| {
                    segment.source = AudioSourceKind::File;
                    segment.status = SegmentStatus::Final;
                    segment
                })
                .collect();

            // Map the gated (compressed) timestamps back onto the original clip before anything
            // time-based — diarization and display — relies on them.
            if need_timestamps {
                if let Some(gated) = &gate {
                    remap_to_original(&mut collected, gated);
                }
            }

            // Label who said what. Best-effort: if diarization fails, keep the transcript and leave
            // speakers unlabelled rather than losing the whole result. With word-level timings the
            // attribution splits any segment that spans a speaker change at the exact handover.
            if let Some(model_dir) = &diarize_dir {
                match speaker_spans(model_dir, &audio) {
                    Ok(spans) => collected = attribute_speakers_by_word(collected, &spans),
                    Err(e) => eprintln!("speaker diarization skipped: {e}"),
                }
            }

            // Assign stable ids last: a speaker split turns one segment into several, and the UI
            // keys each line by id.
            for (i, segment) in collected.iter_mut().enumerate() {
                segment.id = i as u64;
            }

            for segment in &collected {
                let _ = task_app.emit(FILE_SEGMENT_EVENT, SegmentDto::from(segment));
            }
            Ok(collected)
        })
        .await
        .map_err(|e| e.to_string())?
        .map_err(|e| e.to_string())?;

    // A cancelled run stores no partial, but still signals "done" so the UI clears its transcribing
    // state and the next run can start (the busy guard releases as this returns).
    if state.file_cancel.load(Ordering::Relaxed) {
        let _ = app.emit(FILE_DONE_EVENT, ());
        return Ok(());
    }

    *state
        .file_segments
        .lock()
        .map_err(|_| "state lock poisoned".to_owned())? = segments;
    let _ = app.emit(FILE_DONE_EVENT, ());
    Ok(())
}

/// Signals the running file transcription to stop at the next window boundary and drop its partial
/// result. A no-op when nothing is running.
#[tauri::command]
fn cancel_file_transcription(state: State<'_, AppState>) {
    state.file_cancel.store(true, Ordering::Relaxed);
}

/// Runs offline diarization over `audio`, returning the speaker spans it found.
fn speaker_spans(model_dir: &Path, audio: &[f32]) -> WispResult<Vec<SpeakerSpan>> {
    let mut diarizer = SherpaDiarizer::new(
        &model_dir.join("segmentation.onnx"),
        &model_dir.join("embedding.onnx"),
    )?;
    diarizer.diarize_clip(audio, TARGET_SAMPLE_RATE)
}

/// Builds the chosen denoiser by id, or `None` for "off"/unknown. RNNoise is the light built-in
/// option (no `model_dir`); downloadable models like GTCRN load from their downloaded directory.
fn build_denoiser(id: &str, model_dir: Option<&Path>) -> Option<Box<dyn Denoiser>> {
    match id {
        "rnnoise" => Some(Box::new(RnnoiseDenoiser::new())),
        "denoise-gtcrn" => match GtcrnDenoiser::new(&model_dir?.join("gtcrn_simple.onnx")) {
            Ok(denoiser) => Some(Box::new(denoiser)),
            Err(e) => {
                eprintln!("wisp: GTCRN denoiser load failed ({e}); leaving audio untouched");
                None
            }
        },
        other => {
            eprintln!("wisp: unknown denoiser '{other}', leaving audio untouched");
            None
        }
    }
}

/// Runs the neural VAD over the whole clip and returns its speech concatenated gap-free, with a
/// timeline back to the original — so the engine never sees the silence/music it would hallucinate
/// over while still decoding the speech as one context-carrying long-form pass.
fn gate_clip(app: &AppHandle, audio: &[f32]) -> GatedClip {
    let mut segmenter = build_segmenter(app, AudioSourceKind::File);
    let frame = Duration::from_millis(FRAME_CHUNK_MS);
    let frame_samples = (TARGET_SAMPLE_RATE as u64 * FRAME_CHUNK_MS / 1000) as usize;

    let mut utterances = Vec::new();
    for (i, chunk) in audio.chunks(frame_samples).enumerate() {
        utterances.extend(segmenter.push(chunk, frame * i as u32, frame));
    }
    utterances.extend(segmenter.flush());

    GatedClip::from_utterances(utterances)
}

/// Note metadata the frontend supplies for a Markdown export — everything the pure formatter can't
/// derive from the segments. All optional; the document degrades gracefully when fields are absent.
#[derive(Debug, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
struct MarkdownMetaInput {
    title: Option<String>,
    date: Option<String>,
    engine: Option<String>,
    language: Option<String>,
    summary: Option<String>,
}

impl From<MarkdownMetaInput> for MeetingMeta {
    fn from(m: MarkdownMetaInput) -> Self {
        MeetingMeta {
            title: m.title,
            date: m.date,
            engine: m.engine,
            language: m.language,
            summary: m.summary,
        }
    }
}

/// Writes a transcript to `dest` in `format` (`txt`/`srt`/`vtt`/`md`). `source` selects which transcript
/// — `"live"` (the meeting just captured) or `"file"` (the most recent file transcription, the default).
/// `meta` carries the meeting metadata used by the Markdown format and is ignored by the others.
#[tauri::command]
fn export_transcript(
    state: State<'_, AppState>,
    format: String,
    dest: String,
    source: Option<String>,
    meta: Option<MarkdownMetaInput>,
) -> Result<(), String> {
    let format =
        ExportFormat::from_name(&format).ok_or_else(|| format!("unknown format: {format}"))?;

    let buffer = match source.as_deref() {
        Some("live") => &state.live_segments,
        _ => &state.file_segments,
    };
    let mut segments = buffer
        .lock()
        .map_err(|_| "state lock poisoned".to_owned())?
        .clone();
    if segments.is_empty() {
        return Err("nothing to export — transcribe or record first".to_owned());
    }

    // Mic and system finals arrive interleaved by finalization time; order by start so the document
    // reads in speech order rather than arrival order.
    segments.sort_by_key(|s| s.start);

    let content = match format {
        ExportFormat::Markdown => format_markdown(&segments, &meta.unwrap_or_default().into()),
        other => format_transcript(&segments, other),
    };
    fs::write(&dest, content).map_err(|e| format!("write {dest}: {e}"))?;
    Ok(())
}

/// One stored meeting with its segments, for the Library detail view.
#[derive(Serialize)]
struct LibraryNoteDetail {
    meeting: Note,
    segments: Vec<Segment>,
}

/// Saves the just-finished transcript (`source` `"live"` by default, or `"file"`) into the meeting
/// library under a caller-supplied `id` and `started_at_ms` — the frontend owns those, mirroring
/// `export_transcript`. Re-saving the same `id` replaces the stored meeting.
#[tauri::command]
fn save_note(
    state: State<'_, AppState>,
    id: String,
    meta: MarkdownMetaInput,
    started_at_ms: i64,
    source: Option<String>,
) -> Result<(), String> {
    let buffer = match source.as_deref() {
        Some("file") => &state.file_segments,
        _ => &state.live_segments,
    };
    let mut segments = buffer
        .lock()
        .map_err(|_| "state lock poisoned".to_owned())?
        .clone();
    segments.sort_by_key(|s| s.start);

    let mut library = state
        .library
        .lock()
        .map_err(|_| "library lock poisoned".to_owned())?;
    library
        .save_note(&id, &meta.into(), started_at_ms, &segments)
        .map_err(|e| e.to_string())
}

/// Every stored meeting, newest first, for the Library list.
#[tauri::command]
fn list_library_notes(state: State<'_, AppState>) -> Result<Vec<NoteSummary>, String> {
    state
        .library
        .lock()
        .map_err(|_| "library lock poisoned".to_owned())?
        .list_notes()
        .map_err(|e| e.to_string())
}

/// One stored meeting with its segments, or `null` if it no longer exists.
#[tauri::command]
fn get_library_note(
    state: State<'_, AppState>,
    id: String,
) -> Result<Option<LibraryNoteDetail>, String> {
    let detail = state
        .library
        .lock()
        .map_err(|_| "library lock poisoned".to_owned())?
        .get_note(&id)
        .map_err(|e| e.to_string())?
        .map(|(meeting, segments)| LibraryNoteDetail { meeting, segments });
    Ok(detail)
}

/// Full-text search across stored meetings (default cap 50 hits).
#[tauri::command]
fn search_library(
    state: State<'_, AppState>,
    query: String,
    limit: Option<usize>,
) -> Result<Vec<SearchHit>, String> {
    state
        .library
        .lock()
        .map_err(|_| "library lock poisoned".to_owned())?
        .search(&query, limit.unwrap_or(50))
        .map_err(|e| e.to_string())
}

/// Deletes a stored meeting; returns whether it existed.
#[tauri::command]
fn delete_library_note(state: State<'_, AppState>, id: String) -> Result<bool, String> {
    state
        .library
        .lock()
        .map_err(|_| "library lock poisoned".to_owned())?
        .delete_note(&id)
        .map_err(|e| e.to_string())
}

/// One downloadable local embedding model the Settings picker can offer.
#[derive(Serialize)]
struct EmbeddingModelInfo {
    id: String,
    label: String,
    dim: usize,
    size_mb: u32,
}

/// The catalog of vetted local embedding models for semantic / hybrid note search.
#[tauri::command]
fn list_embedding_models() -> Vec<EmbeddingModelInfo> {
    wisp_embed::CATALOG
        .iter()
        .map(|m| EmbeddingModelInfo {
            id: m.id.to_owned(),
            label: m.label.to_owned(),
            dim: m.dim,
            size_mb: m.size_mb,
        })
        .collect()
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_dialog::init())
        .plugin(dictation::shortcut_plugin())
        .setup(|app| {
            let data_dir = app.path().app_data_dir()?;
            let active_model_path = data_dir.join("active-model");
            let cloud_keys_path = data_dir.join("cloud-keys.json");
            let cloud_keys = load_cloud_keys(&cloud_keys_path);
            let cloud_custom_models_path = data_dir.join("cloud-custom-models.json");
            let cloud_custom_models = load_cloud_custom_models(&cloud_custom_models_path);
            let cloud_custom_endpoints_path = data_dir.join("cloud-custom-endpoints.json");
            let cloud_custom_endpoints = load_cloud_endpoints(&cloud_custom_endpoints_path);
            let custom_models_dir = data_dir.join("custom-models");
            let _ = fs::create_dir_all(&custom_models_dir);
            let custom_models = load_custom_models(&custom_models_dir);
            // The store manages both ASR models and the (separate) diarization models, so it can
            // download either; only ASR models are ever the "active" transcription model.
            let asr_catalog = builtin_catalog();
            let store = Arc::new(FsModelStore::new(
                data_dir.join("models"),
                [asr_catalog.clone(), diarization_models(), denoise_models()].concat(),
                Box::new(HttpDownloader),
            ));

            // Prefer the last chosen model (if still installed), then the first installed, then the
            // model recommended for this Mac's memory.
            let persisted = fs::read_to_string(&active_model_path)
                .ok()
                .map(|s| ModelId(s.trim().to_owned()))
                .filter(|id| store.local_path(id).is_some());
            let active = persisted
                .or_else(|| {
                    asr_catalog
                        .iter()
                        .find(|d| store.local_path(&d.id).is_some())
                        .map(|d| d.id.clone())
                })
                .or_else(|| Some(recommended_default_model(&detect_machine(), &asr_catalog)));

            let _ = fs::create_dir_all(&data_dir);
            let library = Library::open(data_dir.join("library.db"))?;

            app.manage(AppState {
                store,
                sessions: Mutex::new(Vec::new()),
                dictation: Mutex::new(None),
                dictation_hotkey: Mutex::new(dictation::DEFAULT_DICTATION_HOTKEY.to_owned()),
                dictation_enabled: Mutex::new(false),
                active: Mutex::new(active),
                mic_device: Mutex::new(None),
                // Default to capturing everything: mic (you) + system audio (all scenarios).
                system_device: Mutex::new(Some(SYSTEM_CAPTURE_ID.to_owned())),
                language: Mutex::new(String::new()),
                live_denoiser: Mutex::new(None),
                live_diarize_model: Mutex::new(None),
                live_prompt: Mutex::new(String::new()),
                live_accurate: Mutex::new(false),
                tee: Mutex::new(None),
                assist_tees: Mutex::new(Vec::new()),
                assist_audio: Mutex::new(None),
                assist_worker: Mutex::new(None),
                assist_finals_tx: Mutex::new(None),
                file_segments: Mutex::new(Vec::new()),
                file_cancel: Arc::new(AtomicBool::new(false)),
                file_busy: Arc::new(AtomicBool::new(false)),
                live_segments: Mutex::new(Vec::new()),
                library: Mutex::new(library),
                mic_muted: Arc::new(AtomicBool::new(false)),
                system_muted: Arc::new(AtomicBool::new(false)),
                active_model_path,
                cloud_keys: Mutex::new(cloud_keys),
                cloud_keys_path,
                cloud_custom_models: Mutex::new(cloud_custom_models),
                cloud_custom_models_path,
                cloud_custom_endpoints: Mutex::new(cloud_custom_endpoints),
                cloud_custom_endpoints_path,
                custom_models: Mutex::new(custom_models),
                custom_models_dir,
            });
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            list_models,
            list_diarization_models,
            list_denoise_models,
            list_cloud_providers,
            set_cloud_key,
            cloud_key_set,
            add_cloud_custom_model,
            remove_cloud_custom_model,
            add_cloud_endpoint,
            update_cloud_endpoint,
            remove_cloud_endpoint,
            streaming_params,
            batch_params,
            assist_params,
            assist_realtime_params,
            run_llm_task,
            run_assist_stream,
            download_model,
            import_custom_model,
            download_coreml,
            select_model,
            remove_model,
            list_input_devices,
            system_audio_id,
            mic_off_id,
            screen_recording_authorized,
            request_screen_recording,
            open_privacy_settings,
            restart_app,
            microphone_blocked,
            session_running,
            set_language,
            set_denoise,
            set_live_diarize,
            set_live_decode,
            set_devices,
            start_session,
            stop_session,
            start_assist_realtime,
            stop_assist_realtime,
            assist_hint_now,
            transcribe_file,
            cancel_file_transcription,
            export_transcript,
            dictation::dictation_status,
            dictation::set_dictation_enabled,
            dictation::open_accessibility_settings,
            set_stream_muted,
            save_note,
            list_library_notes,
            get_library_note,
            search_library,
            delete_library_note,
            list_embedding_models
        ])
        .build(tauri::generate_context!())
        .expect("error while building tauri application")
        .run(|app_handle, event| {
            // Release audio capture on quit so an abrupt exit never leaves the CoreAudio HAL or
            // ScreenCaptureKit wedged for the next launch. Bounded (it reuses the stop teardown), so a
            // stuck native handle can't hang the quit either; a no-op when nothing is running.
            if let tauri::RunEvent::ExitRequested { .. } = event {
                let _ = stop_session_blocking(app_handle.clone());
            }
        });
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A unique temp path per test, so parallel runs don't collide on the shared temp dir.
    fn temp_path(tag: &str) -> PathBuf {
        std::env::temp_dir().join(format!("wisp-cloud-keys-{}-{tag}.json", std::process::id()))
    }

    #[test]
    fn only_downloaded_catalog_models_are_deletable() {
        // Deletable means files of ours are present on disk to reclaim. OS-provided models (no files)
        // and not-yet-downloaded ones have nothing to delete.
        assert!(model_deletable(true, true)); // downloaded catalog model
        assert!(!model_deletable(false, true)); // OS-provided (e.g. Apple Speech) — no files
        assert!(!model_deletable(true, false)); // catalog model, not downloaded yet
        assert!(!model_deletable(false, false));
    }

    #[test]
    fn cloud_live_protocol_realtime_only_when_provider_and_model_both_stream() {
        let model = |id: &str, streaming: bool, batch: bool| CloudModel {
            id: id.to_owned(),
            display_name: id.to_owned(),
            streaming,
            batch,
            languages: vec![],
            description: String::new(),
            recommended: true,
            diarizes: false,
        };
        let provider = |streaming, models| CloudProvider {
            id: "p".to_owned(),
            display_name: "P".to_owned(),
            protocol: CloudProtocol::OpenAi,
            base_url: "https://x/v1".to_owned(),
            keys_url: String::new(),
            auth: CloudAuth::bearer(),
            streaming,
            models,
        };

        // Realtime provider + realtime-capable model → use the socket.
        let rt = provider(
            Some(StreamingProtocol::OpenAiRealtime),
            vec![model("rt", true, false)],
        );
        assert_eq!(
            cloud_live_protocol(&rt, "rt"),
            Some(StreamingProtocol::OpenAiRealtime)
        );

        // Realtime provider but a batch-only model → segment-batch (None).
        let mixed = provider(
            Some(StreamingProtocol::OpenAiRealtime),
            vec![model("b", false, true)],
        );
        assert_eq!(cloud_live_protocol(&mixed, "b"), None);

        // Unknown model id → no realtime.
        assert_eq!(cloud_live_protocol(&rt, "ghost"), None);

        // The real custom-endpoint construction is batch-only, so it always runs segment-batch.
        let custom = CustomCloudEndpoint {
            id: "custom-x".to_owned(),
            name: "X".to_owned(),
            base_url: "http://h:1/v1".to_owned(),
            protocol: "openai".to_owned(),
            model: "metis".to_owned(),
            assist: AssistParams::default(),
        }
        .to_provider();
        assert_eq!(cloud_live_protocol(&custom, "metis"), None);
    }

    #[test]
    fn assist_helpers_budget_chunk_combine_and_normalize() {
        // No window (or 0) disables map-reduce; a window reserves headroom (3 chars/token × 3/5).
        assert_eq!(input_char_budget(None), None);
        assert_eq!(input_char_budget(Some(0)), None);
        assert_eq!(input_char_budget(Some(1000)), Some(1800));

        // Chunking breaks only at line boundaries, stays under budget, and never drops content.
        let text = "aaaa\nbbbb\ncccc\ndddd";
        let chunks = chunk_by_chars(text, 9);
        assert!(chunks.iter().all(|c| c.chars().count() <= 9));
        assert_eq!(chunks.join("\n"), text);

        // A single over-long line becomes its own chunk rather than being split mid-line.
        let long = "x".repeat(50);
        assert_eq!(chunk_by_chars(&long, 10).len(), 1);

        // The standing instruction is prepended, blank-safe.
        assert_eq!(combine_system("  ", "task"), "task");
        assert_eq!(
            combine_system("Reply in Cantonese.", "Summarize."),
            "Reply in Cantonese.\n\nSummarize."
        );

        // Normalize clamps ranges and drops zero token caps to "unset".
        let n = normalize_assist(AssistParams {
            temperature: Some(5.0),
            max_tokens: Some(0),
            context_tokens: Some(8000),
            top_p: Some(-1.0),
            frequency_penalty: Some(3.0),
            presence_penalty: Some(-3.0),
            system_prompt: "  hi  ".to_owned(),
        });
        assert_eq!(n.temperature, Some(2.0));
        assert_eq!(n.top_p, Some(0.0));
        assert_eq!(n.frequency_penalty, Some(2.0));
        assert_eq!(n.presence_penalty, Some(-2.0));
        assert_eq!(n.max_tokens, None);
        assert_eq!(n.context_tokens, Some(8000));
        assert_eq!(n.system_prompt, "hi");
    }

    #[test]
    fn overlay_assist_params_applies_only_set_knobs_and_clamps() {
        let base = AssistParams {
            temperature: None,
            max_tokens: None,
            context_tokens: Some(8000),
            top_p: None,
            frequency_penalty: None,
            presence_penalty: None,
            system_prompt: "persona".to_owned(),
        };

        // A knob the user set overrides; unset knobs leave the resolved value untouched.
        let mut set = ParamValues::new();
        set.set("temperature", ParamValue::Float(0.3));
        let out = overlay_assist_params(base.clone(), &set);
        assert_eq!(out.temperature, Some(0.3), "set temperature overrides");
        assert_eq!(out.top_p, None, "unset top_p stays unset");
        assert_eq!(out.presence_penalty, None, "unset penalty stays unset");
        assert_eq!(out.context_tokens, Some(8000), "untouched fields preserved");
        assert_eq!(out.system_prompt, "persona");

        // Empty overrides leave every field as resolved — so an unset temperature is still omitted.
        let untouched = overlay_assist_params(base.clone(), &ParamValues::new());
        assert_eq!(untouched.temperature, None);

        // Overrides bypass the save-time normaliser, so the overlay clamps them itself.
        let mut wild = ParamValues::new();
        wild.set("temperature", ParamValue::Float(5.0));
        wild.set("top_p", ParamValue::Float(2.0));
        wild.set("frequency_penalty", ParamValue::Float(9.0));
        wild.set("presence_penalty", ParamValue::Float(-9.0));
        wild.set("max_tokens", ParamValue::Int(512));
        let clamped = overlay_assist_params(base.clone(), &wild);
        assert_eq!(clamped.temperature, Some(2.0), "temperature clamped to 2");
        assert_eq!(clamped.top_p, Some(1.0), "top_p clamped to 1");
        assert_eq!(
            clamped.frequency_penalty,
            Some(2.0),
            "frequency clamped to 2"
        );
        assert_eq!(
            clamped.presence_penalty,
            Some(-2.0),
            "presence clamped to -2"
        );
        assert_eq!(clamped.max_tokens, Some(512));

        // A max_tokens of 0 is the neutral default ("no cap") → treated as unset.
        let mut zero = ParamValues::new();
        zero.set("max_tokens", ParamValue::Int(0));
        assert_eq!(overlay_assist_params(base, &zero).max_tokens, None);
    }

    #[test]
    fn load_cloud_keys_is_empty_for_missing_or_garbage_files() {
        let missing = temp_path("missing");
        let _ = fs::remove_file(&missing);
        assert!(
            load_cloud_keys(&missing).is_empty(),
            "absent file → no keys"
        );

        let garbage = temp_path("garbage");
        fs::write(&garbage, b"not json at all").unwrap();
        assert!(
            load_cloud_keys(&garbage).is_empty(),
            "unparsable file → no keys, no panic"
        );
        let _ = fs::remove_file(&garbage);
    }

    #[test]
    fn save_then_load_round_trips_keys() {
        let path = temp_path("roundtrip");
        let _ = fs::remove_file(&path);

        let mut keys = HashMap::new();
        keys.insert("openai".to_owned(), "sk-test-123".to_owned());
        keys.insert("groq".to_owned(), "gsk-test-456".to_owned());
        save_cloud_keys(&path, &keys);

        assert_eq!(
            load_cloud_keys(&path),
            keys,
            "keys survive a save/load cycle"
        );

        let _ = fs::remove_file(&path);
    }

    #[test]
    fn cloud_provider_by_id_finds_seeded_and_rejects_unknown() {
        assert!(cloud_provider_by_id("openai").is_some());
        assert!(cloud_provider_by_id("does-not-exist").is_none());
    }

    #[test]
    fn with_custom_models_appends_matching_ids_and_skips_dupes() {
        let base = cloud_provider_by_id("openai").unwrap();
        let original = base.models.len();
        let existing = base.models[0].id.clone();

        let customs = vec![
            CloudCustomModel {
                provider: "openai".to_owned(),
                id: "gpt-4o-transcribe-next".to_owned(),
                name: "Next".to_owned(),
            },
            // Already in the catalog → skipped.
            CloudCustomModel {
                provider: "openai".to_owned(),
                id: existing,
                name: "dup".to_owned(),
            },
            // A different provider → not merged here.
            CloudCustomModel {
                provider: "google".to_owned(),
                id: "other".to_owned(),
                name: String::new(),
            },
            // Blank id → skipped.
            CloudCustomModel {
                provider: "openai".to_owned(),
                id: "   ".to_owned(),
                name: String::new(),
            },
        ];

        let merged = with_custom_models(base, &customs);
        assert_eq!(
            merged.models.len(),
            original + 1,
            "only the one new id is added"
        );

        let added = merged
            .model("gpt-4o-transcribe-next")
            .expect("custom id present");
        assert!(
            added.batch && !added.streaming,
            "custom cloud models are file-only"
        );
        assert_eq!(added.display_name, "Next");
        assert!(
            merged.model("other").is_none(),
            "other provider's id not merged"
        );
    }

    #[test]
    fn custom_cloud_models_round_trip_through_the_registry() {
        let dir = std::env::temp_dir().join(format!("wisp-cloud-customs-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("cloud-custom-models.json");

        assert!(
            load_cloud_custom_models(&path).is_empty(),
            "absent file → none"
        );

        let models = vec![CloudCustomModel {
            provider: "google".to_owned(),
            id: "gemini-3-flash".to_owned(),
            name: "Gemini 3 Flash".to_owned(),
        }];
        save_cloud_custom_models(&path, &models);
        assert_eq!(load_cloud_custom_models(&path), models);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn custom_endpoint_protocol_and_provider_mapping() {
        let mk = |protocol: &str| CustomCloudEndpoint {
            id: "custom-gw".to_owned(),
            name: "My Gateway".to_owned(),
            base_url: "http://host:40000/v1".to_owned(),
            protocol: protocol.to_owned(),
            model: "metis-coder".to_owned(),
            assist: AssistParams::default(),
        };
        assert_eq!(mk("chat").cloud_protocol(), CloudProtocol::OpenAiChatAudio);
        assert_eq!(mk("openai").cloud_protocol(), CloudProtocol::OpenAi);
        assert_eq!(
            mk("whatever").cloud_protocol(),
            CloudProtocol::OpenAi,
            "unknown protocol → transcription shape"
        );

        let p = mk("chat").to_provider();
        assert_eq!(p.id, "custom-gw");
        assert_eq!(p.base_url, "http://host:40000/v1");
        assert!(p.streaming.is_none(), "custom endpoints are file-only");
        let m = p.model("metis-coder").expect("its single model");
        assert!(m.batch && !m.streaming && !m.diarizes);
    }

    #[test]
    fn resolve_cloud_provider_checks_catalog_then_endpoints() {
        let endpoints = vec![CustomCloudEndpoint {
            id: "custom-gw".to_owned(),
            name: "GW".to_owned(),
            base_url: "https://gw/v1".to_owned(),
            protocol: "openai".to_owned(),
            model: "m".to_owned(),
            assist: AssistParams::default(),
        }];
        assert_eq!(
            resolve_cloud_provider("openai", &endpoints).unwrap().id,
            "openai"
        );
        assert_eq!(
            resolve_cloud_provider("custom-gw", &endpoints)
                .unwrap()
                .base_url,
            "https://gw/v1"
        );
        assert!(resolve_cloud_provider("nope", &endpoints).is_none());
    }

    #[test]
    fn unique_endpoint_id_slugifies_and_disambiguates() {
        assert_eq!(unique_endpoint_id("My Gateway!", &[]), "custom-my-gateway");

        let existing = vec![CustomCloudEndpoint {
            id: "custom-gw".to_owned(),
            name: "GW".to_owned(),
            base_url: "https://x".to_owned(),
            protocol: "openai".to_owned(),
            model: "m".to_owned(),
            assist: AssistParams::default(),
        }];
        assert_eq!(unique_endpoint_id("GW", &existing), "custom-gw-2");
        assert_eq!(unique_endpoint_id("!!!", &[]), "custom-endpoint");
    }

    #[test]
    fn custom_endpoints_round_trip_through_the_registry() {
        let dir = std::env::temp_dir().join(format!("wisp-cloud-eps-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("cloud-custom-endpoints.json");

        assert!(load_cloud_endpoints(&path).is_empty(), "absent file → none");

        let endpoints = vec![CustomCloudEndpoint {
            id: "custom-gw".to_owned(),
            name: "GW".to_owned(),
            base_url: "http://host:40000".to_owned(),
            protocol: "chat".to_owned(),
            model: "metis-coder".to_owned(),
            assist: AssistParams::default(),
        }];
        save_cloud_endpoints(&path, &endpoints);
        assert_eq!(load_cloud_endpoints(&path), endpoints);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn clean_endpoint_fields_validates_and_normalizes() {
        // Trims fields, normalizes the protocol, and strips a trailing slash from the URL.
        let (name, url, proto, model) =
            clean_endpoint_fields("  My GW ", "http://host:40000/v1/ ", "chat", " m ").unwrap();
        assert_eq!(
            (name.as_str(), url.as_str(), proto.as_str(), model.as_str()),
            ("My GW", "http://host:40000/v1", "chat", "m")
        );

        // Unknown protocol defaults to the transcription shape.
        assert_eq!(
            clean_endpoint_fields("n", "https://x", "weird", "m")
                .unwrap()
                .2,
            "openai"
        );

        // Blank fields and a non-http(s) URL are rejected.
        assert!(clean_endpoint_fields("", "https://x", "openai", "m").is_err());
        assert!(clean_endpoint_fields("n", "  ", "openai", "m").is_err());
        assert!(clean_endpoint_fields("n", "https://x", "openai", "").is_err());
        assert!(clean_endpoint_fields("n", "ftp://x", "openai", "m").is_err());
    }

    #[test]
    fn mask_key_reveals_only_a_prefix_and_last_four() {
        // A normal key: leading "sk-" prefix + last four, nothing in between.
        assert_eq!(mask_key("sk-abcdefghijklmnop").as_deref(), Some("sk-…mnop"));
        // Blank or whitespace-only → no hint.
        assert_eq!(mask_key("   "), None);
        assert_eq!(mask_key(""), None);
        // A short key never reveals more than its last four characters.
        assert_eq!(mask_key("abcd1234").as_deref(), Some("…1234"));
        // The full secret is never present in the hint.
        let key = "sk-secretsecretsecret9999";
        assert!(!mask_key(key).unwrap().contains("secret"));
    }

    #[test]
    fn is_ggml_accepts_whisper_magics_and_rejects_others() {
        assert!(is_ggml(b"GGUF\x00\x00"));
        assert!(is_ggml(b"ggml...."));
        assert!(is_ggml(b"lmgg....")); // ggml magic as a little-endian u32 on disk
        assert!(!is_ggml(b"RIFF")); // a WAV, not a model
        assert!(!is_ggml(b"\x00\x00\x00\x00"));
        assert!(!is_ggml(b"GG")); // too short to carry a magic
    }

    #[test]
    fn unique_custom_id_slugs_and_disambiguates() {
        let existing = vec![CustomModel {
            id: "custom-my-model".to_owned(),
            name: "My Model".to_owned(),
            kind: "whisper-cpp".to_owned(),
            files: vec![],
        }];
        // Sanitised to a filesystem-safe slug (non-alphanumerics collapse, edges trimmed).
        assert_eq!(unique_custom_id("My Model!", &[]), "custom-my-model");
        // Collides with the existing id → suffixed.
        assert_eq!(unique_custom_id("My Model", &existing), "custom-my-model-2");
    }

    #[test]
    fn custom_models_round_trip_through_the_registry() {
        let dir = std::env::temp_dir().join(format!("wisp-custom-{}", std::process::id()));
        let _ = fs::create_dir_all(&dir);
        assert!(
            load_custom_models(&dir).is_empty(),
            "absent registry → none"
        );

        let models = vec![CustomModel {
            id: "custom-x".to_owned(),
            name: "X".to_owned(),
            kind: "whisper-cpp".to_owned(),
            files: vec!["ggml-x.bin".to_owned()],
        }];
        save_custom_models(&dir, &models);
        assert_eq!(load_custom_models(&dir), models);

        let _ = fs::remove_dir_all(&dir);
    }
}
