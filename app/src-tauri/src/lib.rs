//! Wisp desktop app — a thin Tauri shell over `wisp_pipeline::Session`.
//!
//! Commands: list/download/select models; list input devices and choose mic + system sources;
//! start/stop transcription. A session is spawned per audio source (microphone = "Me", system
//! loopback = meeting participants), all forwarding transcript segments to the webview.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use tauri::{path::BaseDirectory, AppHandle, Emitter, Manager, State};

use wisp_aec::WebrtcEchoCanceller;
use wisp_audio::{
    normalize_for_asr, tee, to_mono_16k, ChannelSource, EchoCancellingSource, MediaSource,
    MicSource, RnnoiseDenoiser, Tee, FRAME_CHUNK_MS, TARGET_SAMPLE_RATE,
};
use wisp_core::audio::AudioSource;
use wisp_core::dedup::CrossStreamEchoFilter;
use wisp_core::denoise::Denoiser;
use wisp_core::diarize::{attribute_speakers_by_word, ClipDiarizer, SpeakerSpan};
use wisp_core::engine::{AsrEngine, ClipOptions};
use wisp_core::error::{Result as WispResult, WispError};
use wisp_core::export::{format_transcript, ExportFormat};
use wisp_core::model::{ModelDescriptor, ModelFamily, ModelId, ModelStore};
use wisp_core::transcript::{AudioSourceKind, SegmentStatus, TranscriptEvent, TranscriptSegment};
use wisp_engine_sherpa::{
    GtcrnDenoiser, SenseVoiceEngine, SherpaDiarizer, SileroSegmenter, WhisperEngine,
};
use wisp_models::{
    builtin_catalog, denoise_models, diarization_models, FsModelStore, HttpDownloader,
};
use wisp_pipeline::{
    remap_to_original, EnergySegmenter, EnergyVad, GatedClip, Segmenter, Session, Transcriber, Vad,
    DEFAULT_SILENCE_HANGOVER,
};
use wisp_screencapture::ScreenCaptureSource;

mod permissions;

/// Event channel the UI listens on for transcript segments.
const SEGMENT_EVENT: &str = "transcript://segment";

/// Event channel the UI listens on for model-download progress.
const DOWNLOAD_PROGRESS_EVENT: &str = "download://progress";

/// Event channels for file transcription: total duration up front, decode progress (0–100), each
/// segment as it's produced, and a completion signal.
const FILE_META_EVENT: &str = "file://meta";
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
    active: Mutex<Option<ModelId>>,
    mic_device: Mutex<Option<String>>,
    system_device: Mutex<Option<String>>,
    /// Transcription language code (`zh`/`yue`/…); empty = auto-detect.
    language: Mutex<String>,
    /// The system-audio fan-out, present only while mic + system run together (AEC active).
    tee: Mutex<Option<Tee>>,
    /// Segments from the most recent file transcription, kept for export.
    file_segments: Mutex<Vec<TranscriptSegment>>,
    /// File where the active model id is persisted, so the choice survives a restart.
    active_model_path: PathBuf,
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
        other => Err(WispError::Engine(format!(
            "no engine for model family {other:?} yet"
        ))),
    }
}

/// Builds the GPU (Metal) whisper.cpp engine from a downloaded GGUF model. macOS only; elsewhere
/// this family isn't offered, so the stub just reports it.
#[cfg(target_os = "macos")]
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

#[cfg(not(target_os = "macos"))]
fn build_whisper_cpp_engine(
    _descriptor: &ModelDescriptor,
    _dir: &Path,
    _language: &str,
) -> WispResult<Box<dyn AsrEngine>> {
    Err(WispError::Engine(
        "the whisper.cpp GPU engine is only available on macOS".to_owned(),
    ))
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

/// Spawns one transcription session over `source`, tagging its segments with `kind` and
/// forwarding them to the webview.
///
/// When `dedup` is set (the mic + system case), every segment is routed through the shared
/// [`CrossStreamEchoFilter`] first, so a mic segment that echoes a recent meeting segment is
/// dropped rather than emitted.
fn spawn_session(
    app: &AppHandle,
    descriptor: &ModelDescriptor,
    dir: &Path,
    source: Box<dyn AudioSource>,
    kind: AudioSourceKind,
    dedup: Option<Arc<Mutex<CrossStreamEchoFilter>>>,
    language: &str,
) -> WispResult<Session> {
    let engine = build_engine(descriptor, dir, language)?;
    let segmenter = build_segmenter(app, kind);
    let transcriber = Transcriber::new(engine, kind);

    let emitter = app.clone();
    let sink: wisp_pipeline::EventSink = Box::new(move |event| {
        if let TranscriptEvent::Segment(segment) = event {
            let emit = match &dedup {
                Some(filter) => filter.lock().map(|mut f| f.admit(&segment)).unwrap_or(true),
                None => true,
            };
            if emit {
                let _ = emitter.emit(SEGMENT_EVENT, SegmentDto::from(&segment));
            }
        }
    });

    // Decoupled live session: capture+segmentation runs real-time; the (slow) engine runs on its
    // own thread draining complete utterances, so it never stalls capture or drops mid-sentence.
    Ok(Session::spawn_live(segmenter, transcriber, source, sink))
}

/// Maps a catalog descriptor to its UI DTO, resolving install/active status against the store.
fn to_model_info(
    d: ModelDescriptor,
    store: &FsModelStore,
    active: Option<&ModelId>,
) -> ModelInfoDto {
    let installed = store.local_path(&d.id).is_some();
    let is_active = active == Some(&d.id);
    let size_bytes = d.total_size_bytes();
    ModelInfoDto {
        id: d.id.0,
        name: d.display_name,
        size_bytes,
        languages: d.languages,
        description: d.description,
        installed,
        active: is_active,
    }
}

#[tauri::command]
fn list_models(state: State<'_, AppState>) -> Result<Vec<ModelInfoDto>, String> {
    let active = state
        .active
        .lock()
        .map_err(|_| "state lock poisoned".to_owned())?
        .clone();
    let models = state
        .store
        .available()
        .into_iter()
        .filter(|d| d.family != ModelFamily::Diarization)
        .map(|d| to_model_info(d, &state.store, active.as_ref()))
        .collect();
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
        .map(|d| to_model_info(d, &state.store, None))
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
        .map(|d| to_model_info(d, &state.store, None))
        .collect();
    Ok(models)
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

#[tauri::command]
fn select_model(state: State<'_, AppState>, id: String) -> Result<(), String> {
    let model_id = ModelId(id);
    if !state.store.available().iter().any(|d| d.id == model_id) {
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

#[tauri::command]
fn start_session(app: AppHandle, state: State<'_, AppState>) -> Result<(), String> {
    let mut sessions = state
        .sessions
        .lock()
        .map_err(|_| "state lock poisoned".to_owned())?;
    if !sessions.is_empty() {
        return Err("a session is already running".to_owned());
    }

    let active = state
        .active
        .lock()
        .map_err(|_| "state lock poisoned".to_owned())?
        .clone()
        .ok_or("no model selected")?;
    let descriptor = state
        .store
        .available()
        .into_iter()
        .find(|d| d.id == active)
        .ok_or("selected model is not in the catalog")?;
    let dir = state
        .store
        .local_path(&active)
        .ok_or_else(|| format!("model '{}' is not downloaded yet", active.as_str()))?;
    let language = state
        .language
        .lock()
        .map_err(|_| "state lock poisoned".to_owned())?
        .clone();

    // Open the audio sources first (these can fail: device missing / no mic permission).
    let mic_device = state
        .mic_device
        .lock()
        .map_err(|_| "state lock poisoned".to_owned())?
        .clone();
    let mic_source: Option<Box<dyn AudioSource>> = match mic_device {
        Some(name) if name == MIC_OFF_ID => None,
        Some(name) => Some(Box::new(
            MicSource::from_device(&name).map_err(|e| e.to_string())?,
        )),
        None => Some(Box::new(
            MicSource::from_default().map_err(|e| e.to_string())?,
        )),
    };

    let system_device = state
        .system_device
        .lock()
        .map_err(|_| "state lock poisoned".to_owned())?
        .clone();
    let system_source: Option<Box<dyn AudioSource>> = match system_device {
        Some(name) if name == SYSTEM_CAPTURE_ID => Some(Box::new(
            ScreenCaptureSource::new().map_err(|e| e.to_string())?,
        )),
        Some(name) => Some(Box::new(
            MicSource::from_device(&name).map_err(|e| e.to_string())?,
        )),
        None => None,
    };

    match (mic_source, system_source) {
        // Both on → the mic re-hears the system audio on speakers (echo). Tee the system capture
        // into the meeting pipeline and the AEC far-end reference, and clean the mic against it.
        (Some(mic), Some(system)) => {
            let system_info = system.info();
            let (tee_handle, meeting_rx, reference_rx) = tee(system);

            // Shared across both sessions: drops any residual echo AEC leaves on the mic stream.
            let dedup = Arc::new(Mutex::new(CrossStreamEchoFilter::new()));

            let canceller = Box::new(WebrtcEchoCanceller::new().map_err(|e| e.to_string())?);
            let aec_mic: Box<dyn AudioSource> =
                Box::new(EchoCancellingSource::new(mic, reference_rx, canceller));
            sessions.push(
                spawn_session(
                    &app,
                    &descriptor,
                    &dir,
                    aec_mic,
                    AudioSourceKind::Microphone,
                    Some(Arc::clone(&dedup)),
                    &language,
                )
                .map_err(|e| e.to_string())?,
            );

            let meeting: Box<dyn AudioSource> =
                Box::new(ChannelSource::new(meeting_rx, system_info));
            sessions.push(
                spawn_session(
                    &app,
                    &descriptor,
                    &dir,
                    meeting,
                    AudioSourceKind::System,
                    Some(dedup),
                    &language,
                )
                .map_err(|e| e.to_string())?,
            );

            *state
                .tee
                .lock()
                .map_err(|_| "state lock poisoned".to_owned())? = Some(tee_handle);
        }

        // Mic only → no playback to echo; capture it directly.
        (Some(mic), None) => {
            sessions.push(
                spawn_session(
                    &app,
                    &descriptor,
                    &dir,
                    mic,
                    AudioSourceKind::Microphone,
                    None,
                    &language,
                )
                .map_err(|e| e.to_string())?,
            );
        }

        // System only → clean digital capture, no echo path; transcribe it directly.
        (None, Some(system)) => {
            sessions.push(
                spawn_session(
                    &app,
                    &descriptor,
                    &dir,
                    system,
                    AudioSourceKind::System,
                    None,
                    &language,
                )
                .map_err(|e| e.to_string())?,
            );
        }

        (None, None) => {
            return Err(
                "no audio source — enable the microphone or select meeting audio".to_owned(),
            );
        }
    }
    Ok(())
}

#[tauri::command]
fn stop_session(state: State<'_, AppState>) -> Result<(), String> {
    let sessions = std::mem::take(
        &mut *state
            .sessions
            .lock()
            .map_err(|_| "state lock poisoned".to_owned())?,
    );

    let mut last_error = None;
    for session in sessions {
        if let Err(e) = session.stop() {
            last_error = Some(e.to_string());
        }
    }

    // Tear down the system-audio tee (present only when AEC was active): dropping it stops the
    // pump thread and the underlying capture.
    *state
        .tee
        .lock()
        .map_err(|_| "state lock poisoned".to_owned())? = None;

    match last_error {
        Some(e) => Err(e),
        None => Ok(()),
    }
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
    let active = state
        .active
        .lock()
        .map_err(|_| "state lock poisoned".to_owned())?
        .clone()
        .ok_or("no model selected")?;
    let descriptor = state
        .store
        .available()
        .into_iter()
        .find(|d| d.id == active)
        .ok_or("selected model is not in the catalog")?;
    let dir = state
        .store
        .local_path(&active)
        .ok_or_else(|| format!("model '{}' is not downloaded yet", active.as_str()))?;
    let language = state
        .language
        .lock()
        .map_err(|_| "state lock poisoned".to_owned())?
        .clone();

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

            // Decode the whole clip to 16 kHz mono — no VAD chunking, so the engine sees the full
            // audio with cross-window context (much more accurate than per-utterance chunks).
            let mut audio = Vec::new();
            while let Some(frame) = source.next_frame()? {
                audio.extend_from_slice(&to_mono_16k(&frame));
            }

            // Optionally suppress background noise first, so everything downstream — level
            // normalization, VAD gating, the engine, and diarization — works on the cleaner signal.
            if let Some(id) = options.denoiser.as_deref() {
                if let Some(mut denoiser) = build_denoiser(id, denoise_dir.as_deref()) {
                    audio = denoiser.denoise(&audio, TARGET_SAMPLE_RATE);
                }
            }

            // Clean up level and rumble before the engine sees the clip — quiet or hot recordings
            // decode more reliably at a consistent, healthy level.
            let audio = normalize_for_asr(&audio, TARGET_SAMPLE_RATE);

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

            let mut engine = build_engine(&descriptor, &dir, &language)?;

            // Forward the engine's 0–100 decode progress to the UI, de-duplicated so we emit only on
            // change. Engines without a native long-form path simply never report.
            let last_pct = std::cell::Cell::new(u8::MAX);
            let progress = |pct: u8| {
                if pct != last_pct.get() {
                    last_pct.set(pct);
                    let _ = task_app.emit(FILE_PROGRESS_EVENT, pct);
                }
            };
            let result = engine.transcribe_clip(
                asr_audio,
                TARGET_SAMPLE_RATE,
                ClipOptions::new(need_timestamps, options.accurate, &options.prompt)
                    .with_progress(&progress),
            )?;

            let mut collected: Vec<TranscriptSegment> = result
                .segments
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

    *state
        .file_segments
        .lock()
        .map_err(|_| "state lock poisoned".to_owned())? = segments;
    let _ = app.emit(FILE_DONE_EVENT, ());
    Ok(())
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

/// Writes the most recent file transcription to `dest` in `format` (`txt`/`srt`/`vtt`).
#[tauri::command]
fn export_transcript(
    state: State<'_, AppState>,
    format: String,
    dest: String,
) -> Result<(), String> {
    let format =
        ExportFormat::from_name(&format).ok_or_else(|| format!("unknown format: {format}"))?;
    let segments = state
        .file_segments
        .lock()
        .map_err(|_| "state lock poisoned".to_owned())?;
    if segments.is_empty() {
        return Err("nothing to export — transcribe a file first".to_owned());
    }
    let content = format_transcript(&segments, format);
    fs::write(&dest, content).map_err(|e| format!("write {dest}: {e}"))?;
    Ok(())
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_dialog::init())
        .setup(|app| {
            let data_dir = app.path().app_data_dir()?;
            let active_model_path = data_dir.join("active-model");
            // The store manages both ASR models and the (separate) diarization models, so it can
            // download either; only ASR models are ever the "active" transcription model.
            let asr_catalog = builtin_catalog();
            let store = Arc::new(FsModelStore::new(
                data_dir.join("models"),
                [asr_catalog.clone(), diarization_models(), denoise_models()].concat(),
                Box::new(HttpDownloader),
            ));

            // Prefer the last chosen model (if still installed), then the first installed, then the
            // first in the catalog.
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
                .or_else(|| asr_catalog.first().map(|d| d.id.clone()));

            app.manage(AppState {
                store,
                sessions: Mutex::new(Vec::new()),
                active: Mutex::new(active),
                mic_device: Mutex::new(None),
                // Default to capturing everything: mic (you) + system audio (all scenarios).
                system_device: Mutex::new(Some(SYSTEM_CAPTURE_ID.to_owned())),
                language: Mutex::new(String::new()),
                tee: Mutex::new(None),
                file_segments: Mutex::new(Vec::new()),
                active_model_path,
            });
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            list_models,
            list_diarization_models,
            list_denoise_models,
            download_model,
            select_model,
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
            set_devices,
            start_session,
            stop_session,
            transcribe_file,
            export_transcript
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
