//! Wisp desktop app — a thin Tauri shell over `wisp_pipeline::Session`.
//!
//! Commands: list/download/select models; list input devices and choose mic + system sources;
//! start/stop transcription. A session is spawned per audio source (microphone = "Me", system
//! loopback = meeting participants), all forwarding transcript segments to the webview.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use tauri::{path::BaseDirectory, AppHandle, Emitter, Manager, State};

#[cfg(target_os = "macos")]
use wisp_aec::WebrtcEchoCanceller;
use wisp_audio::{
    normalize_for_asr, tee, to_mono_16k, ChannelSource, EchoCancellingSource, MediaSource,
    MicSource, RnnoiseDenoiser, Tee, FRAME_CHUNK_MS, TARGET_SAMPLE_RATE,
};
use wisp_core::aec::{EchoCanceller, PassthroughEchoCanceller};
use wisp_core::audio::AudioSource;
use wisp_core::cloud::CloudProvider;
use wisp_core::dedup::CrossStreamEchoFilter;
use wisp_core::denoise::Denoiser;
use wisp_core::diarize::{attribute_speakers_by_word, ClipDiarizer, SpeakerSpan};
use wisp_core::engine::{AsrEngine, ClipOptions, StreamingAsrEngine};
use wisp_core::error::{Result as WispResult, WispError};
use wisp_core::export::{format_transcript, ExportFormat};
use wisp_core::model::{ModelDescriptor, ModelFamily, ModelFile, ModelId, ModelStore, Quant};
use wisp_core::transcript::{AudioSourceKind, SegmentStatus, TranscriptEvent, TranscriptSegment};
use wisp_engine_cloud::CloudEngine;
use wisp_engine_sherpa::{
    GtcrnDenoiser, SenseVoiceEngine, SherpaDiarizer, SherpaLiveDiarizer, SileroSegmenter,
    StreamingTransducerEngine, WhisperEngine,
};
#[cfg(target_os = "windows")]
use wisp_loopback::WasapiLoopbackSource;
use wisp_models::{
    builtin_catalog, cloud_catalog, coreml_asset, denoise_models, diarization_models,
    family_runnable, recommended_accurate_model, recommended_default_model, Accelerator,
    FsModelStore, GpuTier, HttpDownloader, MachineProfile,
};
use wisp_pipeline::{
    remap_to_original, transcribe_in_windows, EnergySegmenter, EnergyVad, GatedClip, Segmenter,
    Session, Transcriber, Vad, DEFAULT_SILENCE_HANGOVER,
};
#[cfg(target_os = "macos")]
use wisp_screencapture::ScreenCaptureSource;

mod permissions;

/// Event channel the UI listens on for transcript segments.
const SEGMENT_EVENT: &str = "transcript://segment";

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
    /// Segments from the most recent file transcription, kept for export.
    file_segments: Mutex<Vec<TranscriptSegment>>,
    /// File where the active model id is persisted, so the choice survives a restart.
    active_model_path: PathBuf,
    /// Per-provider cloud API keys, kept purely on-device (persisted to `cloud_keys_path`).
    cloud_keys: Mutex<HashMap<String, String>>,
    /// File the cloud API keys persist to — local app data only, never synced or sent anywhere
    /// except as the auth header to the provider the key belongs to.
    cloud_keys_path: PathBuf,
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
    settings: &LiveSettings,
) -> WispResult<Session> {
    // The per-session denoiser and the event sink (cross-stream echo dedup on finals) are the same
    // regardless of engine kind, so build them once up front.
    let denoiser = settings
        .denoiser
        .as_deref()
        .and_then(|id| build_denoiser(id, settings.denoise_dir.as_deref()));

    let emitter = app.clone();
    let sink: wisp_pipeline::EventSink = Box::new(move |event| {
        if let TranscriptEvent::Segment(segment) = event {
            // Provisional partials bypass the cross-stream echo filter — only a committed final
            // updates its dedup state (a partial priming it would make the final look like an echo,
            // and partials of the same utterance would flap in and out of the feed).
            let emit = if matches!(segment.status, SegmentStatus::Final) {
                match &dedup {
                    Some(filter) => filter.lock().map(|mut f| f.admit(&segment)).unwrap_or(true),
                    None => true,
                }
            } else {
                true
            };
            if emit {
                let _ = emitter.emit(SEGMENT_EVENT, SegmentDto::from(&segment));
            }
        }
    });

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

    let mut engine = build_engine(descriptor, dir, &settings.language)?;
    engine.configure_streaming(&settings.prompt, settings.accurate);
    let segmenter = build_segmenter(app, kind);
    let mut transcriber = Transcriber::new(engine, kind);
    // Live speaker labels: a per-session diarizer (separate numbering per stream). Best-effort —
    // if the model won't load, transcribe without labels rather than failing the session.
    if let Some(diarize_dir) = &settings.diarize_dir {
        match SherpaLiveDiarizer::new(&diarize_dir.join("embedding.onnx")) {
            Ok(diarizer) => transcriber = transcriber.with_diarizer(Box::new(diarizer)),
            Err(e) => eprintln!("wisp: live diarizer load failed ({e}); skipping speaker labels"),
        }
    }

    Ok(Session::spawn_live(
        segmenter,
        transcriber,
        source,
        sink,
        denoiser,
    ))
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

#[cfg(not(target_os = "macos"))]
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
fn to_model_info(
    d: ModelDescriptor,
    store: &FsModelStore,
    active: Option<&ModelId>,
    live_rec: Option<&ModelId>,
    file_rec: Option<&ModelId>,
) -> ModelInfoDto {
    let installed = store.local_path(&d.id).is_some();
    let is_active = active == Some(&d.id);
    let recommended_live = live_rec == Some(&d.id);
    let recommended_file = file_rec == Some(&d.id);
    let family = format!("{:?}", d.family);
    let size_bytes = d.total_size_bytes();

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
    }
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
        // Only offer models this machine can actually start — e.g. the Metal whisper.cpp models are
        // hidden off macOS, where building their engine would fail.
        .filter(|d| family_runnable(d.family, machine.accelerator))
        .map(|d| {
            to_model_info(
                d,
                &state.store,
                active.as_ref(),
                Some(&live_rec),
                Some(&file_rec),
            )
        })
        .collect();

    // Append the user's imported custom models — always installed, gated by the same runnability
    // rule (a Metal-only `.bin` stays hidden off macOS).
    let custom = state
        .custom_models
        .lock()
        .map_err(|_| "state lock poisoned".to_owned())?;
    for cm in custom.iter() {
        let Some(family) = custom_family(&cm.kind) else {
            continue;
        };
        if family_runnable(family, machine.accelerator) {
            if let Some(info) = custom_model_info(cm, active.as_ref()) {
                models.push(info);
            }
        }
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
    let providers = cloud_catalog()
        .into_iter()
        .map(|p| {
            let key_hint = keys.get(&p.id).and_then(|k| mask_key(k));

            CloudProviderDto {
                key_set: key_hint.is_some(),
                key_hint,
                keys_url: p.keys_url,
                id: p.id,
                name: p.display_name,
                models: p
                    .models
                    .into_iter()
                    .map(|m| CloudModelDto {
                        id: m.id,
                        name: m.display_name,
                        streaming: m.streaming,
                        batch: m.batch,
                        description: m.description,
                    })
                    .collect(),
            }
        })
        .collect();
    Ok(providers)
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
    })
}

/// Resolves a model id to its descriptor + directory, checking the built-in catalog first and then
/// the user's imported custom models. The single resolution path for both transcribe and live start.
fn resolve_local_model(
    state: &AppState,
    id: &ModelId,
) -> Result<(ModelDescriptor, PathBuf), String> {
    if let Some(descriptor) = state.store.available().into_iter().find(|d| d.id == *id) {
        let dir = state
            .store
            .local_path(id)
            .ok_or_else(|| format!("model '{}' is not downloaded yet", id.as_str()))?;
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

/// Starts a live session. Returns an optional non-fatal notice (e.g. system audio was unavailable
/// so it fell back to mic-only) for the UI to surface; `None` means everything started as requested.
#[tauri::command]
fn start_session(app: AppHandle, state: State<'_, AppState>) -> Result<Option<String>, String> {
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
    let (descriptor, dir) = resolve_local_model(&state, &active)?;
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

            // Echo-cancel the mic against the system reference (WebRTC on macOS, passthrough where
            // there's no platform AEC); either way the reference is consumed and the dedup mops up.
            let aec_mic: Box<dyn AudioSource> = Box::new(EchoCancellingSource::new(
                mic,
                reference_rx,
                echo_canceller(),
            ));
            sessions.push(
                spawn_session(
                    &app,
                    &descriptor,
                    &dir,
                    aec_mic,
                    AudioSourceKind::Microphone,
                    Some(Arc::clone(&dedup)),
                    &settings,
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
                    &settings,
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
                    &settings,
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
    Ok(degraded_notice)
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
        provider: String,
        model: String,
        key: String,
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
            let provider = options
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
                .get(&provider)
                .filter(|k| !k.trim().is_empty())
                .cloned()
                .ok_or_else(|| format!("no API key saved for {provider}"))?;

            FileEngine::Cloud {
                provider,
                model,
                key,
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
            let mut audio = Vec::new();
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

            let _ = task_app.emit(FILE_STAGE_EVENT, "transcribing");
            let mut engine: Box<dyn AsrEngine> = match &file_engine {
                FileEngine::Local { descriptor, dir } => build_engine(descriptor, dir, &language)?,
                FileEngine::Cloud {
                    provider,
                    model,
                    key,
                } => {
                    let provider = cloud_provider_by_id(provider).ok_or_else(|| {
                        WispError::Model(format!("unknown cloud provider {provider}"))
                    })?;
                    Box::new(CloudEngine::new(&provider, model, key, &language)?)
                }
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
                    need_timestamps,
                    options.accurate,
                    &options.prompt,
                    &progress,
                )?
            };

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
            let cloud_keys_path = data_dir.join("cloud-keys.json");
            let cloud_keys = load_cloud_keys(&cloud_keys_path);
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

            app.manage(AppState {
                store,
                sessions: Mutex::new(Vec::new()),
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
                file_segments: Mutex::new(Vec::new()),
                active_model_path,
                cloud_keys: Mutex::new(cloud_keys),
                cloud_keys_path,
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
            download_model,
            import_custom_model,
            download_coreml,
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
            set_denoise,
            set_live_diarize,
            set_live_decode,
            set_devices,
            start_session,
            stop_session,
            transcribe_file,
            export_transcript
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A unique temp path per test, so parallel runs don't collide on the shared temp dir.
    fn temp_path(tag: &str) -> PathBuf {
        std::env::temp_dir().join(format!("wisp-cloud-keys-{}-{tag}.json", std::process::id()))
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
