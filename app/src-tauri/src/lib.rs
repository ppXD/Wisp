//! Wisp desktop app — a thin Tauri shell over `wisp_pipeline::Session`.
//!
//! Commands: list/download/select models; list input devices and choose mic + system sources;
//! start/stop transcription. A session is spawned per audio source (microphone = "Me", system
//! loopback = meeting participants), all forwarding transcript segments to the webview.

use std::path::Path;
use std::sync::{Arc, Mutex};

use serde::Serialize;
use tauri::{AppHandle, Emitter, Manager, State};

use wisp_aec::WebrtcEchoCanceller;
use wisp_audio::{tee, ChannelSource, EchoCancellingSource, MicSource, Tee};
use wisp_core::audio::AudioSource;
use wisp_core::dedup::CrossStreamEchoFilter;
use wisp_core::engine::AsrEngine;
use wisp_core::error::{Result as WispResult, WispError};
use wisp_core::model::{ModelDescriptor, ModelFamily, ModelId, ModelStore};
use wisp_core::transcript::{AudioSourceKind, SegmentStatus, TranscriptEvent, TranscriptSegment};
use wisp_engine_sherpa::SenseVoiceEngine;
use wisp_models::{builtin_catalog, FsModelStore, HttpDownloader};
use wisp_pipeline::{EnergyVad, Pipeline, Session, DEFAULT_SILENCE_HANGOVER};
use wisp_screencapture::ScreenCaptureSource;

/// Event channel the UI listens on for transcript segments.
const SEGMENT_EVENT: &str = "transcript://segment";

/// Sentinel "device" id selecting one-click system-audio capture (ScreenCaptureKit, no setup).
/// Must match the value used by the UI.
const SYSTEM_CAPTURE_ID: &str = "__wisp_system_audio__";

/// Sentinel mic id meaning "no microphone" — capture system audio only. Matches the UI.
const MIC_OFF_ID: &str = "__wisp_mic_off__";

/// Shared application state.
struct AppState {
    store: Arc<FsModelStore>,
    sessions: Mutex<Vec<Session>>,
    active: Mutex<Option<ModelId>>,
    mic_device: Mutex<Option<String>>,
    system_device: Mutex<Option<String>>,
    /// The system-audio fan-out, present only while mic + system run together (AEC active).
    tee: Mutex<Option<Tee>>,
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

/// A catalog model with its install/active status, for the picker UI.
#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct ModelInfoDto {
    id: String,
    name: String,
    size_bytes: u64,
    languages: Vec<String>,
    installed: bool,
    active: bool,
}

/// Builds the ASR engine for a downloaded model based on its family.
fn build_engine(descriptor: &ModelDescriptor, dir: &Path) -> WispResult<Box<dyn AsrEngine>> {
    match descriptor.family {
        ModelFamily::SenseVoice => {
            let model_name = descriptor
                .files
                .iter()
                .find(|f| f.name.ends_with(".onnx"))
                .map(|f| f.name.clone())
                .ok_or_else(|| WispError::Model("model descriptor has no .onnx file".to_owned()))?;
            let engine = SenseVoiceEngine::new(&dir.join(model_name), &dir.join("tokens.txt"))?;
            Ok(Box::new(engine))
        }
        other => Err(WispError::Engine(format!(
            "no engine for model family {other:?} yet"
        ))),
    }
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
) -> WispResult<Session> {
    let engine = build_engine(descriptor, dir)?;
    let pipeline = Pipeline::new(
        engine,
        Box::new(EnergyVad::default()),
        kind,
        DEFAULT_SILENCE_HANGOVER,
    );

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

    Ok(Session::spawn(pipeline, source, sink))
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
        .map(|d| {
            let installed = state.store.local_path(&d.id).is_some();
            let is_active = active.as_ref() == Some(&d.id);
            let size_bytes = d.total_size_bytes();
            ModelInfoDto {
                id: d.id.0,
                name: d.display_name,
                size_bytes,
                languages: d.languages,
                installed,
                active: is_active,
            }
        })
        .collect();
    Ok(models)
}

#[tauri::command]
async fn download_model(state: State<'_, AppState>, id: String) -> Result<(), String> {
    let store = Arc::clone(&state.store);
    let model_id = ModelId(id);
    tauri::async_runtime::spawn_blocking(move || store.ensure(&model_id))
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

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .setup(|app| {
            let models_root = app.path().app_data_dir()?.join("models");
            let store = Arc::new(FsModelStore::new(
                models_root,
                builtin_catalog(),
                Box::new(HttpDownloader),
            ));

            let catalog = store.available();
            let active = catalog
                .iter()
                .find(|d| store.local_path(&d.id).is_some())
                .or_else(|| catalog.first())
                .map(|d| d.id.clone());

            app.manage(AppState {
                store,
                sessions: Mutex::new(Vec::new()),
                active: Mutex::new(active),
                mic_device: Mutex::new(None),
                // Default to capturing everything: mic (you) + system audio (all scenarios).
                system_device: Mutex::new(Some(SYSTEM_CAPTURE_ID.to_owned())),
                tee: Mutex::new(None),
            });
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            list_models,
            download_model,
            select_model,
            list_input_devices,
            system_audio_id,
            mic_off_id,
            set_devices,
            start_session,
            stop_session
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
