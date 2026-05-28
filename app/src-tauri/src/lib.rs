//! Wisp desktop app — a thin Tauri shell over `wisp_pipeline::Session`.
//!
//! `start_session` builds a microphone-driven pipeline using the sherpa-onnx SenseVoice engine
//! and forwards transcript segments to the webview; `stop_session` stops it.

use std::sync::Mutex;

use serde::Serialize;
use tauri::{AppHandle, Emitter, Manager, State};

use wisp_audio::MicSource;
use wisp_core::transcript::{AudioSourceKind, SegmentStatus, TranscriptEvent, TranscriptSegment};
use wisp_engine_sherpa::SenseVoiceEngine;
use wisp_pipeline::{EnergyVad, Pipeline, Session, DEFAULT_SILENCE_HANGOVER};

/// Event channel the UI listens on for transcript segments.
const SEGMENT_EVENT: &str = "transcript://segment";

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

/// Shared application state: the currently running session, if any.
#[derive(Default)]
struct AppState {
    session: Mutex<Option<Session>>,
}

#[tauri::command]
fn start_session(app: AppHandle, state: State<'_, AppState>) -> Result<(), String> {
    let mut guard = state.session.lock().map_err(|_| "state lock poisoned".to_owned())?;
    if guard.is_some() {
        return Err("a session is already running".to_owned());
    }

    let dir = app
        .path()
        .app_data_dir()
        .map_err(|e| e.to_string())?
        .join("models")
        .join("sense-voice");
    let model = dir.join("model.int8.onnx");
    let tokens = dir.join("tokens.txt");
    if !model.is_file() || !tokens.is_file() {
        return Err(format!(
            "SenseVoice model not found in {}. Download it (see app/README) and retry.",
            dir.display()
        ));
    }

    let engine = SenseVoiceEngine::new(&model, &tokens).map_err(|e| e.to_string())?;
    let pipeline = Pipeline::new(
        Box::new(engine),
        Box::new(EnergyVad::default()),
        AudioSourceKind::Microphone,
        DEFAULT_SILENCE_HANGOVER,
    );
    let source = MicSource::from_default().map_err(|e| e.to_string())?;

    let emitter = app.clone();
    let sink: wisp_pipeline::EventSink = Box::new(move |event| {
        if let TranscriptEvent::Segment(segment) = event {
            let _ = emitter.emit(SEGMENT_EVENT, SegmentDto::from(&segment));
        }
    });

    *guard = Some(Session::spawn(pipeline, Box::new(source), sink));
    Ok(())
}

#[tauri::command]
fn stop_session(state: State<'_, AppState>) -> Result<(), String> {
    let session = state.session.lock().map_err(|_| "state lock poisoned".to_owned())?.take();
    match session {
        Some(session) => session.stop().map_err(|e| e.to_string()),
        None => Ok(()),
    }
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .manage(AppState::default())
        .invoke_handler(tauri::generate_handler![start_session, stop_session])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
