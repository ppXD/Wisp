//! Wisp desktop app — a thin Tauri shell over `wisp_pipeline::Session`.
//!
//! `start_session` builds a microphone-driven pipeline (with a placeholder engine so the app
//! runs without a real model) and forwards transcript segments to the webview; `stop_session`
//! stops it. The real sherpa-onnx engine is a drop-in replacement for [`PlaceholderEngine`].

use std::sync::Mutex;
use std::time::Duration;

use serde::Serialize;
use tauri::{AppHandle, Emitter, State};

use wisp_audio::{MicSource, TARGET_SAMPLE_RATE};
use wisp_core::engine::{AsrEngine, EngineInfo, TranscriptionResult};
use wisp_core::error::Result as WispResult;
use wisp_core::transcript::{AudioSourceKind, SegmentStatus, TranscriptEvent, TranscriptSegment};
use wisp_pipeline::{EnergyVad, Pipeline, Session, DEFAULT_SILENCE_HANGOVER};

/// Event channel the UI listens on for transcript segments.
const SEGMENT_EVENT: &str = "transcript://segment";

/// A stand-in ASR engine: emits one segment per utterance reporting its duration, so the whole
/// capture → pipeline → UI path runs without a model. Replaced by the sherpa engine later.
#[derive(Default)]
struct PlaceholderEngine;

impl AsrEngine for PlaceholderEngine {
    fn info(&self) -> EngineInfo {
        EngineInfo { name: "placeholder".to_owned(), streaming: false }
    }

    fn transcribe(&mut self, audio: &[f32], sample_rate: u32) -> WispResult<TranscriptionResult> {
        let rate = if sample_rate == 0 { TARGET_SAMPLE_RATE } else { sample_rate };
        let secs = audio.len() as f32 / rate as f32;
        let segment = TranscriptSegment::new(
            0,
            format!("[speech {secs:.1}s]"),
            Duration::ZERO..Duration::from_secs_f32(secs),
            AudioSourceKind::Microphone,
        );
        Ok(TranscriptionResult { segments: vec![segment] })
    }
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

    let source = MicSource::from_default().map_err(|e| e.to_string())?;
    let pipeline = Pipeline::new(
        Box::new(PlaceholderEngine),
        Box::new(EnergyVad::default()),
        AudioSourceKind::Microphone,
        DEFAULT_SILENCE_HANGOVER,
    );

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
