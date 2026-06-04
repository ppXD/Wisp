//! Push-to-talk dictation: hold a global hotkey, speak, release — Wisp transcribes on-device (Apple
//! SpeechAnalyzer) and pastes the text into whatever app has focus.
//!
//! It reuses the streaming pipeline (mic → Apple engine → finals) with a sink that accumulates text
//! instead of driving the UI; on release the accumulated text is inserted via `wisp-textinject`. macOS
//! only — gated on Apple on-device speech (macOS 26) plus Accessibility permission for the paste.

use std::sync::{Arc, Mutex};

use serde::Serialize;
use tauri::{AppHandle, Manager, State};
use tauri_plugin_global_shortcut::{GlobalShortcutExt, Shortcut, ShortcutState};

use wisp_audio::MicSource;
use wisp_core::audio::AudioSource;
use wisp_core::transcript::{AudioSourceKind, SegmentStatus, TranscriptEvent};
use wisp_pipeline::Session;

use crate::{apple_speech_available, build_apple_speech_engine, permissions, AppState};

/// Default push-to-talk hotkey — hold to dictate, release to insert. The user can change it.
pub(crate) const DEFAULT_DICTATION_HOTKEY: &str = "CmdOrCtrl+Shift+D";

/// A running dictation: the streaming session capturing the mic, and the text it has accumulated.
pub(crate) struct Dictation {
    session: Session,
    buffer: Arc<Mutex<DictationBuffer>>,
}

/// Accumulated dictation text: committed finals plus the still-open partial. They're combined on
/// release so the last (not-yet-finalised) words aren't lost when the key comes up.
#[derive(Default)]
struct DictationBuffer {
    committed: Vec<String>,
    pending: String,
}

impl DictationBuffer {
    fn push_final(&mut self, text: &str) {
        let text = text.trim();
        if !text.is_empty() {
            self.committed.push(text.to_owned());
        }
        self.pending.clear();
    }

    fn set_partial(&mut self, text: &str) {
        self.pending = text.trim().to_owned();
    }

    /// The full dictated text — finals then the open partial — joined with the spacing rule.
    fn collect(&self) -> String {
        let mut pieces = self.committed.clone();
        let pending = self.pending.trim();
        if !pending.is_empty() {
            pieces.push(pending.to_owned());
        }
        join_pieces(&pieces)
    }
}

/// Joins transcript pieces, inserting a space only before an ASCII-word-leading piece (so spaced
/// languages read correctly while CJK runs together).
fn join_pieces(pieces: &[String]) -> String {
    let mut out = String::new();
    for piece in pieces {
        let needs_space = !out.is_empty()
            && piece
                .chars()
                .next()
                .is_some_and(|c| c.is_ascii_alphanumeric());
        if needs_space {
            out.push(' ');
        }
        out.push_str(piece);
    }
    out
}

/// Accumulates the streaming session's segments into `buffer`: finals commit, partials replace the
/// pending tail. (No UI emission — dictation's output is the paste, not the feed.)
fn dictation_sink(buffer: Arc<Mutex<DictationBuffer>>) -> wisp_pipeline::EventSink {
    Box::new(move |event| {
        if let TranscriptEvent::Segment(segment) = event {
            if let Ok(mut buffer) = buffer.lock() {
                match segment.status {
                    SegmentStatus::Final => buffer.push_final(&segment.text),
                    _ => buffer.set_partial(&segment.text),
                }
            }
        }
    })
}

/// Starts capturing on key-down: builds the Apple engine + default mic and spawns a streaming session
/// whose finals accumulate in a buffer. A no-op if already dictating (key-repeat) or a meeting is live.
fn start_dictation(app: &AppHandle) -> Result<(), String> {
    let state = app.state::<AppState>();

    if state
        .dictation
        .lock()
        .map_err(|_| "state lock poisoned".to_owned())?
        .is_some()
    {
        return Ok(());
    }
    if !state
        .sessions
        .lock()
        .map_err(|_| "state lock poisoned".to_owned())?
        .is_empty()
    {
        return Err("stop the live session before dictating".to_owned());
    }

    let language = state
        .language
        .lock()
        .map_err(|_| "state lock poisoned".to_owned())?
        .clone();

    let engine = build_apple_speech_engine(&language).map_err(|e| e.to_string())?;
    let mic: Box<dyn AudioSource> = Box::new(MicSource::from_default().map_err(|e| e.to_string())?);

    let buffer = Arc::new(Mutex::new(DictationBuffer::default()));
    let session = Session::spawn_streaming(
        engine,
        mic,
        dictation_sink(buffer.clone()),
        None,
        AudioSourceKind::Microphone,
    );

    *state
        .dictation
        .lock()
        .map_err(|_| "state lock poisoned".to_owned())? = Some(Dictation { session, buffer });
    Ok(())
}

/// Stops capturing on key-up and pastes the accumulated text into the frontmost app.
fn finish_dictation(app: &AppHandle) {
    let state = app.state::<AppState>();
    let Some(dictation) = state.dictation.lock().ok().and_then(|mut d| d.take()) else {
        return;
    };

    let _ = dictation.session.stop();
    let text = dictation
        .buffer
        .lock()
        .map(|buffer| buffer.collect())
        .unwrap_or_default();

    if !text.is_empty() {
        if let Err(e) = wisp_textinject::paste_text(&text) {
            eprintln!("wisp: dictation paste failed: {e}");
        }
    }
}

/// Routes a hotkey press/release into start/finish. Errors are non-fatal (logged) — a missed dictation
/// must never crash the app.
fn on_shortcut(app: &AppHandle, state: ShortcutState) {
    match state {
        ShortcutState::Pressed => {
            if let Err(e) = start_dictation(app) {
                eprintln!("wisp: dictation start skipped: {e}");
            }
        }
        ShortcutState::Released => finish_dictation(app),
    }
}

/// The global-shortcut plugin wired to push-to-talk dictation. Only the dictation hotkey is ever
/// registered, so any fired shortcut is the dictation key. Built for the app's `Wry` runtime.
pub(crate) fn shortcut_plugin() -> tauri::plugin::TauriPlugin<tauri::Wry> {
    tauri_plugin_global_shortcut::Builder::new()
        .with_handler(|app, _shortcut, event| on_shortcut(app, event.state()))
        .build()
}

/// Dictation availability + current config, for the settings UI.
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct DictationStatus {
    /// On-device dictation can run here (Apple speech, macOS 26+).
    available: bool,
    /// Accessibility permission is granted (needed to paste into other apps).
    accessibility_ok: bool,
    /// The hotkey is currently registered.
    enabled: bool,
    /// The configured push-to-talk hotkey.
    hotkey: String,
}

fn status(state: &AppState) -> Result<DictationStatus, String> {
    Ok(DictationStatus {
        available: apple_speech_available(),
        accessibility_ok: permissions::accessibility_authorized(),
        enabled: *state
            .dictation_enabled
            .lock()
            .map_err(|_| "state lock poisoned".to_owned())?,
        hotkey: state
            .dictation_hotkey
            .lock()
            .map_err(|_| "state lock poisoned".to_owned())?
            .clone(),
    })
}

#[tauri::command]
pub(crate) fn dictation_status(state: State<'_, AppState>) -> Result<DictationStatus, String> {
    status(&state)
}

/// Enables or disables push-to-talk dictation, optionally changing the hotkey. Registers/unregisters
/// the global shortcut accordingly and returns the updated status.
#[tauri::command]
pub(crate) fn set_dictation_enabled(
    app: AppHandle,
    enabled: bool,
    hotkey: Option<String>,
) -> Result<DictationStatus, String> {
    let state = app.state::<AppState>();

    if let Some(hotkey) = hotkey.filter(|h| !h.trim().is_empty()) {
        *state
            .dictation_hotkey
            .lock()
            .map_err(|_| "state lock poisoned".to_owned())? = hotkey;
    }
    let hotkey = state
        .dictation_hotkey
        .lock()
        .map_err(|_| "state lock poisoned".to_owned())?
        .clone();

    let shortcuts = app.global_shortcut();
    let _ = shortcuts.unregister_all();

    if enabled {
        if !apple_speech_available() {
            return Err("dictation needs Apple on-device speech (macOS 26 or newer)".to_owned());
        }
        let shortcut: Shortcut = hotkey
            .parse()
            .map_err(|_| format!("invalid hotkey: {hotkey}"))?;
        shortcuts
            .register(shortcut)
            .map_err(|e| format!("could not register the hotkey: {e}"))?;
    }

    *state
        .dictation_enabled
        .lock()
        .map_err(|_| "state lock poisoned".to_owned())? = enabled;

    status(&state)
}

/// Opens System Settings → Privacy → Accessibility so the user can grant the paste permission.
#[tauri::command]
pub(crate) fn open_accessibility_settings() -> Result<(), String> {
    permissions::open_privacy_settings("accessibility")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn join_pieces_spaces_ascii_but_runs_cjk_together() {
        assert_eq!(
            join_pieces(&["hello".to_owned(), "world".to_owned()]),
            "hello world"
        );
        assert_eq!(
            join_pieces(&["你好".to_owned(), "世界".to_owned()]),
            "你好世界"
        );
        // Mixed: no space before the CJK piece, a space before the ASCII one.
        assert_eq!(
            join_pieces(&["開會".to_owned(), "OK".to_owned()]),
            "開會 OK"
        );
    }

    #[test]
    fn buffer_combines_finals_with_the_open_partial() {
        let mut buffer = DictationBuffer::default();
        buffer.set_partial("hello wor");
        buffer.push_final("hello world");
        buffer.set_partial("how are");
        // committed "hello world" + still-open partial "how are".
        assert_eq!(buffer.collect(), "hello world how are");
    }
}
