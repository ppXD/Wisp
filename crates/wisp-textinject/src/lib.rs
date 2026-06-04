//! Insert text into the frontmost application, for dictation: set the clipboard to the text and
//! synthesize a paste keystroke (⌘V on macOS), then restore the previous clipboard a moment later.
//!
//! Pasting is the most reliable cross-app insertion method — robust for CJK/IME composition and long
//! text, where per-character key synthesis is flaky. macOS only today; other platforms return an error
//! until their injection path is added.

/// Inserts `text` at the cursor of the frontmost app by pasting it (clipboard + ⌘V), restoring the
/// previous clipboard contents shortly after.
///
/// macOS requires Accessibility permission for the synthesized keystroke to land; without it the OS
/// silently drops the event (no error here), so a caller should verify permission first.
#[cfg(target_os = "macos")]
pub fn paste_text(text: &str) -> Result<(), String> {
    mac::paste_text(text)
}

/// Text injection is macOS-only for now; every other platform reports it unimplemented.
#[cfg(not(target_os = "macos"))]
pub fn paste_text(_text: &str) -> Result<(), String> {
    Err("text injection is only implemented on macOS".to_owned())
}

#[cfg(target_os = "macos")]
mod mac {
    use std::thread;
    use std::time::Duration;

    use core_graphics::event::{CGEvent, CGEventFlags, CGEventTapLocation};
    use core_graphics::event_source::{CGEventSource, CGEventSourceStateID};

    /// `v` on the ANSI layout — the key code paired with ⌘ to paste.
    const KEY_V: u16 = 9;

    /// Wait before restoring the prior clipboard, so the synthesized paste reads the dictated text
    /// first. A small delay the user never notices.
    const RESTORE_DELAY: Duration = Duration::from_millis(180);

    pub(super) fn paste_text(text: &str) -> Result<(), String> {
        if text.is_empty() {
            return Ok(());
        }

        let mut clipboard =
            arboard::Clipboard::new().map_err(|e| format!("clipboard unavailable: {e}"))?;
        let previous = clipboard.get_text().ok();

        clipboard
            .set_text(text.to_owned())
            .map_err(|e| format!("set clipboard: {e}"))?;

        press_cmd_v()?;

        // Restore what the user had copied, so dictation doesn't clobber their clipboard. Best-effort
        // on its own thread after the paste has been delivered.
        if let Some(prev) = previous {
            thread::spawn(move || {
                thread::sleep(RESTORE_DELAY);
                if let Ok(mut clipboard) = arboard::Clipboard::new() {
                    let _ = clipboard.set_text(prev);
                }
            });
        }

        Ok(())
    }

    /// Synthesizes a ⌘V press + release at the HID level. Needs Accessibility permission to land.
    fn press_cmd_v() -> Result<(), String> {
        let source = CGEventSource::new(CGEventSourceStateID::CombinedSessionState)
            .map_err(|()| "could not create a CGEventSource".to_owned())?;

        for key_down in [true, false] {
            let event = CGEvent::new_keyboard_event(source.clone(), KEY_V, key_down)
                .map_err(|()| "could not create the paste keystroke".to_owned())?;
            event.set_flags(CGEventFlags::CGEventFlagCommand);
            event.post(CGEventTapLocation::HID);
        }

        Ok(())
    }
}
