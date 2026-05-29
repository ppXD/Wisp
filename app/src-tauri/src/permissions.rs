//! Privacy-permission checks for the capture sources Wisp uses.
//!
//! macOS gates microphone and screen-recording access behind TCC. This exposes a proactive check
//! and a request for Screen Recording (used by system-audio capture) so the UI can prompt the user
//! up front instead of failing at start. On other platforms these are no-ops that report "granted".

/// Whether Screen Recording (required for system-audio capture) is currently authorized.
pub fn screen_recording_authorized() -> bool {
    imp::screen_recording_authorized()
}

/// Prompts for Screen Recording access. Returns whether it is granted afterwards.
///
/// macOS only prompts the first time; after a prior denial this returns `false` without a prompt
/// and the user must grant it in System Settings (see [`open_privacy_settings`]).
pub fn request_screen_recording() -> bool {
    imp::request_screen_recording()
}

/// Opens the System Settings privacy pane for `pane` ("screen" or "microphone"). macOS only.
pub fn open_privacy_settings(pane: &str) -> Result<(), String> {
    imp::open_privacy_settings(pane)
}

/// Whether microphone access is blocked (explicitly denied or restricted).
///
/// `false` for both "authorized" and "not yet determined" — in the latter case the OS prompts on
/// first capture, so there is nothing to warn about up front; only an outright block needs action.
pub fn microphone_blocked() -> bool {
    imp::microphone_blocked()
}

#[cfg(target_os = "macos")]
mod imp {
    use objc2::runtime::AnyObject;
    use objc2::{class, msg_send};

    // CoreGraphics screen-capture access APIs (macOS 10.15+).
    #[link(name = "CoreGraphics", kind = "framework")]
    extern "C" {
        fn CGPreflightScreenCaptureAccess() -> bool;
        fn CGRequestScreenCaptureAccess() -> bool;
    }

    // `AVMediaTypeAudio` constant from AVFoundation (an `NSString *`).
    #[link(name = "AVFoundation", kind = "framework")]
    extern "C" {
        static AVMediaTypeAudio: *const AnyObject;
    }

    pub(super) fn screen_recording_authorized() -> bool {
        unsafe { CGPreflightScreenCaptureAccess() }
    }

    pub(super) fn request_screen_recording() -> bool {
        unsafe { CGRequestScreenCaptureAccess() }
    }

    pub(super) fn microphone_blocked() -> bool {
        // AVAuthorizationStatus: 0 notDetermined, 1 restricted, 2 denied, 3 authorized.
        let status: isize = unsafe {
            msg_send![class!(AVCaptureDevice), authorizationStatusForMediaType: AVMediaTypeAudio]
        };
        status == 1 || status == 2
    }

    pub(super) fn open_privacy_settings(pane: &str) -> Result<(), String> {
        let anchor = match pane {
            "screen" => "Privacy_ScreenCapture",
            "microphone" => "Privacy_Microphone",
            other => return Err(format!("unknown settings pane: {other}")),
        };
        let url = format!("x-apple.systempreferences:com.apple.preference.security?{anchor}");
        std::process::Command::new("open")
            .arg(url)
            .spawn()
            .map(|_| ())
            .map_err(|e| format!("open settings: {e}"))
    }
}

#[cfg(not(target_os = "macos"))]
mod imp {
    pub(super) fn screen_recording_authorized() -> bool {
        true
    }

    pub(super) fn request_screen_recording() -> bool {
        true
    }

    pub(super) fn open_privacy_settings(_pane: &str) -> Result<(), String> {
        Ok(())
    }

    pub(super) fn microphone_blocked() -> bool {
        false
    }
}
