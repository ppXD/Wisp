//! On-device streaming speech recognition through macOS 26's `SpeechAnalyzer` / `SpeechTranscriber`,
//! behind the same [`StreamingAsrEngine`] trait as the other live engines.
//!
//! Unlike the Whisper/Parakeet engines this downloads **no model of ours** — Apple ships the recogniser
//! and fetches the per-language asset itself on first use — so it's a zero-download, low-power live
//! option on a Mac. The native work lives in `swift/WispSpeech.swift`; this is the safe Rust wrapper.
//!
//! macOS only; every other target compiles a stub whose [`is_available`] returns `false`. The pure
//! helpers ([`locale_for_language`], emission selection) live at the crate root so they compile and
//! test on every platform.

use std::collections::VecDeque;

/// Whether on-device Apple speech recognition can run here (macOS 26+). Always `false` off macOS.
#[cfg(not(target_os = "macos"))]
pub fn is_available() -> bool {
    false
}

#[cfg(target_os = "macos")]
pub use mac::{is_available, AppleSpeechEngine};

/// Maps a Wisp language code (`"en"`, `"zh"`, `"yue"`, `"auto"`, …) to the BCP-47 locale identifier the
/// Apple recogniser expects (`"en-US"`, `"zh-CN"`, `"zh-HK"`, …). `"auto"` is passed through so the
/// Swift side can resolve it to the system locale; an unknown bare code falls back to US English, while
/// an already region-qualified tag (e.g. `"en-GB"`) passes through verbatim.
pub fn locale_for_language(language: &str) -> String {
    let lower = language.trim().to_ascii_lowercase();
    let mapped = match lower.as_str() {
        "auto" | "" => "auto",
        "en" => "en-US",
        "zh" | "cmn" => "zh-CN",
        // Cantonese: Apple ships its on-device asset under the Hong Kong Chinese locale.
        "yue" => "zh-HK",
        "ja" => "ja-JP",
        "ko" => "ko-KR",
        "es" => "es-ES",
        "fr" => "fr-FR",
        "de" => "de-DE",
        "it" => "it-IT",
        "pt" => "pt-BR",
        "ru" => "ru-RU",
        "ar" => "ar-SA",
        "hi" => "hi-IN",
        // Already a region-qualified tag (e.g. "en-GB", "zh-TW") — pass it through untouched.
        other if other.contains('-') => return language.trim().to_owned(),
        _ => "en-US",
    };
    mapped.to_owned()
}

/// What to surface from the recogniser this poll. A finalised utterance takes priority; otherwise a
/// *changed* volatile partial; otherwise nothing — so an unchanged partial isn't re-emitted every frame.
// Used by the `mac` engine and the cross-platform `pure_tests`; off macOS the engine is absent, so the
// non-test lib build sees these as unused — allow it there rather than re-derive per platform.
#[derive(Debug, PartialEq, Eq)]
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
enum Emission {
    Final(String),
    Partial(String),
    Idle,
}

/// Picks the next [`Emission`] from the drained finals, the current volatile hypothesis, and the text
/// last emitted: a pending final (utterance boundary) wins; else a volatile that actually changed; else
/// idle. Emitting a final clears `last_partial`, so the next utterance's first partial always surfaces.
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
fn select_emission(
    finals: &mut VecDeque<String>,
    volatile: &str,
    last_partial: &mut String,
) -> Emission {
    if let Some(text) = finals.pop_front() {
        last_partial.clear();
        return Emission::Final(text);
    }

    if volatile.is_empty() || volatile == last_partial {
        return Emission::Idle;
    }

    *last_partial = volatile.to_owned();
    Emission::Partial(volatile.to_owned())
}

#[cfg(test)]
mod pure_tests {
    use super::*;

    #[test]
    fn locale_maps_known_codes_and_passes_through_qualified_tags() {
        assert_eq!(locale_for_language("en"), "en-US");
        assert_eq!(locale_for_language("zh"), "zh-CN");
        assert_eq!(locale_for_language("yue"), "zh-HK");
        assert_eq!(locale_for_language("ja"), "ja-JP");
        assert_eq!(locale_for_language("auto"), "auto");
        // Region-qualified tags pass through verbatim; unknown bare codes fall back to US English.
        assert_eq!(locale_for_language("en-GB"), "en-GB");
        assert_eq!(locale_for_language("zh-TW"), "zh-TW");
        assert_eq!(locale_for_language("tlh"), "en-US");
        // Case/whitespace-insensitive.
        assert_eq!(locale_for_language(" YUE "), "zh-HK");
    }

    #[test]
    fn a_final_takes_priority_and_clears_the_partial_dedup() {
        let mut finals = VecDeque::from(["hello world".to_owned()]);
        let mut last = "hello".to_owned();
        assert_eq!(
            select_emission(&mut finals, "hello", &mut last),
            Emission::Final("hello world".to_owned())
        );
        assert!(last.is_empty(), "emitting a final resets the de-dup");
    }

    #[test]
    fn a_changed_volatile_emits_once_then_idles_until_it_changes() {
        let mut finals = VecDeque::new();
        let mut last = String::new();

        assert_eq!(
            select_emission(&mut finals, "how are", &mut last),
            Emission::Partial("how are".to_owned())
        );
        // Same volatile again → no re-emit (de-duped).
        assert_eq!(
            select_emission(&mut finals, "how are", &mut last),
            Emission::Idle
        );
        // Grown volatile → emits again.
        assert_eq!(
            select_emission(&mut finals, "how are you", &mut last),
            Emission::Partial("how are you".to_owned())
        );
    }

    #[test]
    fn an_empty_volatile_is_idle() {
        let mut finals = VecDeque::new();
        let mut last = String::new();
        assert_eq!(select_emission(&mut finals, "", &mut last), Emission::Idle);
    }
}

// `apple_speech_real` (set by build.rs) gates this to builds that compiled the genuine macOS-26 shim;
// on an older SDK we link the unavailable stub, where asserting availability would (correctly) fail.
#[cfg(all(test, target_os = "macos", apple_speech_real))]
mod tests {
    /// Calling across the FFI proves the Swift shim links (static lib + Speech/AVFoundation frameworks +
    /// Swift runtime) and runs; on the build machine (macOS 26+) it should report available.
    #[test]
    fn reports_availability_on_this_os() {
        assert!(
            super::is_available(),
            "macOS 26+ should report Apple on-device speech available"
        );
    }
}

#[cfg(target_os = "macos")]
mod mac {
    use std::collections::VecDeque;
    use std::ffi::{c_char, c_void, CStr, CString};
    use std::sync::mpsc::{channel, Receiver, Sender};

    use wisp_core::engine::{StreamingAsrEngine, StreamingResult};
    use wisp_core::error::{Result, WispError};

    use super::{select_emission, Emission};

    /// The mono rate fed to the shim; the Swift side resamples it to the analyzer's native format.
    const INPUT_RATE: u32 = 16_000;

    extern "C" {
        fn wisp_applespeech_available() -> bool;
        fn wisp_applespeech_start(
            locale: *const c_char,
            ctx: *mut c_void,
            callback: extern "C" fn(*mut c_void, *const c_char, bool),
        ) -> *mut c_void;
        fn wisp_applespeech_feed(handle: *mut c_void, samples: *const f32, count: usize, rate: f64);
        fn wisp_applespeech_stop(handle: *mut c_void);
    }

    /// Whether on-device Apple speech recognition can run here (macOS 26+).
    pub fn is_available() -> bool {
        unsafe { wisp_applespeech_available() }
    }

    /// The C callback the Swift shim invokes for each (partial/final) result. `ctx` is a leaked
    /// `Sender<Event>` — see [`AppleSpeechEngine::new`] for why it is intentionally never freed.
    extern "C" fn on_result(ctx: *mut c_void, text: *const c_char, is_final: bool) {
        if ctx.is_null() || text.is_null() {
            return;
        }
        let tx = unsafe { &*(ctx as *const Sender<Event>) };
        let text = unsafe { CStr::from_ptr(text) }
            .to_string_lossy()
            .into_owned();
        let _ = tx.send(Event { text, is_final });
    }

    struct Event {
        text: String,
        is_final: bool,
    }

    /// A live [`StreamingAsrEngine`] backed by the Apple recogniser. Results arrive asynchronously on the
    /// shim's thread and are drained, on the pull-based [`accept_waveform`](Self::accept_waveform), into
    /// Wisp's partial/final model: a final segment ends an utterance (`is_endpoint`), the in-progress
    /// volatile text is the partial (emitted only when it changes).
    pub struct AppleSpeechEngine {
        handle: *mut c_void,
        events: Receiver<Event>,
        /// Final segments waiting to be surfaced one-per-call as endpoints.
        finals: VecDeque<String>,
        /// The current in-progress (volatile) hypothesis.
        volatile: String,
        /// The partial text last emitted, so an unchanged volatile isn't re-sent every frame.
        last_partial: String,
    }

    // The handle is a heap session owned solely by this engine; the `Receiver` is `Send`. The shim only
    // touches the handle/ctx, never Rust state, so moving the engine across threads is sound.
    unsafe impl Send for AppleSpeechEngine {}

    impl AppleSpeechEngine {
        /// Starts a recogniser for `locale` (BCP-47, e.g. `en-US`, `zh-CN`, `zh-HK`, or `auto` for the
        /// system locale). Errors if the OS is too old for the API.
        pub fn new(locale: &str) -> Result<Self> {
            if !is_available() {
                return Err(WispError::Engine(
                    "Apple on-device speech needs macOS 26 or newer".to_owned(),
                ));
            }

            let (tx, events) = channel::<Event>();
            // The shim holds this `ctx` for the session's lifetime and its result Task may fire the
            // callback briefly *after* `stop`, so freeing the `Sender` on drop would risk a use-after-free.
            // It is one small allocation per session, so we leak it deliberately rather than race.
            let ctx = Box::into_raw(Box::new(tx)) as *mut c_void;

            let locale_c = CString::new(locale).unwrap_or_default();
            let handle = unsafe { wisp_applespeech_start(locale_c.as_ptr(), ctx, on_result) };
            if handle.is_null() {
                return Err(WispError::Engine(
                    "Apple on-device speech is unavailable".to_owned(),
                ));
            }

            Ok(Self {
                handle,
                events,
                finals: VecDeque::new(),
                volatile: String::new(),
                last_partial: String::new(),
            })
        }

        /// Drains every result the shim has delivered into `finals` / `volatile`.
        fn drain(&mut self) {
            while let Ok(event) = self.events.try_recv() {
                if event.is_final {
                    let text = event.text.trim().to_owned();
                    if !text.is_empty() {
                        self.finals.push_back(text);
                    }
                    self.volatile.clear();
                } else {
                    self.volatile = event.text;
                }
            }
        }
    }

    impl StreamingAsrEngine for AppleSpeechEngine {
        fn accept_waveform(&mut self, sample_rate: u32, samples: &[f32]) -> StreamingResult {
            if !samples.is_empty() {
                unsafe {
                    wisp_applespeech_feed(
                        self.handle,
                        samples.as_ptr(),
                        samples.len(),
                        f64::from(sample_rate),
                    );
                }
            }

            self.drain();

            match select_emission(&mut self.finals, &self.volatile, &mut self.last_partial) {
                Emission::Final(text) => StreamingResult::new(text, true),
                Emission::Partial(text) => StreamingResult::new(text, false),
                Emission::Idle => StreamingResult::new(String::new(), false),
            }
        }

        fn input_sample_rate(&self) -> u32 {
            INPUT_RATE
        }

        fn reset(&mut self) {
            self.finals.clear();
            self.volatile.clear();
            self.last_partial.clear();
        }
    }

    impl Drop for AppleSpeechEngine {
        fn drop(&mut self) {
            unsafe { wisp_applespeech_stop(self.handle) };
        }
    }
}
