//! whisper.cpp ASR engine with Metal (GPU) acceleration.
//!
//! Wraps the vendored whisper.cpp behind [`wisp_core::AsrEngine`] so it drops into the pipeline
//! like the sherpa engines — but runs on the Apple GPU via Metal instead of CPU-only ONNX, which
//! makes large models (e.g. large-v3-turbo) usable in real time. macOS-only for now.

#![cfg(target_os = "macos")]

mod sys {
    #![allow(
        non_upper_case_globals,
        non_camel_case_types,
        non_snake_case,
        dead_code
    )]
    include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
}

use std::ffi::{CStr, CString};
use std::os::raw::{c_int, c_void};
use std::path::Path;
use std::time::Duration;

use wisp_core::align::{merge_tokens_into_words, TokenTiming};
use wisp_core::engine::{AsrEngine, ClipOptions, EngineInfo, TranscriptionResult};
use wisp_core::error::{Result, WispError};
use wisp_core::transcript::{AudioSourceKind, TranscriptSegment, Word};

/// Rate whisper.cpp expects (and the rest of the pipeline runs at).
const SAMPLE_RATE: u32 = 16_000;

/// Decoder threads — whisper.cpp's encoder runs on the GPU, but the rest still benefits from CPU
/// threads. Scales to the machine's physical cores via the shared policy (overridable with
/// `WISP_ASR_THREADS`).
fn decode_threads() -> i32 {
    let n = wisp_core::perf::asr_threads(
        std::env::var(wisp_core::perf::ASR_THREADS_ENV).ok(),
        num_cpus::get_physical(),
    );
    i32::try_from(n).unwrap_or(i32::MAX)
}

/// Whisper's decoding-robustness thresholds.
///
/// whisper.cpp retries a 30 s window at a higher temperature when a decode looks unreliable — its
/// average token log-prob falls below `logprob_thold`, or its token entropy exceeds `entropy_thold`
/// (a proxy for a repetition loop) — stepping temperature up by `temperature_inc` each attempt. A
/// window whose no-speech probability exceeds `no_speech_thold` is treated as silence. We pin these
/// to whisper.cpp's proven defaults explicitly, so an upstream default change can't silently shift
/// our output, and allow per-deployment tuning through environment variables.
#[derive(Debug, Clone, Copy, PartialEq)]
struct DecodeThresholds {
    temperature_inc: f32,
    entropy_thold: f32,
    logprob_thold: f32,
    no_speech_thold: f32,
}

impl Default for DecodeThresholds {
    fn default() -> Self {
        Self {
            temperature_inc: 0.2,
            entropy_thold: 2.4,
            logprob_thold: -1.0,
            no_speech_thold: 0.6,
        }
    }
}

/// Resolves thresholds from a key→value lookup, using the default for any unset or unparsable value.
fn resolve_thresholds(lookup: impl Fn(&str) -> Option<String>) -> DecodeThresholds {
    let d = DecodeThresholds::default();
    DecodeThresholds {
        temperature_inc: parse_f32_or(
            lookup(WhisperCppEngine::TEMPERATURE_INC_ENV),
            d.temperature_inc,
        ),
        entropy_thold: parse_f32_or(lookup(WhisperCppEngine::ENTROPY_THOLD_ENV), d.entropy_thold),
        logprob_thold: parse_f32_or(lookup(WhisperCppEngine::LOGPROB_THOLD_ENV), d.logprob_thold),
        no_speech_thold: parse_f32_or(
            lookup(WhisperCppEngine::NO_SPEECH_THOLD_ENV),
            d.no_speech_thold,
        ),
    }
}

/// Parses a trimmed `f32` from `raw`, falling back to `default` for `None` or an invalid value.
fn parse_f32_or(raw: Option<String>, default: f32) -> f32 {
    raw.and_then(|v| v.trim().parse::<f32>().ok())
        .unwrap_or(default)
}

/// Reads decoding thresholds from the process environment.
fn thresholds_from_env() -> DecodeThresholds {
    resolve_thresholds(|key| std::env::var(key).ok())
}

/// Applies decoding thresholds onto whisper.cpp params — one place, used by both decode paths.
fn apply_thresholds(params: &mut sys::whisper_full_params, thresholds: DecodeThresholds) {
    params.temperature_inc = thresholds.temperature_inc;
    params.entropy_thold = thresholds.entropy_thold;
    params.logprob_thold = thresholds.logprob_thold;
    params.no_speech_thold = thresholds.no_speech_thold;
}

/// C trampoline forwarding whisper.cpp's 0–100 progress to the Rust `&dyn Fn(u8)` passed as user
/// data, so the caller can drive a progress bar.
///
/// # Safety
/// `user_data` must be the pointer to the `&dyn Fn(u8)` set in `progress_callback_user_data`, and
/// must stay valid for the whole `whisper_full` call.
unsafe extern "C" fn progress_trampoline(
    _ctx: *mut sys::whisper_context,
    _state: *mut sys::whisper_state,
    progress: c_int,
    user_data: *mut c_void,
) {
    if user_data.is_null() {
        return;
    }
    let callback: &dyn Fn(u8) = *(user_data as *const &dyn Fn(u8));
    callback(progress.clamp(0, 100) as u8);
}

/// An [`AsrEngine`] backed by whisper.cpp on the Metal backend.
///
/// The `whisper_context` is not thread-safe for concurrent use, but the pipeline drives one engine
/// from a single transcription thread, so it is `Send` (moved onto that thread) and used serially.
pub struct WhisperCppEngine {
    ctx: *mut sys::whisper_context,
    language: CString,
    /// True when the user left the language on auto-detect. Only then does the live path lock onto
    /// the first detected language, so short utterances stop re-detecting and flapping (see
    /// `transcribe`).
    auto: bool,
    /// Sticky language locked from the first real auto-detected utterance and reused for the rest of
    /// the session, so per-utterance detection can't drift. `None` until locked (or when not on auto).
    detected_language: Option<CString>,
    n_threads: i32,
    thresholds: DecodeThresholds,
    /// Static streaming-path (live) biasing hints (names, jargon), set via `configure_streaming`.
    stream_hints: String,
    /// Recent finalized transcript, fed via `set_context`, so live decoding stays consistent on
    /// names/terms across utterances. Folded into the initial prompt alongside `stream_hints`.
    rolling_context: String,
    /// Beam-search choice for the streaming path, set via `configure_streaming`.
    stream_beam: bool,
}

unsafe impl Send for WhisperCppEngine {}

impl WhisperCppEngine {
    /// Env var overriding the temperature-fallback step (whisper.cpp `temperature_inc`).
    pub const TEMPERATURE_INC_ENV: &'static str = "WISP_WHISPER_TEMPERATURE_INC";
    /// Env var overriding the entropy (repetition-loop) fallback threshold.
    pub const ENTROPY_THOLD_ENV: &'static str = "WISP_WHISPER_ENTROPY_THOLD";
    /// Env var overriding the average-log-prob fallback threshold.
    pub const LOGPROB_THOLD_ENV: &'static str = "WISP_WHISPER_LOGPROB_THOLD";
    /// Env var overriding the no-speech (silence) suppression threshold.
    pub const NO_SPEECH_THOLD_ENV: &'static str = "WISP_WHISPER_NO_SPEECH_THOLD";

    /// Loads a GGUF/GGML whisper model from `model`. `language` is a Whisper code (e.g. `"yue"`);
    /// an empty string means auto-detect.
    pub fn new(model: &Path, language: &str) -> Result<Self> {
        let model_c = CString::new(model.to_string_lossy().as_bytes())
            .map_err(|_| WispError::Engine("model path has an interior NUL".to_owned()))?;
        let auto = language.trim().is_empty() || language.trim() == "auto";
        let language = CString::new(if language.is_empty() {
            "auto"
        } else {
            language
        })
        .map_err(|_| WispError::Engine("language has an interior NUL".to_owned()))?;

        // SAFETY: `model_c` is a valid C string; the returned context is checked for null below.
        let ctx = unsafe {
            let mut cparams = sys::whisper_context_default_params();
            cparams.use_gpu = true; // Metal
            cparams.flash_attn = true; // faster + less memory on Metal, accuracy-neutral
            sys::whisper_init_from_file_with_params(model_c.as_ptr(), cparams)
        };
        if ctx.is_null() {
            return Err(WispError::Engine(format!(
                "whisper.cpp: failed to load model {}",
                model.display()
            )));
        }

        Ok(Self {
            ctx,
            language,
            auto,
            detected_language: None,
            n_threads: decode_threads(),
            thresholds: thresholds_from_env(),
            stream_hints: String::new(),
            rolling_context: String::new(),
            stream_beam: false,
        })
    }
}

/// Joins the static hints and the rolling recent-text context into one biasing prompt (either may
/// be empty). whisper.cpp uses it as `initial_prompt` — text bias only, never carried decoder state.
fn combined_prompt(hints: &str, context: &str) -> String {
    match (hints.trim(), context.trim()) {
        ("", c) => c.to_owned(),
        (h, "") => h.to_owned(),
        (h, c) => format!("{h} {c}"),
    }
}

impl AsrEngine for WhisperCppEngine {
    fn info(&self) -> EngineInfo {
        EngineInfo {
            name: "whisper.cpp-metal".to_owned(),
            streaming: false,
        }
    }

    fn transcribe(&mut self, audio: &[f32], _sample_rate: u32) -> Result<TranscriptionResult> {
        let strategy = if self.stream_beam {
            sys::whisper_sampling_strategy::WHISPER_SAMPLING_BEAM_SEARCH
        } else {
            sys::whisper_sampling_strategy::WHISPER_SAMPLING_GREEDY
        };
        // Static hints + rolling recent-text context, biasing the decode toward consistent
        // names/terms. Built before the unsafe block so it outlives the `whisper_full` call.
        let prompt_c = CString::new(combined_prompt(&self.stream_hints, &self.rolling_context))
            .unwrap_or_default();

        // Sticky language: reuse the locked language once auto-detect has settled, else the
        // configured one ("auto" until then). Read from `self`, so the pointer stays valid through
        // `whisper_full`. `want_detect` gates the post-decode FFI read to when we still need to lock.
        let language_ptr = self
            .detected_language
            .as_deref()
            .unwrap_or(self.language.as_c_str())
            .as_ptr();
        let want_detect = self.auto && self.detected_language.is_none();

        // SAFETY: `self.ctx` is a live context; `audio` is a valid slice; the language and
        // `prompt_c` pointers outlive the call so they stay valid through `whisper_full`.
        let (text, detected) = unsafe {
            let mut params = sys::whisper_full_default_params(strategy);
            params.n_threads = self.n_threads;
            params.language = language_ptr;
            params.translate = false;
            params.no_context = true; // no carried decoder state — context is text-only, via the prompt
            params.no_timestamps = true;
            params.print_progress = false;
            params.print_realtime = false;
            params.print_special = false;
            params.print_timestamps = false;
            params.single_segment = false;
            if !prompt_c.as_bytes().is_empty() {
                params.initial_prompt = prompt_c.as_ptr();
            }
            apply_thresholds(&mut params, self.thresholds);

            let rc = sys::whisper_full(self.ctx, params, audio.as_ptr(), audio.len() as c_int);
            if rc != 0 {
                return Err(WispError::Engine(format!(
                    "whisper.cpp: whisper_full failed (rc={rc})"
                )));
            }

            let mut text = String::new();
            for i in 0..sys::whisper_full_n_segments(self.ctx) {
                let ptr = sys::whisper_full_get_segment_text(self.ctx, i);
                if !ptr.is_null() {
                    text.push_str(&CStr::from_ptr(ptr).to_string_lossy());
                }
            }
            // Read what language whisper settled on, but only while we still need to lock one.
            let detected = if want_detect {
                detected_language_code(self.ctx)
            } else {
                String::new()
            };
            (text, detected)
        };

        // Lock the session language the first time real speech yields a detection, so later short
        // utterances can't independently re-detect and flap between languages.
        if should_lock_language(
            self.auto,
            self.detected_language.is_some(),
            !text.trim().is_empty(),
            &detected,
        ) {
            self.detected_language = CString::new(detected).ok();
        }

        Ok(to_result(text.trim(), audio))
    }

    fn transcribe_clip(
        &mut self,
        audio: &[f32],
        _sample_rate: u32,
        options: ClipOptions<'_>,
    ) -> Result<TranscriptionResult> {
        // Native long-form: feed the whole clip and carry context across whisper's 30 s windows for
        // far better consistency than per-utterance chunks. Beam search (vs greedy) and timestamps
        // follow `options`, so the caller can trade accuracy against speed.
        let strategy = if options.beam {
            sys::whisper_sampling_strategy::WHISPER_SAMPLING_BEAM_SEARCH
        } else {
            sys::whisper_sampling_strategy::WHISPER_SAMPLING_GREEDY
        };
        // Context primer (names, jargon) biasing the decoder toward correct spellings. Held in this
        // scope so its pointer stays valid for the whole `whisper_full` call; empty = no biasing.
        let prompt = CString::new(options.prompt).unwrap_or_default();
        // The progress sink, held here so the pointer we hand whisper.cpp stays valid through the call.
        let progress = options.progress;
        // SAFETY: live context; valid slice; `self.language`, `prompt`, and `progress` outlive the call.
        let segments = unsafe {
            let mut params = sys::whisper_full_default_params(strategy);
            params.n_threads = self.n_threads;
            params.language = self.language.as_ptr();
            params.translate = false;
            params.no_context = false; // carry context across windows
            params.no_timestamps = !options.timestamps;
            params.token_timestamps = options.timestamps; // per-token times → word-level alignment
            params.suppress_nst = true; // suppress non-speech tokens (fewer hallucinations)
            if !options.prompt.is_empty() {
                params.initial_prompt = prompt.as_ptr();
            }
            params.print_progress = false;
            params.print_realtime = false;
            params.print_special = false;
            params.print_timestamps = false;
            params.single_segment = false;
            apply_thresholds(&mut params, self.thresholds);
            if let Some(callback) = &progress {
                params.progress_callback = Some(progress_trampoline);
                params.progress_callback_user_data = callback as *const &dyn Fn(u8) as *mut c_void;
            }

            let rc = sys::whisper_full(self.ctx, params, audio.as_ptr(), audio.len() as c_int);
            if rc != 0 {
                return Err(WispError::Engine(format!(
                    "whisper.cpp: whisper_full failed (rc={rc})"
                )));
            }

            let mut segments = Vec::new();
            for i in 0..sys::whisper_full_n_segments(self.ctx) {
                let ptr = sys::whisper_full_get_segment_text(self.ctx, i);
                if ptr.is_null() {
                    continue;
                }
                let text = CStr::from_ptr(ptr).to_string_lossy().trim().to_owned();
                if text.is_empty() {
                    continue;
                }
                let (start, end) = if options.timestamps {
                    // whisper.cpp reports segment times in centiseconds (10 ms units).
                    let t0 = sys::whisper_full_get_segment_t0(self.ctx, i).max(0) as u64 * 10;
                    let t1 = sys::whisper_full_get_segment_t1(self.ctx, i).max(0) as u64 * 10;
                    (Duration::from_millis(t0), Duration::from_millis(t1))
                } else {
                    (Duration::ZERO, Duration::ZERO)
                };
                let mut segment =
                    TranscriptSegment::new(0, text.as_str(), start..end, AudioSourceKind::File);
                if options.timestamps {
                    segment.words = segment_words(self.ctx, i);
                }
                segments.push(segment);
            }
            segments
        };

        Ok(TranscriptionResult { segments })
    }

    fn configure_streaming(&mut self, prompt: &str, beam: bool) {
        self.stream_hints = prompt.to_owned();
        self.stream_beam = beam;
    }

    fn set_context(&mut self, recent_text: &str) {
        self.rolling_context = recent_text.to_owned();
    }

    fn reset(&mut self) {
        // `no_context = true` makes each `transcribe` independent — there is no carried decoder
        // state. Rolling context is a separate text prompt, intentionally preserved across resets.
    }
}

impl Drop for WhisperCppEngine {
    fn drop(&mut self) {
        // SAFETY: `self.ctx` was created by `whisper_init_*` and is freed exactly once here.
        unsafe { sys::whisper_free(self.ctx) };
    }
}

/// Extracts whisper.cpp's per-token timings for segment `i` and merges them into words.
///
/// # Safety
/// `ctx` must be a live context whose last `whisper_full` ran with `token_timestamps` enabled, and
/// `i` a valid segment index.
unsafe fn segment_words(ctx: *mut sys::whisper_context, i: c_int) -> Vec<Word> {
    let n_tokens = sys::whisper_full_n_tokens(ctx, i);
    let mut tokens = Vec::with_capacity(n_tokens.max(0) as usize);
    for j in 0..n_tokens {
        let ptr = sys::whisper_full_get_token_text(ctx, i, j);
        if ptr.is_null() {
            continue;
        }
        let text = CStr::from_ptr(ptr).to_string_lossy().into_owned();
        // Token times, like segment times, are centiseconds (10 ms units).
        let data = sys::whisper_full_get_token_data(ctx, i, j);
        let start = Duration::from_millis(data.t0.max(0) as u64 * 10);
        let end = Duration::from_millis(data.t1.max(0) as u64 * 10);
        tokens.push(TokenTiming::new(text, start, end));
    }
    merge_tokens_into_words(&tokens)
}

/// Wraps recognized `text` (spanning `audio` at 16 kHz) as a single-segment result; empty → empty.
fn to_result(text: &str, audio: &[f32]) -> TranscriptionResult {
    if text.is_empty() {
        return TranscriptionResult::empty();
    }
    let secs = audio.len() as f32 / SAMPLE_RATE as f32;
    let segment = TranscriptSegment::new(
        0,
        text,
        Duration::ZERO..Duration::from_secs_f32(secs),
        AudioSourceKind::Microphone,
    );
    TranscriptionResult {
        segments: vec![segment],
    }
}

/// Whether the live auto-detect should lock onto `detected` as the session language. Locks only when
/// the user left the language on auto, nothing is locked yet, the utterance produced text, and the
/// detection is a real language code (not empty or `"auto"`). Pure, so the policy is testable with no
/// model loaded.
fn should_lock_language(
    auto: bool,
    already_locked: bool,
    produced_text: bool,
    detected: &str,
) -> bool {
    auto && !already_locked && produced_text && !detected.is_empty() && detected != "auto"
}

/// The language whisper.cpp settled on for the last `whisper_full` decode, as its short code (e.g.
/// `"en"`, `"zh"`), or `""` if unavailable.
///
/// # Safety
/// `ctx` must be a live whisper context that has just completed a `whisper_full` call.
unsafe fn detected_language_code(ctx: *mut sys::whisper_context) -> String {
    let id = sys::whisper_full_lang_id(ctx);
    if id < 0 {
        return String::new();
    }
    let ptr = sys::whisper_lang_str(id);
    if ptr.is_null() {
        String::new()
    } else {
        CStr::from_ptr(ptr).to_string_lossy().into_owned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    /// End-to-end (real Metal inference): loads a model and transcribes a fixture WAV. Skip-guarded
    /// behind two env vars so `cargo test` is clean where the (multi-MB) model isn't available:
    ///   WISP_WHISPER_CPP_TEST_MODEL=/path/to/ggml-*.bin
    ///   WISP_WHISPER_CPP_TEST_WAV=/path/to/16khz-mono.wav
    #[test]
    fn transcribes_fixture_wav_when_assets_present() {
        let (Some(model), Some(wav)) = (
            std::env::var_os("WISP_WHISPER_CPP_TEST_MODEL"),
            std::env::var_os("WISP_WHISPER_CPP_TEST_WAV"),
        ) else {
            return;
        };
        let (model, wav) = (PathBuf::from(model), PathBuf::from(wav));
        if !model.exists() || !wav.exists() {
            return;
        }

        let mut reader = hound::WavReader::open(&wav).expect("open fixture wav");
        let audio: Vec<f32> = reader
            .samples::<i16>()
            .map(|s| s.expect("read sample") as f32 / 32768.0)
            .collect();

        let mut engine = WhisperCppEngine::new(&model, "en").expect("load model");
        let result = engine.transcribe(&audio, SAMPLE_RATE).expect("transcribe");

        let text: String = result.segments.iter().map(|s| s.text.as_str()).collect();
        assert!(
            !text.trim().is_empty(),
            "expected a non-empty transcription, got {text:?}"
        );
    }

    #[test]
    fn combined_prompt_joins_hints_and_context_handling_empties() {
        assert_eq!(combined_prompt("", ""), "");
        assert_eq!(combined_prompt("Acme, kubectl", ""), "Acme, kubectl");
        assert_eq!(
            combined_prompt("", "we discussed the migration"),
            "we discussed the migration"
        );
        assert_eq!(
            combined_prompt("Acme, kubectl", "we discussed the migration"),
            "Acme, kubectl we discussed the migration"
        );
        // Surrounding whitespace on either part is trimmed before joining.
        assert_eq!(combined_prompt("  hints  ", "  ctx  "), "hints ctx");
    }

    #[test]
    fn locks_language_on_first_real_auto_detection() {
        // Auto, nothing locked yet, this utterance produced text, real code → lock it.
        assert!(should_lock_language(true, false, true, "en"));
    }

    #[test]
    fn does_not_lock_when_language_is_pinned() {
        // User chose a language explicitly → never override it with a detection.
        assert!(!should_lock_language(false, false, true, "en"));
    }

    #[test]
    fn does_not_lock_again_once_locked() {
        // Already locked → keep it; later utterances must not re-detect.
        assert!(!should_lock_language(true, true, true, "ja"));
    }

    #[test]
    fn does_not_lock_on_silence_or_unknown_detection() {
        // No text (silence/noise), or a non-language detection, must not lock a bogus language.
        assert!(!should_lock_language(true, false, false, "en"));
        assert!(!should_lock_language(true, false, true, ""));
        assert!(!should_lock_language(true, false, true, "auto"));
    }

    #[test]
    fn threshold_env_var_names_are_pinned() {
        // Renaming these breaks any operator who tuned decoding via a private env override. Pin the
        // exact literals so a rename is a visible, deliberate change.
        assert_eq!(
            WhisperCppEngine::TEMPERATURE_INC_ENV,
            "WISP_WHISPER_TEMPERATURE_INC"
        );
        assert_eq!(
            WhisperCppEngine::ENTROPY_THOLD_ENV,
            "WISP_WHISPER_ENTROPY_THOLD"
        );
        assert_eq!(
            WhisperCppEngine::LOGPROB_THOLD_ENV,
            "WISP_WHISPER_LOGPROB_THOLD"
        );
        assert_eq!(
            WhisperCppEngine::NO_SPEECH_THOLD_ENV,
            "WISP_WHISPER_NO_SPEECH_THOLD"
        );
    }

    #[test]
    fn thresholds_default_when_unset() {
        // Nothing in the environment → whisper.cpp's pinned defaults.
        assert_eq!(resolve_thresholds(|_| None), DecodeThresholds::default());
    }

    #[test]
    fn thresholds_parse_every_override() {
        let resolved = resolve_thresholds(|key| {
            Some(
                match key {
                    WhisperCppEngine::TEMPERATURE_INC_ENV => "0.4",
                    WhisperCppEngine::ENTROPY_THOLD_ENV => "3.0",
                    WhisperCppEngine::LOGPROB_THOLD_ENV => "-0.5",
                    WhisperCppEngine::NO_SPEECH_THOLD_ENV => "0.8",
                    _ => return None,
                }
                .to_owned(),
            )
        });
        assert_eq!(
            resolved,
            DecodeThresholds {
                temperature_inc: 0.4,
                entropy_thold: 3.0,
                logprob_thold: -0.5,
                no_speech_thold: 0.8,
            }
        );
    }

    #[test]
    fn thresholds_ignore_invalid_values() {
        let resolved = resolve_thresholds(|key| {
            (key == WhisperCppEngine::NO_SPEECH_THOLD_ENV).then(|| "not-a-number".to_owned())
        });
        assert_eq!(
            resolved,
            DecodeThresholds::default(),
            "an unparsable override falls back to the default"
        );
    }

    #[test]
    fn thresholds_allow_a_single_trimmed_override() {
        let resolved = resolve_thresholds(|key| {
            (key == WhisperCppEngine::NO_SPEECH_THOLD_ENV).then(|| "  0.75  ".to_owned())
        });
        // Only the one field changes; whitespace is trimmed before parsing.
        assert_eq!(
            resolved,
            DecodeThresholds {
                no_speech_thold: 0.75,
                ..DecodeThresholds::default()
            }
        );
    }
}
