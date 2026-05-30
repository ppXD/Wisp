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
use std::os::raw::c_int;
use std::path::Path;
use std::time::Duration;

use wisp_core::align::{merge_tokens_into_words, TokenTiming};
use wisp_core::engine::{AsrEngine, ClipOptions, EngineInfo, TranscriptionResult};
use wisp_core::error::{Result, WispError};
use wisp_core::transcript::{AudioSourceKind, TranscriptSegment, Word};

/// Rate whisper.cpp expects (and the rest of the pipeline runs at).
const SAMPLE_RATE: u32 = 16_000;

/// Decoder threads — whisper.cpp's encoder runs on the GPU, but the rest still benefits from a few
/// CPU threads.
fn decode_threads() -> i32 {
    std::thread::available_parallelism()
        .map(|n| n.get().min(8) as i32)
        .unwrap_or(4)
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

/// An [`AsrEngine`] backed by whisper.cpp on the Metal backend.
///
/// The `whisper_context` is not thread-safe for concurrent use, but the pipeline drives one engine
/// from a single transcription thread, so it is `Send` (moved onto that thread) and used serially.
pub struct WhisperCppEngine {
    ctx: *mut sys::whisper_context,
    language: CString,
    n_threads: i32,
    thresholds: DecodeThresholds,
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
            n_threads: decode_threads(),
            thresholds: thresholds_from_env(),
        })
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
        // SAFETY: `self.ctx` is a live context; `audio` is a valid slice; `self.language` outlives
        // the call so its pointer stays valid for the duration of `whisper_full`.
        let text = unsafe {
            let mut params = sys::whisper_full_default_params(
                sys::whisper_sampling_strategy::WHISPER_SAMPLING_GREEDY,
            );
            params.n_threads = self.n_threads;
            params.language = self.language.as_ptr();
            params.translate = false;
            params.no_context = true; // each utterance is independent — avoids drift/hallucination
            params.no_timestamps = true;
            params.print_progress = false;
            params.print_realtime = false;
            params.print_special = false;
            params.print_timestamps = false;
            params.single_segment = false;
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
            text
        };

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
        // SAFETY: live context; valid slice; `self.language` and `prompt` outlive the call.
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

    fn reset(&mut self) {
        // `no_context = true` makes each `transcribe` independent — there is no carried state.
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
