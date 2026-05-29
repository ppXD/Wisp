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

use wisp_core::engine::{AsrEngine, ClipOptions, EngineInfo, TranscriptionResult};
use wisp_core::error::{Result, WispError};
use wisp_core::transcript::{AudioSourceKind, TranscriptSegment};

/// Rate whisper.cpp expects (and the rest of the pipeline runs at).
const SAMPLE_RATE: u32 = 16_000;

/// Decoder threads — whisper.cpp's encoder runs on the GPU, but the rest still benefits from a few
/// CPU threads.
fn decode_threads() -> i32 {
    std::thread::available_parallelism()
        .map(|n| n.get().min(8) as i32)
        .unwrap_or(4)
}

/// An [`AsrEngine`] backed by whisper.cpp on the Metal backend.
///
/// The `whisper_context` is not thread-safe for concurrent use, but the pipeline drives one engine
/// from a single transcription thread, so it is `Send` (moved onto that thread) and used serially.
pub struct WhisperCppEngine {
    ctx: *mut sys::whisper_context,
    language: CString,
    n_threads: i32,
}

unsafe impl Send for WhisperCppEngine {}

impl WhisperCppEngine {
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
        options: ClipOptions,
    ) -> Result<TranscriptionResult> {
        // Native long-form: feed the whole clip and carry context across whisper's 30 s windows for
        // far better consistency than per-utterance chunks. Beam search (vs greedy) and timestamps
        // follow `options`, so the caller can trade accuracy against speed.
        let strategy = if options.beam {
            sys::whisper_sampling_strategy::WHISPER_SAMPLING_BEAM_SEARCH
        } else {
            sys::whisper_sampling_strategy::WHISPER_SAMPLING_GREEDY
        };
        // SAFETY: live context; valid slice; `self.language` outlives the call.
        let segments = unsafe {
            let mut params = sys::whisper_full_default_params(strategy);
            params.n_threads = self.n_threads;
            params.language = self.language.as_ptr();
            params.translate = false;
            params.no_context = false; // carry context across windows
            params.no_timestamps = !options.timestamps;
            params.suppress_nst = true; // suppress non-speech tokens (fewer hallucinations)
            params.print_progress = false;
            params.print_realtime = false;
            params.print_special = false;
            params.print_timestamps = false;
            params.single_segment = false;

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
                segments.push(TranscriptSegment::new(
                    0,
                    text.as_str(),
                    start..end,
                    AudioSourceKind::File,
                ));
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
}
