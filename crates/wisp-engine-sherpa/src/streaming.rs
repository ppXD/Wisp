//! True streaming (online) transducer ASR via sherpa-onnx.
//!
//! Unlike the offline engines ([`SenseVoiceEngine`](crate::SenseVoiceEngine),
//! [`WhisperEngine`](crate::WhisperEngine)) — which transcribe a whole VAD-segmented utterance at
//! once and so can't emit a word until the speaker pauses — an online transducer (Zipformer)
//! consumes audio incrementally and emits a growing hypothesis within a couple hundred milliseconds,
//! detecting utterance endpoints itself. That removes the segmentation latency, which is the core
//! gap versus realtime systems; it also runs comfortably on the CPU, so it's a fast path on every
//! platform (no GPU required).
//!
//! sherpa-rs only wraps the *offline* recognizers, so this drives the online C API
//! (`sherpa_rs_sys`) directly — the same FFI pattern `keyword_spot`/`speaker_id` use for their
//! online streams.

use std::ffi::{CStr, CString};
use std::path::Path;

use wisp_core::engine::{StreamingAsrEngine, StreamingResult};
use wisp_core::error::{Result, WispError};

/// Endpoint detection: finalize after this much trailing silence even if nothing was decoded.
const RULE1_MIN_TRAILING_SILENCE: f32 = 2.4;
/// Endpoint detection: finalize after this much trailing silence once non-blank text was decoded —
/// the usual "speaker paused" trigger for live transcription.
const RULE2_MIN_TRAILING_SILENCE: f32 = 1.2;
/// Endpoint detection: force a boundary once an utterance runs this long, so a monologue still
/// finalizes periodically.
const RULE3_MIN_UTTERANCE_LENGTH: f32 = 20.0;

/// A streaming online transducer (Zipformer) recognizer.
///
/// The sherpa recognizer/stream are not safe for concurrent use, but — like the other engines — the
/// pipeline drives one from a single transcription thread, so this is `Send` (moved onto that
/// thread) and used serially.
pub struct StreamingTransducerEngine {
    recognizer: *const sherpa_rs_sys::SherpaOnnxOnlineRecognizer,
    stream: *const sherpa_rs_sys::SherpaOnnxOnlineStream,
}

// SAFETY: the raw pointers are owned by this struct and only ever touched from the one thread the
// engine is moved to; nothing aliases or shares them across threads.
unsafe impl Send for StreamingTransducerEngine {}

impl StreamingTransducerEngine {
    /// Loads a streaming transducer from its `encoder`/`decoder`/`joiner` ONNX files and `tokens`.
    /// `num_threads` should be the machine's physical-core count (see `wisp_core::perf`).
    pub fn new(
        encoder: &Path,
        decoder: &Path,
        joiner: &Path,
        tokens: &Path,
        num_threads: i32,
    ) -> Result<Self> {
        // Every C string must outlive the create call below, so bind them all first.
        let encoder_c = path_cstring(encoder)?;
        let decoder_c = path_cstring(decoder)?;
        let joiner_c = path_cstring(joiner)?;
        let tokens_c = path_cstring(tokens)?;
        let provider_c = CString::new("cpu").expect("static str has no NUL");
        let decoding_c = CString::new("greedy_search").expect("static str has no NUL");

        // SAFETY: zero-init nulls every unused model sub-config (sherpa treats a null path as "not
        // this model type"); we then fill only the transducer fields. All pointers reference the
        // CStrings above, which outlive this call.
        let recognizer = unsafe {
            let mut config: sherpa_rs_sys::SherpaOnnxOnlineRecognizerConfig = std::mem::zeroed();
            config.feat_config.sample_rate = 16_000;
            config.feat_config.feature_dim = 80;
            config.model_config.transducer.encoder = encoder_c.as_ptr();
            config.model_config.transducer.decoder = decoder_c.as_ptr();
            config.model_config.transducer.joiner = joiner_c.as_ptr();
            config.model_config.tokens = tokens_c.as_ptr();
            config.model_config.num_threads = num_threads.max(1);
            config.model_config.provider = provider_c.as_ptr();
            config.decoding_method = decoding_c.as_ptr();
            config.enable_endpoint = 1;
            config.rule1_min_trailing_silence = RULE1_MIN_TRAILING_SILENCE;
            config.rule2_min_trailing_silence = RULE2_MIN_TRAILING_SILENCE;
            config.rule3_min_utterance_length = RULE3_MIN_UTTERANCE_LENGTH;
            sherpa_rs_sys::SherpaOnnxCreateOnlineRecognizer(&config)
        };
        if recognizer.is_null() {
            return Err(WispError::Engine(
                "streaming transducer: failed to create recognizer (check model files)".to_owned(),
            ));
        }

        // SAFETY: `recognizer` is non-null (checked); on failure we free it before returning.
        let stream = unsafe { sherpa_rs_sys::SherpaOnnxCreateOnlineStream(recognizer) };
        if stream.is_null() {
            unsafe { sherpa_rs_sys::SherpaOnnxDestroyOnlineRecognizer(recognizer) };
            return Err(WispError::Engine(
                "streaming transducer: failed to create stream".to_owned(),
            ));
        }

        Ok(Self { recognizer, stream })
    }

    /// Reads the recognizer's current hypothesis text, freeing the result it allocates.
    ///
    /// # Safety
    /// `self.recognizer`/`self.stream` must be live.
    unsafe fn current_text(&self) -> String {
        let result = sherpa_rs_sys::SherpaOnnxGetOnlineStreamResult(self.recognizer, self.stream);
        if result.is_null() {
            return String::new();
        }
        let text = if (*result).text.is_null() {
            String::new()
        } else {
            CStr::from_ptr((*result).text)
                .to_string_lossy()
                .into_owned()
        };
        sherpa_rs_sys::SherpaOnnxDestroyOnlineRecognizerResult(result);
        text
    }
}

impl StreamingAsrEngine for StreamingTransducerEngine {
    /// Feeds a chunk of 16 kHz mono `f32` audio and returns the updated hypothesis. When the
    /// recognizer detects an utterance endpoint, the result's `is_endpoint` is true and the stream
    /// is reset for the next utterance.
    fn accept_waveform(&mut self, sample_rate: u32, samples: &[f32]) -> StreamingResult {
        let rate = if sample_rate == 0 {
            16_000
        } else {
            sample_rate
        } as i32;

        // SAFETY: recognizer/stream are live; `samples` is a valid slice; the result pointer from
        // GetResult is freed in `current_text`.
        unsafe {
            sherpa_rs_sys::SherpaOnnxOnlineStreamAcceptWaveform(
                self.stream,
                rate,
                samples.as_ptr(),
                samples.len() as i32,
            );
            while sherpa_rs_sys::SherpaOnnxIsOnlineStreamReady(self.recognizer, self.stream) != 0 {
                sherpa_rs_sys::SherpaOnnxDecodeOnlineStream(self.recognizer, self.stream);
            }

            // Read the hypothesis *before* any reset, so an endpoint result carries the final text.
            let text = self.current_text();
            let is_endpoint =
                sherpa_rs_sys::SherpaOnnxOnlineStreamIsEndpoint(self.recognizer, self.stream) != 0;
            if is_endpoint {
                sherpa_rs_sys::SherpaOnnxOnlineStreamReset(self.recognizer, self.stream);
            }
            StreamingResult { text, is_endpoint }
        }
    }

    /// Drops any in-progress hypothesis and starts a fresh utterance (e.g. on session restart).
    fn reset(&mut self) {
        // SAFETY: recognizer/stream are live for the lifetime of `self`.
        unsafe { sherpa_rs_sys::SherpaOnnxOnlineStreamReset(self.recognizer, self.stream) };
    }
}

impl Drop for StreamingTransducerEngine {
    fn drop(&mut self) {
        // SAFETY: both handles were created by sherpa and are destroyed exactly once here — the
        // stream before the recognizer that owns it.
        unsafe {
            sherpa_rs_sys::SherpaOnnxDestroyOnlineStream(self.stream);
            sherpa_rs_sys::SherpaOnnxDestroyOnlineRecognizer(self.recognizer);
        }
    }
}

/// Turns a path into a C string, erroring on an interior NUL.
fn path_cstring(p: &Path) -> Result<CString> {
    CString::new(p.to_string_lossy().as_bytes())
        .map_err(|_| WispError::Engine(format!("path has an interior NUL: {}", p.display())))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use wisp_core::engine::StreamingAsrEngine;

    /// End-to-end (real streaming inference): loads a streaming Zipformer transducer and feeds a
    /// fixture WAV in small chunks, asserting it produces text and reaches an endpoint. Skip-guarded
    /// behind env vars so `cargo test` is clean where the (multi-hundred-MB) model isn't present:
    ///   WISP_STREAMING_TEST_DIR=/path/to/sherpa-onnx-streaming-zipformer-*   (encoder/decoder/joiner/tokens)
    ///   WISP_STREAMING_TEST_WAV=/path/to/16khz-mono.wav
    #[test]
    fn streams_fixture_wav_when_assets_present() {
        let (Some(dir), Some(wav)) = (
            std::env::var_os("WISP_STREAMING_TEST_DIR"),
            std::env::var_os("WISP_STREAMING_TEST_WAV"),
        ) else {
            return;
        };
        let dir = PathBuf::from(dir);
        let find = |needle: &str| {
            let mut matches: Vec<PathBuf> = std::fs::read_dir(&dir)
                .expect("read model dir")
                .filter_map(|e| e.ok().map(|e| e.path()))
                .filter(|p| {
                    p.file_name()
                        .is_some_and(|n| n.to_string_lossy().contains(needle))
                })
                .collect();
            // Prefer the smaller int8 variant when both precisions are present.
            matches.sort_by_key(|p| !p.to_string_lossy().contains("int8"));
            matches
                .into_iter()
                .next()
                .unwrap_or_else(|| panic!("no model file containing {needle:?} in {dir:?}"))
        };
        let (encoder, decoder, joiner) = (find("encoder"), find("decoder"), find("joiner"));
        let tokens = dir.join("tokens.txt");

        let mut reader = hound::WavReader::open(PathBuf::from(wav)).expect("open fixture wav");
        let audio: Vec<f32> = reader
            .samples::<i16>()
            .map(|s| s.expect("read sample") as f32 / 32768.0)
            .collect();

        let mut engine = StreamingTransducerEngine::new(&encoder, &decoder, &joiner, &tokens, 4)
            .expect("load streaming model");

        // Feed ~100 ms chunks like the live pipeline would, tracking the latest hypothesis.
        let mut last_text = String::new();
        let mut saw_endpoint = false;
        for chunk in audio.chunks(1_600) {
            let r = engine.accept_waveform(16_000, chunk);
            if !r.text.is_empty() {
                last_text = r.text;
            }
            saw_endpoint |= r.is_endpoint;
        }
        // Flush trailing silence so the final utterance endpoints.
        for _ in 0..20 {
            let r = engine.accept_waveform(16_000, &[0.0; 1_600]);
            if !r.text.is_empty() {
                last_text = r.text;
            }
            saw_endpoint |= r.is_endpoint;
        }

        eprintln!("streaming transcription: {last_text:?} (endpoint seen: {saw_endpoint})");
        assert!(
            !last_text.trim().is_empty(),
            "expected a non-empty streaming transcription"
        );
        assert!(saw_endpoint, "expected at least one endpoint over the clip");
    }
}
