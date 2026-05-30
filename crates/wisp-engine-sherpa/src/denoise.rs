//! GTCRN neural denoiser via sherpa-onnx's offline speech-enhancement C API.
//!
//! sherpa-rs (0.6) doesn't wrap the denoiser, so we declare the five C functions ourselves and link
//! them from the `libsherpa-onnx-c-api` the crate already pulls in. That reuses the ONNX runtime
//! already linked for ASR — no second inference engine — and runs at GTCRN's native 16 kHz, so
//! there is no resampling.

use std::ffi::CString;
use std::path::Path;

use wisp_core::denoise::Denoiser;
use wisp_core::error::{Result, WispError};

#[allow(non_snake_case, non_camel_case_types)]
mod sys {
    use std::os::raw::{c_char, c_float, c_int};

    #[repr(C)]
    pub struct GtcrnModelConfig {
        pub model: *const c_char,
    }

    #[repr(C)]
    pub struct ModelConfig {
        pub gtcrn: GtcrnModelConfig,
        pub num_threads: c_int,
        pub debug: c_int,
        pub provider: *const c_char,
    }

    #[repr(C)]
    pub struct Config {
        pub model: ModelConfig,
    }

    #[repr(C)]
    pub struct DenoisedAudio {
        pub samples: *const c_float,
        pub n: c_int,
        pub sample_rate: c_int,
    }

    /// Opaque denoiser handle.
    pub enum OfflineSpeechDenoiser {}

    extern "C" {
        pub fn SherpaOnnxCreateOfflineSpeechDenoiser(
            config: *const Config,
        ) -> *const OfflineSpeechDenoiser;
        pub fn SherpaOnnxDestroyOfflineSpeechDenoiser(sd: *const OfflineSpeechDenoiser);
        pub fn SherpaOnnxOfflineSpeechDenoiserRun(
            sd: *const OfflineSpeechDenoiser,
            samples: *const c_float,
            n: c_int,
            sample_rate: c_int,
        ) -> *const DenoisedAudio;
        pub fn SherpaOnnxDestroyDenoisedAudio(p: *const DenoisedAudio);
    }
}

/// The downloadable GTCRN denoiser: stronger than the light built-in RNNoise on real-world noise,
/// while reusing the ONNX runtime already linked for ASR.
pub struct GtcrnDenoiser {
    handle: *const sys::OfflineSpeechDenoiser,
}

// SAFETY: the handle is used serially from a single transcription thread (the engine is moved onto
// it) and never shared across threads concurrently.
unsafe impl Send for GtcrnDenoiser {}

impl GtcrnDenoiser {
    /// Loads the GTCRN model at `model` (the downloaded `gtcrn_simple.onnx`).
    pub fn new(model: &Path) -> Result<Self> {
        let model_c = CString::new(model.to_string_lossy().as_bytes())
            .map_err(|_| WispError::Engine("denoiser model path has an interior NUL".to_owned()))?;
        let provider = CString::new("cpu").expect("static str has no interior NUL");

        // The config and the strings it points at only need to be valid for the create call —
        // sherpa-onnx copies them internally.
        let config = sys::Config {
            model: sys::ModelConfig {
                gtcrn: sys::GtcrnModelConfig {
                    model: model_c.as_ptr(),
                },
                num_threads: 1,
                debug: 0,
                provider: provider.as_ptr(),
            },
        };

        // SAFETY: `config` and the strings it references outlive this call.
        let handle = unsafe { sys::SherpaOnnxCreateOfflineSpeechDenoiser(&config) };
        if handle.is_null() {
            return Err(WispError::Engine(format!(
                "gtcrn: failed to load denoiser model {}",
                model.display()
            )));
        }
        Ok(Self { handle })
    }
}

impl Denoiser for GtcrnDenoiser {
    fn denoise(&mut self, audio: &[f32], sample_rate: u32) -> Vec<f32> {
        if audio.is_empty() {
            return Vec::new();
        }

        // SAFETY: `self.handle` is live; `audio` is a valid slice; the returned buffer is copied out
        // and freed before we return, exactly once.
        unsafe {
            let out = sys::SherpaOnnxOfflineSpeechDenoiserRun(
                self.handle,
                audio.as_ptr(),
                audio.len() as i32,
                sample_rate as i32,
            );
            if out.is_null() {
                return audio.to_vec();
            }
            let denoised = &*out;
            let len = denoised.n.max(0) as usize;
            let mut result = std::slice::from_raw_parts(denoised.samples, len).to_vec();
            sys::SherpaOnnxDestroyDenoisedAudio(out);
            // Honour the trait's same-length contract so downstream timestamps still line up (our
            // pipeline is 16 kHz, GTCRN's native rate, so any difference is sub-frame rounding).
            result.resize(audio.len(), 0.0);
            result
        }
    }
}

impl Drop for GtcrnDenoiser {
    fn drop(&mut self) {
        // SAFETY: `handle` came from `SherpaOnnxCreateOfflineSpeechDenoiser` and is freed once here.
        unsafe { sys::SherpaOnnxDestroyOfflineSpeechDenoiser(self.handle) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    /// Real GTCRN inference. Skip-guarded behind an env var so `cargo test` is clean where the model
    /// isn't available:
    ///   WISP_GTCRN_TEST_MODEL=/path/to/gtcrn_simple.onnx
    #[test]
    fn denoises_a_buffer_when_model_present() {
        let Some(model) = std::env::var_os("WISP_GTCRN_TEST_MODEL") else {
            return;
        };
        let model = PathBuf::from(model);
        if !model.exists() {
            return;
        }

        let mut denoiser = GtcrnDenoiser::new(&model).expect("load gtcrn model");
        let noisy = vec![0.05f32; 16_000]; // 1 s at 16 kHz
        let out = denoiser.denoise(&noisy, 16_000);

        assert_eq!(out.len(), noisy.len(), "length is preserved");
        assert!(out.iter().all(|s| s.is_finite()), "no NaN/inf");
    }
}
