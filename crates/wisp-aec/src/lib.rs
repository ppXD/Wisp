//! WebRTC AEC3 echo cancellation, behind the [`wisp_core::EchoCanceller`] trait.
//!
//! Wraps the WebRTC audio-processing module ([`webrtc_audio_processing`]) in Full AEC3 mode with an
//! adaptive delay estimator — the right choice here because the microphone and the system-audio
//! reference are captured on independent clocks, so their relative delay is not known up front.
//!
//! The WebRTC processor only accepts fixed **10 ms** frames (160 samples at 16 kHz mono), so this
//! buffers both the render (far-end) and capture (near-end) streams and processes them one whole
//! frame at a time, carrying any sub-frame remainder to the next call.
//!
//! This crate compiles a bundled native library and is excluded from the library CI build; it is
//! verified by running the app on macOS. Building it requires `meson` and `ninja` on `PATH`
//! (`brew install meson ninja`) plus a C/C++ toolchain.

use std::env;

use webrtc_audio_processing::config::{
    EchoCanceller as AecMode, HighPassFilter, NoiseSuppression, NoiseSuppressionLevel,
};
use webrtc_audio_processing::{Config, Processor};

use wisp_core::aec::EchoCanceller;
use wisp_core::error::{Result, WispError};

/// Rate the canceller runs at — the engine's 16 kHz mono input, so its output feeds the pipeline
/// directly. Matches the [`EchoCanceller`] trait contract (16 kHz mono).
const SAMPLE_RATE_HZ: u32 = 16_000;

/// Overrides how far (in ms) the far-end reference is made to lead the captured mic before echo
/// cancellation. ScreenCaptureKit can deliver the system/reference audio later than the cpal mic;
/// delaying the capture by this much lets the reference lead, so AEC3's delay estimator sees a
/// positive delay it can lock onto. Tune per machine if speaker echo persists.
pub const REFERENCE_LEAD_MS_ENV: &str = "WISP_AEC_REFERENCE_LEAD_MS";

/// Reference-lead delay (ms) used when [`REFERENCE_LEAD_MS_ENV`] is unset or unparseable.
const DEFAULT_REFERENCE_LEAD_MS: u32 = 150;

/// Echo canceller backed by WebRTC's AEC3, operating at 16 kHz mono.
pub struct WebrtcEchoCanceller {
    processor: Processor,
    frame_size: usize,
    /// Mic samples held back so the reference leads the capture fed to AEC3 (see the env var).
    capture_delay_samples: usize,
    render_buf: Vec<f32>,
    capture_buf: Vec<f32>,
}

impl WebrtcEchoCanceller {
    /// Creates a 16 kHz mono echo canceller with Full AEC3, reading the reference-lead delay from
    /// [`REFERENCE_LEAD_MS_ENV`] (falling back to [`DEFAULT_REFERENCE_LEAD_MS`]).
    pub fn new() -> Result<Self> {
        let lead_ms = env::var(REFERENCE_LEAD_MS_ENV)
            .ok()
            .and_then(|v| v.trim().parse::<u32>().ok())
            .unwrap_or(DEFAULT_REFERENCE_LEAD_MS);
        Self::with_reference_lead_ms(lead_ms)
    }

    /// Creates a canceller with an explicit reference-lead delay in milliseconds (0 = none).
    pub fn with_reference_lead_ms(lead_ms: u32) -> Result<Self> {
        let processor = Processor::new(SAMPLE_RATE_HZ)
            .map_err(|e| WispError::Audio(format!("create AEC processor: {e}")))?;

        processor.set_config(Config {
            echo_canceller: Some(AecMode::Full {
                stream_delay_ms: None,
            }),
            // High-pass removes low-frequency rumble; noise suppression lowers the residual noise
            // floor so that, once echo is cancelled, the near-silent mic gives the ASR far less to
            // hallucinate words from.
            high_pass_filter: Some(HighPassFilter::default()),
            noise_suppression: Some(NoiseSuppression {
                level: NoiseSuppressionLevel::High,
                ..Default::default()
            }),
            ..Default::default()
        });

        let frame_size = processor.num_samples_per_frame();
        let capture_delay_samples = lead_ms as usize * SAMPLE_RATE_HZ as usize / 1000;
        Ok(Self {
            processor,
            frame_size,
            capture_delay_samples,
            render_buf: Vec::new(),
            capture_buf: Vec::new(),
        })
    }
}

impl EchoCanceller for WebrtcEchoCanceller {
    fn push_reference(&mut self, samples: &[f32]) {
        self.render_buf.extend_from_slice(samples);
        while self.render_buf.len() >= self.frame_size {
            let mut chunk: Vec<f32> = self.render_buf.drain(..self.frame_size).collect();
            // The render frame is only analyzed for echo estimation; any modification is discarded.
            let _ = self.processor.process_render_frame([chunk.as_mut_slice()]);
        }
    }

    fn process_capture(&mut self, samples: &[f32]) -> Vec<f32> {
        self.capture_buf.extend_from_slice(samples);

        let mut cleaned = Vec::with_capacity(self.capture_buf.len());
        // Keep `capture_delay_samples` of the most recent audio buffered (unprocessed), so the
        // reference — pushed in real time — leads the capture frames we feed to AEC3.
        while self.capture_buf.len() >= self.capture_delay_samples + self.frame_size {
            let mut chunk: Vec<f32> = self.capture_buf.drain(..self.frame_size).collect();
            // Mono: a single channel of exactly `frame_size` samples, modified in place.
            if self
                .processor
                .process_capture_frame([chunk.as_mut_slice()])
                .is_ok()
            {
                cleaned.extend_from_slice(&chunk);
            }
        }
        cleaned
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constructs() {
        assert!(WebrtcEchoCanceller::new().is_ok());
    }

    #[test]
    fn buffers_sub_frame_capture_until_a_full_frame() {
        let mut aec = WebrtcEchoCanceller::with_reference_lead_ms(0).unwrap();

        // Fewer than one 160-sample frame → nothing ready yet.
        assert!(aec.process_capture(&vec![0.0; 100]).is_empty());

        // Crossing 160 → exactly one frame emitted, remainder (40) buffered.
        assert_eq!(aec.process_capture(&vec![0.0; 100]).len(), 160);

        // 40 buffered + 120 = 160 → one more frame.
        assert_eq!(aec.process_capture(&vec![0.0; 120]).len(), 160);
    }

    #[test]
    fn reduces_a_pure_echo_after_warmup() {
        let mut aec = WebrtcEchoCanceller::with_reference_lead_ms(0).unwrap();
        let frame = 160;

        // A steady tone played as the far-end reference and re-heard verbatim by the mic — i.e. a
        // perfect echo with no near-end voice. After warmup the AEC should strongly attenuate it.
        let tone: Vec<f32> = (0..frame * 60)
            .map(|i| (i as f32 * 0.17).sin() * 0.5)
            .collect();

        let (mut last_in, mut last_out) = (0.0f32, 0.0f32);
        for chunk in tone.chunks(frame) {
            aec.push_reference(chunk);
            let cleaned = aec.process_capture(chunk);
            last_in = chunk.iter().map(|x| x * x).sum();
            last_out = cleaned.iter().map(|x| x * x).sum();
        }

        assert!(
            last_out < last_in * 0.5,
            "echo should be attenuated after warmup: in={last_in}, out={last_out}"
        );
    }

    #[test]
    fn reference_lead_holds_back_capture_by_the_delay() {
        // 10 ms lead at 16 kHz = 160 samples = exactly one frame held back.
        let mut aec = WebrtcEchoCanceller::with_reference_lead_ms(10).unwrap();

        // First 160 samples are all within the delay reserve → nothing emitted yet.
        assert!(aec.process_capture(&vec![0.0; 160]).is_empty());

        // Another 160 → now 320 buffered, one frame can be emitted while 160 stays reserved.
        assert_eq!(aec.process_capture(&vec![0.0; 160]).len(), 160);
    }

    #[test]
    fn reference_lead_env_var_name_is_pinned() {
        // Renaming this breaks any operator who tuned the reference lead via env — pin it.
        assert_eq!(REFERENCE_LEAD_MS_ENV, "WISP_AEC_REFERENCE_LEAD_MS");
    }
}
