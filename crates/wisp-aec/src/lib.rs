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

use webrtc_audio_processing::config::EchoCanceller as AecMode;
use webrtc_audio_processing::{Config, Processor};

use wisp_core::aec::EchoCanceller;
use wisp_core::error::{Result, WispError};

/// Rate the canceller runs at — the engine's 16 kHz mono input, so its output feeds the pipeline
/// directly. Matches the [`EchoCanceller`] trait contract (16 kHz mono).
const SAMPLE_RATE_HZ: u32 = 16_000;

/// Echo canceller backed by WebRTC's AEC3, operating at 16 kHz mono.
pub struct WebrtcEchoCanceller {
    processor: Processor,
    frame_size: usize,
    render_buf: Vec<f32>,
    capture_buf: Vec<f32>,
}

impl WebrtcEchoCanceller {
    /// Creates a 16 kHz mono echo canceller with Full AEC3 and an adaptive delay estimate.
    pub fn new() -> Result<Self> {
        let processor = Processor::new(SAMPLE_RATE_HZ)
            .map_err(|e| WispError::Audio(format!("create AEC processor: {e}")))?;

        processor.set_config(Config {
            echo_canceller: Some(AecMode::Full {
                stream_delay_ms: None,
            }),
            ..Default::default()
        });

        let frame_size = processor.num_samples_per_frame();
        Ok(Self {
            processor,
            frame_size,
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
        while self.capture_buf.len() >= self.frame_size {
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
        let mut aec = WebrtcEchoCanceller::new().unwrap();

        // Fewer than one 160-sample frame → nothing ready yet.
        assert!(aec.process_capture(&vec![0.0; 100]).is_empty());

        // Crossing 160 → exactly one frame emitted, remainder (40) buffered.
        assert_eq!(aec.process_capture(&vec![0.0; 100]).len(), 160);

        // 40 buffered + 120 = 160 → one more frame.
        assert_eq!(aec.process_capture(&vec![0.0; 120]).len(), 160);
    }

    #[test]
    fn reduces_a_pure_echo_after_warmup() {
        let mut aec = WebrtcEchoCanceller::new().unwrap();
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
}
