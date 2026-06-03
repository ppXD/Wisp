//! Echo-cancelling microphone source.
//!
//! Wraps a microphone [`AudioSource`] and a far-end reference channel, running both through an
//! [`EchoCanceller`] so the emitted "me" stream no longer contains an echo of the system audio the
//! mic re-heard from the speakers. The wrapped pipeline is unchanged — this is just another source.

use wisp_core::aec::EchoCanceller;
use wisp_core::audio::{AudioFrame, AudioSource, AudioSourceInfo};
use wisp_core::channel::FrameReceiver;
use wisp_core::error::Result;

use crate::dsp::{Resampler, TARGET_SAMPLE_RATE};

/// An [`AudioSource`] that echo-cancels a microphone against a far-end reference stream.
///
/// `reference` is typically one branch of a [`tee`](crate::tee) of the system audio. Each
/// [`next_frame`](AudioSource::next_frame) feeds any pending reference audio to the
/// [`EchoCanceller`]'s far-end, then returns the echo-cancelled microphone frame (16 kHz mono).
/// The source reports the wrapped mic's [`info`](AudioSource::info) — it is still "the microphone".
pub struct EchoCancellingSource {
    mic: Box<dyn AudioSource>,
    reference: FrameReceiver,
    canceller: Box<dyn EchoCanceller>,
    info: AudioSourceInfo,
    /// Anti-aliased resamplers to 16 kHz. The mic capture and the far-end reference are independent
    /// streams, so each keeps its own filter state.
    capture_resampler: Resampler,
    ref_resampler: Resampler,
}

impl EchoCancellingSource {
    /// Wraps `mic`, cancelling echo of the audio arriving on `reference` using `canceller`.
    pub fn new(
        mic: Box<dyn AudioSource>,
        reference: FrameReceiver,
        canceller: Box<dyn EchoCanceller>,
    ) -> Self {
        let info = mic.info();
        Self {
            mic,
            reference,
            canceller,
            info,
            capture_resampler: Resampler::new(TARGET_SAMPLE_RATE),
            ref_resampler: Resampler::new(TARGET_SAMPLE_RATE),
        }
    }

    /// Feeds all reference audio captured so far to the canceller's far-end.
    fn drain_reference(&mut self) {
        while let Some(frame) = self.reference.try_recv() {
            let resampled = self.ref_resampler.process(&frame);
            self.canceller.push_reference(&resampled);
        }
    }
}

impl AudioSource for EchoCancellingSource {
    fn info(&self) -> AudioSourceInfo {
        self.info.clone()
    }

    fn next_frame(&mut self) -> Result<Option<AudioFrame>> {
        loop {
            let Some(frame) = self.mic.next_frame()? else {
                return Ok(None);
            };

            self.drain_reference();

            let resampled = self.capture_resampler.process(&frame);
            let cleaned = self.canceller.process_capture(&resampled);
            if cleaned.is_empty() {
                continue; // canceller buffered a sub-frame; pull more mic audio
            }

            return Ok(Some(AudioFrame::new(
                cleaned,
                TARGET_SAMPLE_RATE,
                1,
                frame.timestamp,
            )));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};
    use std::time::Duration;
    use wisp_core::channel::frame_channel;
    use wisp_core::testing::MockAudioSource;
    use wisp_core::transcript::AudioSourceKind;

    fn frame16k(samples: Vec<f32>) -> AudioFrame {
        AudioFrame::new(samples, 16_000, 1, Duration::ZERO)
    }

    /// Records what it is fed and doubles capture samples, so the test can prove both paths ran.
    #[derive(Clone, Default)]
    struct RecordingCanceller {
        reference: Arc<Mutex<Vec<f32>>>,
        captured: Arc<Mutex<Vec<Vec<f32>>>>,
    }

    impl EchoCanceller for RecordingCanceller {
        fn push_reference(&mut self, samples: &[f32]) {
            self.reference.lock().unwrap().extend_from_slice(samples);
        }

        fn process_capture(&mut self, samples: &[f32]) -> Vec<f32> {
            self.captured.lock().unwrap().push(samples.to_vec());
            samples.iter().map(|s| s * 2.0).collect()
        }
    }

    #[test]
    fn cancels_mic_using_reference_then_ends() {
        let mic = Box::new(MockAudioSource::new([
            frame16k(vec![1.0, 2.0]),
            frame16k(vec![3.0]),
        ]));
        let (ref_tx, ref_rx) = frame_channel(8);
        ref_tx.send(frame16k(vec![0.5, 0.5])); // reference available before the first mic frame

        let canceller = RecordingCanceller::default();
        let reference = Arc::clone(&canceller.reference);
        let captured = Arc::clone(&canceller.captured);

        let mut src = EchoCancellingSource::new(mic, ref_rx, Box::new(canceller));

        // Reports the wrapped mic's identity (MockAudioSource reports File).
        assert_eq!(src.info().kind, AudioSourceKind::File);

        // First frame: reference drained, capture processed (doubled), 16 kHz mono out.
        let out1 = src.next_frame().unwrap().unwrap();
        assert_eq!(out1.samples, vec![2.0, 4.0]);
        assert_eq!(out1.sample_rate, 16_000);
        assert_eq!(out1.channels, 1);

        let out2 = src.next_frame().unwrap().unwrap();
        assert_eq!(out2.samples, vec![6.0]);

        assert!(src.next_frame().unwrap().is_none()); // mic ended

        assert_eq!(*reference.lock().unwrap(), vec![0.5, 0.5]); // reference was pushed
        assert_eq!(
            *captured.lock().unwrap(),
            vec![vec![1.0, 2.0], vec![3.0]] // both mic frames processed
        );
    }
}
