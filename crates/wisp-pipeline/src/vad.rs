//! Voice-activity detection.

/// Classifies a chunk of 16 kHz mono audio as speech or non-speech.
///
/// A trait so a simple energy gate today can be replaced by a learned detector (e.g. Silero via
/// sherpa-onnx) later without changing the pipeline.
pub trait Vad: Send {
    /// Returns `true` if `samples` (16 kHz mono) contains speech.
    fn is_speech(&mut self, samples: &[f32]) -> bool;
}

/// A simple energy-gate VAD: speech when the chunk's RMS amplitude meets a threshold.
#[derive(Debug, Clone, Copy)]
pub struct EnergyVad {
    threshold: f32,
}

impl EnergyVad {
    /// Creates an energy gate that reports speech when RMS `>= threshold`.
    pub fn new(threshold: f32) -> Self {
        Self { threshold }
    }
}

impl Default for EnergyVad {
    fn default() -> Self {
        Self { threshold: 0.01 }
    }
}

impl Vad for EnergyVad {
    fn is_speech(&mut self, samples: &[f32]) -> bool {
        if samples.is_empty() {
            return false;
        }
        let sum_sq: f32 = samples.iter().map(|x| x * x).sum();
        let rms = (sum_sq / samples.len() as f32).sqrt();
        rms >= self.threshold
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loud_chunk_is_speech() {
        let mut vad = EnergyVad::new(0.01);
        assert!(vad.is_speech(&[0.5; 100]));
    }

    #[test]
    fn silent_chunk_is_not_speech() {
        let mut vad = EnergyVad::new(0.01);
        assert!(!vad.is_speech(&[0.0; 100]));
    }

    #[test]
    fn empty_chunk_is_not_speech() {
        let mut vad = EnergyVad::default();
        assert!(!vad.is_speech(&[]));
    }
}
