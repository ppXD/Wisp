//! Mock implementations of the core traits, for use in tests.
//!
//! Available to downstream crates via the `testing` feature; `wisp-core`'s own tests use them
//! directly. These are deliberately simple, deterministic, and side-effect-free.

use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::Mutex;

use crate::aec::EchoCanceller;
use crate::audio::{AudioFrame, AudioSource, AudioSourceInfo};
use crate::diarize::Diarizer;
use crate::engine::{AsrEngine, EngineInfo, TranscriptionResult};
use crate::error::{Result, WispError};
use crate::model::{ModelDescriptor, ModelId, ModelStore};
use crate::transcript::{AudioSourceKind, SpeakerId};

/// An [`AudioSource`] that yields a fixed list of frames, then ends.
#[derive(Debug, Default)]
pub struct MockAudioSource {
    frames: VecDeque<AudioFrame>,
}

impl MockAudioSource {
    /// Creates a source that will yield `frames` in order.
    pub fn new(frames: impl IntoIterator<Item = AudioFrame>) -> Self {
        Self {
            frames: frames.into_iter().collect(),
        }
    }
}

impl AudioSource for MockAudioSource {
    fn info(&self) -> AudioSourceInfo {
        AudioSourceInfo {
            kind: AudioSourceKind::File,
            name: "mock".to_owned(),
        }
    }

    fn next_frame(&mut self) -> Result<Option<AudioFrame>> {
        Ok(self.frames.pop_front())
    }
}

/// An [`AsrEngine`] that returns canned results in order and records call counts.
#[derive(Debug, Default)]
pub struct MockAsrEngine {
    canned: VecDeque<TranscriptionResult>,
    /// Number of times [`transcribe`](AsrEngine::transcribe) has been called.
    pub transcribe_calls: usize,
    /// Number of times [`reset`](AsrEngine::reset) has been called.
    pub reset_calls: usize,
}

impl MockAsrEngine {
    /// Creates an engine that returns `results` on successive `transcribe` calls; once exhausted
    /// it returns empty results.
    pub fn new(results: impl IntoIterator<Item = TranscriptionResult>) -> Self {
        Self {
            canned: results.into_iter().collect(),
            transcribe_calls: 0,
            reset_calls: 0,
        }
    }
}

impl AsrEngine for MockAsrEngine {
    fn info(&self) -> EngineInfo {
        EngineInfo {
            name: "mock".to_owned(),
            streaming: false,
        }
    }

    fn transcribe(&mut self, _audio: &[f32], _sample_rate: u32) -> Result<TranscriptionResult> {
        self.transcribe_calls += 1;
        Ok(self.canned.pop_front().unwrap_or_default())
    }

    fn reset(&mut self) {
        self.reset_calls += 1;
    }
}

/// An [`EchoCanceller`] that passes capture audio through unchanged, counting what it is fed.
#[derive(Debug, Default)]
pub struct MockEchoCanceller {
    /// Total number of reference samples pushed via [`push_reference`](EchoCanceller::push_reference).
    pub reference_samples: usize,
    /// Number of [`process_capture`](EchoCanceller::process_capture) calls.
    pub capture_calls: usize,
}

impl EchoCanceller for MockEchoCanceller {
    fn push_reference(&mut self, samples: &[f32]) {
        self.reference_samples += samples.len();
    }

    fn process_capture(&mut self, samples: &[f32]) -> Vec<f32> {
        self.capture_calls += 1;
        samples.to_vec()
    }
}

/// A [`Diarizer`] that assigns every segment to the same speaker.
#[derive(Debug, Default)]
pub struct NullDiarizer;

impl Diarizer for NullDiarizer {
    fn identify(&mut self, _audio: &[f32], _sample_rate: u32) -> Result<SpeakerId> {
        Ok(SpeakerId(0))
    }
}

/// An in-memory [`ModelStore`] for tests.
#[derive(Debug, Default)]
pub struct MockModelStore {
    catalog: Vec<ModelDescriptor>,
    installed: Mutex<Vec<ModelId>>,
}

impl MockModelStore {
    /// Creates a store offering `catalog`, with nothing installed.
    pub fn new(catalog: impl IntoIterator<Item = ModelDescriptor>) -> Self {
        Self {
            catalog: catalog.into_iter().collect(),
            installed: Mutex::new(Vec::new()),
        }
    }
}

impl ModelStore for MockModelStore {
    fn available(&self) -> Vec<ModelDescriptor> {
        self.catalog.clone()
    }

    fn installed(&self) -> Result<Vec<ModelId>> {
        Ok(self.installed.lock().expect("lock poisoned").clone())
    }

    fn ensure(&self, id: &ModelId) -> Result<PathBuf> {
        let known = self.catalog.iter().any(|d| d.id == *id);
        if !known {
            return Err(WispError::Model(format!("unknown model {}", id.as_str())));
        }

        let mut installed = self.installed.lock().expect("lock poisoned");
        if !installed.contains(id) {
            installed.push(id.clone());
        }

        Ok(PathBuf::from("/mock/models").join(id.as_str()))
    }

    fn remove(&self, id: &ModelId) -> Result<()> {
        self.installed
            .lock()
            .expect("lock poisoned")
            .retain(|m| m != id);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{ModelFamily, Quant};
    use std::time::Duration;

    #[test]
    fn mock_audio_source_yields_then_ends() {
        let frame = AudioFrame::new(vec![0.0; 4], 16_000, 1, Duration::ZERO);
        let mut src = MockAudioSource::new([frame.clone()]);
        assert_eq!(src.next_frame().unwrap(), Some(frame));
        assert_eq!(src.next_frame().unwrap(), None);
        assert_eq!(src.info().kind, AudioSourceKind::File);
    }

    #[test]
    fn mock_engine_returns_canned_then_empty_and_counts_calls() {
        let mut eng = MockAsrEngine::new([TranscriptionResult::empty()]);
        assert!(eng.transcribe(&[], 16_000).unwrap().segments.is_empty());
        assert!(eng.transcribe(&[], 16_000).unwrap().segments.is_empty());
        eng.reset();
        assert_eq!(eng.transcribe_calls, 2);
        assert_eq!(eng.reset_calls, 1);
        assert_eq!(eng.info().name, "mock");
    }

    #[test]
    fn null_diarizer_is_constant() {
        let mut d = NullDiarizer;
        assert_eq!(d.identify(&[], 16_000).unwrap(), SpeakerId(0));
    }

    #[test]
    fn mock_echo_canceller_passes_through_and_counts() {
        let mut aec = MockEchoCanceller::default();
        aec.push_reference(&[0.1, 0.2, 0.3]);
        let out = aec.process_capture(&[1.0, -1.0]);
        assert_eq!(out, vec![1.0, -1.0]); // passthrough
        assert_eq!(aec.reference_samples, 3);
        assert_eq!(aec.capture_calls, 1);
    }

    #[test]
    fn mock_model_store_install_lifecycle() {
        let desc = ModelDescriptor {
            id: ModelId("m1".into()),
            family: ModelFamily::Whisper,
            quant: Quant::Q5,
            display_name: "M1".into(),
            files: vec![],
            languages: vec![],
        };
        let store = MockModelStore::new([desc]);
        assert!(store.installed().unwrap().is_empty());

        let path = store.ensure(&ModelId("m1".into())).unwrap();
        assert!(path.ends_with("m1"));
        assert_eq!(store.installed().unwrap(), vec![ModelId("m1".into())]);

        assert!(store.ensure(&ModelId("nope".into())).is_err());

        store.remove(&ModelId("m1".into())).unwrap();
        assert!(store.installed().unwrap().is_empty());
    }
}
