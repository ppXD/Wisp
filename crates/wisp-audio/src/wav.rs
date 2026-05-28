//! A file-backed [`AudioSource`] that decodes WAV into [`AudioFrame`]s.

use std::collections::VecDeque;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;
use std::time::Duration;

use wisp_core::audio::{AudioFrame, AudioSource, AudioSourceInfo};
use wisp_core::error::{Result, WispError};
use wisp_core::transcript::AudioSourceKind;

/// Duration of each emitted frame.
const CHUNK_MS: u64 = 100;

/// An [`AudioSource`] that decodes a WAV file up front and yields it as ~100 ms frames.
///
/// Primarily a deterministic, hardware-free source for tests and file transcription.
pub struct WavSource {
    frames: VecDeque<AudioFrame>,
    info: AudioSourceInfo,
}

impl WavSource {
    /// Opens and decodes the WAV file at `path`.
    ///
    /// Integer samples are normalized to `[-1.0, 1.0)`; float samples are taken as-is.
    pub fn open(path: &Path) -> Result<Self> {
        let reader = hound::WavReader::open(path)
            .map_err(|e| WispError::Audio(format!("open wav {}: {e}", path.display())))?;
        let spec = reader.spec();
        let samples = read_samples(reader)?;
        let frames = chunk_frames(samples, spec.sample_rate, spec.channels);

        let name = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("wav")
            .to_owned();

        Ok(Self {
            frames,
            info: AudioSourceInfo {
                kind: AudioSourceKind::File,
                name,
            },
        })
    }
}

impl AudioSource for WavSource {
    fn info(&self) -> AudioSourceInfo {
        self.info.clone()
    }

    fn next_frame(&mut self) -> Result<Option<AudioFrame>> {
        Ok(self.frames.pop_front())
    }
}

fn read_samples(reader: hound::WavReader<BufReader<File>>) -> Result<Vec<f32>> {
    let spec = reader.spec();
    let mut reader = reader;

    match spec.sample_format {
        hound::SampleFormat::Float => reader
            .samples::<f32>()
            .collect::<std::result::Result<Vec<f32>, _>>()
            .map_err(|e| WispError::Audio(format!("read float wav: {e}"))),
        hound::SampleFormat::Int => {
            let divisor = 2f64.powi(i32::from(spec.bits_per_sample.max(1)) - 1);
            reader
                .samples::<i32>()
                .map(|s| s.map(|v| (f64::from(v) / divisor) as f32))
                .collect::<std::result::Result<Vec<f32>, _>>()
                .map_err(|e| WispError::Audio(format!("read int wav: {e}")))
        }
    }
}

fn chunk_frames(samples: Vec<f32>, sample_rate: u32, channels: u16) -> VecDeque<AudioFrame> {
    let channels = channels.max(1) as usize;
    let instants_per_chunk = (u64::from(sample_rate) * CHUNK_MS / 1000).max(1) as usize;
    let per_chunk = instants_per_chunk * channels;

    let mut frames = VecDeque::new();
    let mut instants_emitted: u64 = 0;

    for chunk in samples.chunks(per_chunk) {
        let timestamp = if sample_rate == 0 {
            Duration::ZERO
        } else {
            Duration::from_secs_f64(instants_emitted as f64 / f64::from(sample_rate))
        };
        frames.push_back(AudioFrame::new(
            chunk.to_vec(),
            sample_rate,
            channels as u16,
            timestamp,
        ));
        instants_emitted += (chunk.len() / channels) as u64;
    }

    frames
}

#[cfg(test)]
mod tests {
    use super::*;
    use hound::{SampleFormat, WavSpec, WavWriter};

    fn write_int_wav(path: &Path, spec: WavSpec, samples: &[i32]) {
        let mut writer = WavWriter::create(path, spec).unwrap();
        for &s in samples {
            writer.write_sample(s).unwrap();
        }
        writer.finalize().unwrap();
    }

    fn write_float_wav(path: &Path, spec: WavSpec, samples: &[f32]) {
        let mut writer = WavWriter::create(path, spec).unwrap();
        for &s in samples {
            writer.write_sample(s).unwrap();
        }
        writer.finalize().unwrap();
    }

    #[test]
    fn decodes_16bit_mono_and_chunks_by_100ms() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("mono.wav");
        let spec = WavSpec {
            channels: 1,
            sample_rate: 16_000,
            bits_per_sample: 16,
            sample_format: SampleFormat::Int,
        };
        // 0.25 s => 4000 samples; constant 16384 normalizes to 0.5.
        write_int_wav(&path, spec, &vec![16_384; 4_000]);

        let mut src = WavSource::open(&path).unwrap();
        assert_eq!(src.info().kind, AudioSourceKind::File);

        let mut total = 0usize;
        let mut frame_count = 0usize;
        let mut first_val = None;
        while let Some(f) = src.next_frame().unwrap() {
            assert_eq!(f.sample_rate, 16_000);
            assert_eq!(f.channels, 1);
            if first_val.is_none() {
                first_val = f.samples.first().copied();
            }
            total += f.samples.len();
            frame_count += 1;
        }

        assert_eq!(total, 4_000);
        // 0.25 s / 0.1 s => 3 frames (100 ms, 100 ms, 50 ms).
        assert_eq!(frame_count, 3);
        assert!((first_val.unwrap() - 0.5).abs() < 1e-3);
    }

    #[test]
    fn decodes_float_stereo() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("stereo.wav");
        let spec = WavSpec {
            channels: 2,
            sample_rate: 48_000,
            bits_per_sample: 32,
            sample_format: SampleFormat::Float,
        };
        let samples: Vec<f32> = (0..20).map(|i| i as f32 / 100.0).collect();
        write_float_wav(&path, spec, &samples);

        let mut src = WavSource::open(&path).unwrap();
        let mut total = 0usize;
        while let Some(f) = src.next_frame().unwrap() {
            assert_eq!(f.sample_rate, 48_000);
            assert_eq!(f.channels, 2);
            total += f.samples.len();
        }
        assert_eq!(total, 20);
    }

    #[test]
    fn missing_file_errors() {
        let result = WavSource::open(Path::new("/nonexistent/x.wav"));
        assert!(matches!(result, Err(WispError::Audio(_))));
    }
}
