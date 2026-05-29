//! A media-file [`AudioSource`] backed by symphonia.
//!
//! Decodes any symphonia-supported file — wav, mp3, m4a/aac, alac, flac, ogg/vorbis, and the audio
//! track of mp4/mov/webm — into [`AudioFrame`]s at the file's native rate/channels (the pipeline
//! converts to 16 kHz mono downstream). Pure Rust; no ffmpeg.

use std::collections::VecDeque;
use std::fs::File;
use std::path::Path;
use std::time::Duration;

use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::{DecoderOptions, CODEC_TYPE_NULL};
use symphonia::core::errors::Error as SymphoniaError;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

use wisp_core::audio::{AudioFrame, AudioSource, AudioSourceInfo};
use wisp_core::error::{Result, WispError};
use wisp_core::transcript::AudioSourceKind;

use crate::dsp::{chunk_into_frames, FRAME_CHUNK_MS};

/// An [`AudioSource`] that decodes a media file up front and yields it as ~100 ms frames.
pub struct MediaSource {
    frames: VecDeque<AudioFrame>,
    info: AudioSourceInfo,
    duration: Duration,
}

impl MediaSource {
    /// Opens and fully decodes the media file at `path`.
    pub fn open(path: &Path) -> Result<Self> {
        let (samples, sample_rate, channels) = decode(path)?;
        if samples.is_empty() {
            return Err(WispError::Audio(format!(
                "no audio decoded from {}",
                path.display()
            )));
        }

        let instants = samples.len() / (channels.max(1) as usize);
        let duration = if sample_rate == 0 {
            Duration::ZERO
        } else {
            Duration::from_secs_f64(instants as f64 / f64::from(sample_rate))
        };

        let frames = chunk_into_frames(&samples, sample_rate, channels, FRAME_CHUNK_MS).into();
        let name = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("media")
            .to_owned();

        Ok(Self {
            frames,
            info: AudioSourceInfo {
                kind: AudioSourceKind::File,
                name,
            },
            duration,
        })
    }

    /// Total decoded duration of the clip.
    pub fn duration(&self) -> Duration {
        self.duration
    }
}

impl AudioSource for MediaSource {
    fn info(&self) -> AudioSourceInfo {
        self.info.clone()
    }

    fn next_frame(&mut self) -> Result<Option<AudioFrame>> {
        Ok(self.frames.pop_front())
    }
}

/// Decodes `path` to `(interleaved f32 samples, sample_rate, channels)`.
fn decode(path: &Path) -> Result<(Vec<f32>, u32, u16)> {
    let file =
        File::open(path).map_err(|e| WispError::Audio(format!("open {}: {e}", path.display())))?;
    let stream = MediaSourceStream::new(Box::new(file), Default::default());

    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }

    let probed = symphonia::default::get_probe()
        .format(
            &hint,
            stream,
            &FormatOptions::default(),
            &MetadataOptions::default(),
        )
        .map_err(|e| WispError::Audio(format!("probe {}: {e}", path.display())))?;
    let mut format = probed.format;

    let track = format
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
        .ok_or_else(|| WispError::Audio("no decodable audio track".to_owned()))?;
    let track_id = track.id;

    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &DecoderOptions::default())
        .map_err(|e| WispError::Audio(format!("unsupported codec: {e}")))?;

    let mut samples: Vec<f32> = Vec::new();
    let mut sample_rate = track.codec_params.sample_rate.unwrap_or(0);
    let mut channels = track.codec_params.channels.map_or(0, |c| c.count() as u16);

    loop {
        let packet = match format.next_packet() {
            Ok(packet) => packet,
            Err(SymphoniaError::IoError(_)) => break, // end of stream
            Err(e) => return Err(WispError::Audio(format!("read packet: {e}"))),
        };
        if packet.track_id() != track_id {
            continue;
        }
        match decoder.decode(&packet) {
            Ok(decoded) => {
                let spec = *decoded.spec();
                sample_rate = spec.rate;
                channels = spec.channels.count() as u16;
                let mut buf = SampleBuffer::<f32>::new(decoded.capacity() as u64, spec);
                buf.copy_interleaved_ref(decoded);
                samples.extend_from_slice(buf.samples());
            }
            Err(SymphoniaError::DecodeError(_)) => continue, // skip a corrupt packet
            Err(e) => return Err(WispError::Audio(format!("decode: {e}"))),
        }
    }

    Ok((samples, sample_rate, channels))
}

#[cfg(test)]
mod tests {
    use super::*;
    use hound::{SampleFormat, WavSpec, WavWriter};

    /// Integration: decode a real (generated) WAV through the symphonia path and confirm it yields
    /// the right number of 16 kHz mono samples in frames.
    #[test]
    fn decodes_generated_wav() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("clip.wav");
        let spec = WavSpec {
            channels: 1,
            sample_rate: 16_000,
            bits_per_sample: 16,
            sample_format: SampleFormat::Int,
        };
        let mut writer = WavWriter::create(&path, spec).unwrap();
        for _ in 0..8_000 {
            writer.write_sample(8_192i16).unwrap(); // 0.5 s, constant
        }
        writer.finalize().unwrap();

        let mut src = MediaSource::open(&path).unwrap();
        assert_eq!(src.info().kind, AudioSourceKind::File);
        assert_eq!(src.duration().as_millis(), 500); // 8000 samples @ 16 kHz

        let mut total = 0usize;
        while let Some(frame) = src.next_frame().unwrap() {
            assert_eq!(frame.sample_rate, 16_000);
            assert_eq!(frame.channels, 1);
            total += frame.samples.len();
        }
        assert_eq!(total, 8_000);
    }

    #[test]
    fn missing_file_errors() {
        assert!(matches!(
            MediaSource::open(Path::new("/nonexistent/x.mp3")),
            Err(WispError::Audio(_))
        ));
    }
}
