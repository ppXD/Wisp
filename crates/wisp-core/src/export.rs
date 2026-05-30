//! Transcript export to common text/subtitle formats.
//!
//! Pure formatting over [`TranscriptSegment`]s, so it's fully unit-tested and reused by any shell
//! (the desktop app today, a CLI later).

use std::time::Duration;

use crate::transcript::{SpeakerId, TranscriptSegment};

/// A transcript export format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ExportFormat {
    /// Plain text — one line per segment, no timestamps.
    Txt,
    /// SubRip subtitles (numbered cues, `HH:MM:SS,mmm` timestamps).
    Srt,
    /// WebVTT subtitles (`WEBVTT` header, `HH:MM:SS.mmm` timestamps).
    Vtt,
}

impl ExportFormat {
    /// Parses a format from its name/extension (case-insensitive): `txt`/`text`, `srt`, `vtt`.
    pub fn from_name(name: &str) -> Option<Self> {
        match name.to_ascii_lowercase().as_str() {
            "txt" | "text" => Some(Self::Txt),
            "srt" => Some(Self::Srt),
            "vtt" => Some(Self::Vtt),
            _ => None,
        }
    }

    /// The file extension for this format.
    pub fn extension(self) -> &'static str {
        match self {
            Self::Txt => "txt",
            Self::Srt => "srt",
            Self::Vtt => "vtt",
        }
    }
}

/// Renders `segments` into `format` as a single string. Empty/whitespace-only segments are skipped.
pub fn format_transcript(segments: &[TranscriptSegment], format: ExportFormat) -> String {
    match format {
        ExportFormat::Txt => format_txt(segments),
        ExportFormat::Srt => format_srt(segments),
        ExportFormat::Vtt => format_vtt(segments),
    }
}

fn format_txt(segments: &[TranscriptSegment]) -> String {
    let mut out: String = segments
        .iter()
        .filter(|s| !s.text.trim().is_empty())
        .map(labelled_text)
        .collect::<Vec<_>>()
        .join("\n");
    if !out.is_empty() {
        out.push('\n');
    }
    out
}

fn format_srt(segments: &[TranscriptSegment]) -> String {
    let mut out = String::new();
    let mut index = 1;
    for segment in segments {
        let text = segment.text.trim();
        if text.is_empty() {
            continue;
        }
        out.push_str(&format!(
            "{index}\n{} --> {}\n{}\n\n",
            timestamp(segment.start, ','),
            timestamp(segment.end, ','),
            labelled_text(segment),
        ));
        index += 1;
    }
    out
}

fn format_vtt(segments: &[TranscriptSegment]) -> String {
    let mut out = String::from("WEBVTT\n\n");
    for segment in segments {
        let text = segment.text.trim();
        if text.is_empty() {
            continue;
        }
        out.push_str(&format!(
            "{} --> {}\n{}\n\n",
            timestamp(segment.start, '.'),
            timestamp(segment.end, '.'),
            vtt_cue_body(segment),
        ));
    }
    out
}

/// VTT cue body: when the segment carries word-level timings, emit them as WebVTT inline timestamps
/// (`<00:00:00.500>word`) for karaoke-style highlighting; otherwise the plain labelled text. The
/// speaker label, if any, still prefixes the cue.
fn vtt_cue_body(segment: &TranscriptSegment) -> String {
    if segment.words.is_empty() {
        return labelled_text(segment);
    }

    let mut timed = String::new();
    for (i, word) in segment.words.iter().enumerate() {
        // The first word follows the cue's own start time, so drop its leading space.
        let text = if i == 0 {
            word.text.trim_start()
        } else {
            word.text.as_str()
        };
        timed.push_str(&format!("<{}>{text}", timestamp(word.start, '.')));
    }

    match segment.speaker {
        Some(speaker) => format!("{}: {timed}", speaker_label(speaker)),
        None => timed,
    }
}

/// Segment text, prefixed with its speaker (`Speaker 1: …`) once diarization has labelled it; the
/// raw text otherwise. Assumes the caller already skipped blank segments.
fn labelled_text(segment: &TranscriptSegment) -> String {
    let text = segment.text.trim();
    match segment.speaker {
        Some(speaker) => format!("{}: {text}", speaker_label(speaker)),
        None => text.to_owned(),
    }
}

/// Human label for a speaker, 1-based so `SpeakerId(0)` reads as `Speaker 1`.
fn speaker_label(id: SpeakerId) -> String {
    format!("Speaker {}", id.0 + 1)
}

/// `HH:MM:SS<sep>mmm` — `sep` is `,` for SRT and `.` for VTT.
fn timestamp(d: Duration, sep: char) -> String {
    let ms = d.as_millis();
    let hours = ms / 3_600_000;
    let minutes = (ms % 3_600_000) / 60_000;
    let seconds = (ms % 60_000) / 1_000;
    let millis = ms % 1_000;
    format!("{hours:02}:{minutes:02}:{seconds:02}{sep}{millis:03}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transcript::{AudioSourceKind, Word};

    fn seg(text: &str, start_ms: u64, end_ms: u64) -> TranscriptSegment {
        TranscriptSegment::new(
            0,
            text,
            Duration::from_millis(start_ms)..Duration::from_millis(end_ms),
            AudioSourceKind::File,
        )
    }

    #[test]
    fn from_name_parses_known_formats() {
        assert_eq!(ExportFormat::from_name("TXT"), Some(ExportFormat::Txt));
        assert_eq!(ExportFormat::from_name("srt"), Some(ExportFormat::Srt));
        assert_eq!(ExportFormat::from_name("vtt"), Some(ExportFormat::Vtt));
        assert_eq!(ExportFormat::from_name("docx"), None);
    }

    #[test]
    fn txt_is_plain_lines_skipping_blanks() {
        let segs = vec![
            seg("hello", 0, 1000),
            seg("   ", 1000, 2000),
            seg("world", 2000, 3000),
        ];
        assert_eq!(
            format_transcript(&segs, ExportFormat::Txt),
            "hello\nworld\n"
        );
    }

    #[test]
    fn srt_has_numbered_cues_and_comma_timestamps() {
        let segs = vec![seg("hi", 0, 2_500), seg("there", 3_000, 4_200)];
        let srt = format_transcript(&segs, ExportFormat::Srt);
        assert_eq!(
            srt,
            "1\n00:00:00,000 --> 00:00:02,500\nhi\n\n\
             2\n00:00:03,000 --> 00:00:04,200\nthere\n\n"
        );
    }

    #[test]
    fn vtt_has_header_and_dot_timestamps() {
        let segs = vec![seg("hi", 60_000, 61_000)];
        let vtt = format_transcript(&segs, ExportFormat::Vtt);
        assert_eq!(vtt, "WEBVTT\n\n00:01:00.000 --> 00:01:01.000\nhi\n\n");
    }

    fn seg_with_speaker(text: &str, start_ms: u64, end_ms: u64, speaker: u32) -> TranscriptSegment {
        let mut s = seg(text, start_ms, end_ms);
        s.speaker = Some(SpeakerId(speaker));
        s
    }

    #[test]
    fn prefixes_speaker_label_when_diarized() {
        // SpeakerId is 0-based; the label is 1-based.
        let segs = vec![
            seg_with_speaker("hello", 0, 1_000, 0),
            seg_with_speaker("hi back", 1_000, 2_000, 1),
        ];
        assert_eq!(
            format_transcript(&segs, ExportFormat::Txt),
            "Speaker 1: hello\nSpeaker 2: hi back\n"
        );
        assert_eq!(
            format_transcript(&segs, ExportFormat::Vtt),
            "WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nSpeaker 1: hello\n\n\
             00:00:01.000 --> 00:00:02.000\nSpeaker 2: hi back\n\n"
        );
    }

    #[test]
    fn vtt_emits_word_level_timestamps_when_present() {
        let mut s = seg("Hello world", 0, 1_000);
        s.words = vec![
            Word {
                text: " Hello".into(),
                start: Duration::from_millis(0),
                end: Duration::from_millis(400),
            },
            Word {
                text: " world".into(),
                start: Duration::from_millis(500),
                end: Duration::from_millis(1_000),
            },
        ];
        assert_eq!(
            format_transcript(&[s], ExportFormat::Vtt),
            "WEBVTT\n\n00:00:00.000 --> 00:00:01.000\n\
             <00:00:00.000>Hello<00:00:00.500> world\n\n"
        );
    }

    #[test]
    fn word_timestamps_keep_the_speaker_prefix() {
        let mut s = seg_with_speaker("Hi", 0, 500, 0);
        s.words = vec![Word {
            text: "Hi".into(),
            start: Duration::from_millis(0),
            end: Duration::from_millis(500),
        }];
        let vtt = format_transcript(&[s], ExportFormat::Vtt);
        assert!(vtt.contains("Speaker 1: <00:00:00.000>Hi"));
    }

    #[test]
    fn empty_segments_produce_minimal_output() {
        assert_eq!(format_transcript(&[], ExportFormat::Txt), "");
        assert_eq!(format_transcript(&[], ExportFormat::Srt), "");
        assert_eq!(format_transcript(&[], ExportFormat::Vtt), "WEBVTT\n\n");
    }
}
