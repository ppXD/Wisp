//! Transcript export to common text/subtitle formats.
//!
//! Pure formatting over [`TranscriptSegment`]s, so it's fully unit-tested and reused by any shell
//! (the desktop app today, a CLI later).

use std::time::Duration;

use crate::transcript::TranscriptSegment;

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
        .map(|s| s.text.trim())
        .filter(|t| !t.is_empty())
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
            "{index}\n{} --> {}\n{text}\n\n",
            timestamp(segment.start, ','),
            timestamp(segment.end, ','),
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
            "{} --> {}\n{text}\n\n",
            timestamp(segment.start, '.'),
            timestamp(segment.end, '.'),
        ));
    }
    out
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
    use crate::transcript::AudioSourceKind;

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

    #[test]
    fn empty_segments_produce_minimal_output() {
        assert_eq!(format_transcript(&[], ExportFormat::Txt), "");
        assert_eq!(format_transcript(&[], ExportFormat::Srt), "");
        assert_eq!(format_transcript(&[], ExportFormat::Vtt), "WEBVTT\n\n");
    }
}
