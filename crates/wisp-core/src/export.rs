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
    /// Plain text — segments merged into paragraphs, no timestamps.
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

/// A run of consecutive segments merged into one readable block — what the UI shows as a line and
/// what plain-text export emits as a paragraph. Subtitles (SRT/VTT) keep the original fine segments.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Paragraph {
    pub start: Duration,
    pub end: Duration,
    pub speaker: Option<SpeakerId>,
    pub text: String,
}

/// Longest silence kept inside one paragraph; a longer pause starts a new paragraph.
pub const PARAGRAPH_GAP: Duration = Duration::from_millis(1500);

/// Soft cap on a paragraph's length in characters; the next segment then starts a fresh paragraph.
const MAX_PARAGRAPH_CHARS: usize = 240;

/// Merges consecutive segments into paragraphs, breaking on a speaker change, a pause longer than
/// [`PARAGRAPH_GAP`], or once a paragraph passes [`MAX_PARAGRAPH_CHARS`]. Blank segments are skipped.
/// Engines (and our windowing) emit one segment per utterance, which reads as a wall of short lines;
/// grouping turns that into document-like paragraphs without touching the fine SRT/VTT cues.
pub fn group_paragraphs(segments: &[TranscriptSegment]) -> Vec<Paragraph> {
    let mut paragraphs: Vec<Paragraph> = Vec::new();

    for segment in segments {
        let text = segment.text.trim();
        if text.is_empty() {
            continue;
        }

        match paragraphs.last_mut() {
            Some(para) if continues(para, segment) => {
                para.text = join_text(&para.text, text);
                para.end = segment.end;
            }
            _ => paragraphs.push(Paragraph {
                start: segment.start,
                end: segment.end,
                speaker: segment.speaker,
                text: text.to_owned(),
            }),
        }
    }

    paragraphs
}

/// Whether `segment` continues `para`: same speaker, a short-enough pause, and still under the cap.
fn continues(para: &Paragraph, segment: &TranscriptSegment) -> bool {
    para.speaker == segment.speaker
        && segment.start.saturating_sub(para.end) <= PARAGRAPH_GAP
        && para.text.chars().count() < MAX_PARAGRAPH_CHARS
}

/// Joins `next` onto `prev`, inserting a space only when `next` opens with an ASCII word character —
/// so spaced scripts stay readable while CJK runs together.
fn join_text(prev: &str, next: &str) -> String {
    if next
        .chars()
        .next()
        .is_some_and(|c| c.is_ascii_alphanumeric())
    {
        format!("{prev} {next}")
    } else {
        format!("{prev}{next}")
    }
}

/// Plain text: one paragraph per block (speaker-labelled when diarized), a blank line between blocks.
fn format_txt(segments: &[TranscriptSegment]) -> String {
    let mut out: String = group_paragraphs(segments)
        .iter()
        .map(labelled_paragraph)
        .collect::<Vec<_>>()
        .join("\n\n");
    if !out.is_empty() {
        out.push('\n');
    }
    out
}

/// A paragraph's text, prefixed with its speaker (`Speaker 1: …`) when diarization labelled it.
fn labelled_paragraph(para: &Paragraph) -> String {
    match para.speaker {
        Some(speaker) => format!("{}: {}", speaker_label(speaker), para.text),
        None => para.text.clone(),
    }
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
    fn txt_merges_segments_into_a_paragraph_skipping_blanks() {
        let segs = vec![
            seg("hello", 0, 1000),
            seg("   ", 1000, 2000),
            seg("world", 2000, 3000),
        ];
        // Blank skipped; the two within PARAGRAPH_GAP merge into one paragraph (space before ASCII).
        assert_eq!(format_transcript(&segs, ExportFormat::Txt), "hello world\n");
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
            "Speaker 1: hello\n\nSpeaker 2: hi back\n"
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

    #[test]
    fn paragraphs_break_on_long_pause_and_speaker_change() {
        // Same speaker, short gap → one paragraph.
        assert_eq!(
            group_paragraphs(&[seg("a", 0, 500), seg("b", 800, 1200)]).len(),
            1
        );
        // A pause longer than PARAGRAPH_GAP → two paragraphs.
        assert_eq!(
            group_paragraphs(&[seg("a", 0, 500), seg("b", 3000, 3500)]).len(),
            2
        );
        // A speaker change splits even across a short gap.
        let spk = group_paragraphs(&[
            seg_with_speaker("a", 0, 500, 0),
            seg_with_speaker("b", 600, 1000, 1),
        ]);
        assert_eq!(spk.len(), 2);
    }

    #[test]
    fn cjk_segments_join_without_spaces() {
        let p = group_paragraphs(&[seg("你好", 0, 500), seg("世界", 700, 1200)]);
        assert_eq!(p.len(), 1);
        assert_eq!(p[0].text, "你好世界");
    }
}
