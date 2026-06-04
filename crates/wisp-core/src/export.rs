//! Transcript export to common text/subtitle formats.
//!
//! Pure formatting over [`TranscriptSegment`]s, so it's fully unit-tested and reused by any shell
//! (the desktop app today, a CLI later).

use std::time::Duration;

use crate::transcript::{AudioSourceKind, SpeakerId, TranscriptSegment};

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
    /// Structured Markdown — a meeting document: YAML front-matter, optional summary, then a diarized,
    /// timestamped, speaker-labelled transcript. Human- *and* agent-readable.
    Markdown,
}

impl ExportFormat {
    /// Parses a format from its name/extension (case-insensitive): `txt`/`text`, `srt`, `vtt`, `md`.
    pub fn from_name(name: &str) -> Option<Self> {
        match name.to_ascii_lowercase().as_str() {
            "txt" | "text" => Some(Self::Txt),
            "srt" => Some(Self::Srt),
            "vtt" => Some(Self::Vtt),
            "md" | "markdown" => Some(Self::Markdown),
            _ => None,
        }
    }

    /// The file extension for this format.
    pub fn extension(self) -> &'static str {
        match self {
            Self::Txt => "txt",
            Self::Srt => "srt",
            Self::Vtt => "vtt",
            Self::Markdown => "md",
        }
    }
}

/// Renders `segments` into `format` as a single string. Empty/whitespace-only segments are skipped.
/// Markdown uses empty [`MeetingMeta`]; call [`format_markdown`] directly to pass meeting metadata.
pub fn format_transcript(segments: &[TranscriptSegment], format: ExportFormat) -> String {
    match format {
        ExportFormat::Txt => format_txt(segments),
        ExportFormat::Srt => format_srt(segments),
        ExportFormat::Vtt => format_vtt(segments),
        ExportFormat::Markdown => format_markdown(segments, &MeetingMeta::default()),
    }
}

/// A run of consecutive segments merged into one readable block — what the UI shows as a line and
/// what plain-text export emits as a paragraph. Subtitles (SRT/VTT) keep the original fine segments.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Paragraph {
    pub start: Duration,
    pub end: Duration,
    pub speaker: Option<SpeakerId>,
    /// Which stream this block came from — so a meeting export can label it Me / Them.
    pub source: AudioSourceKind,
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
                source: segment.source,
                text: text.to_owned(),
            }),
        }
    }

    paragraphs
}

/// Whether `segment` continues `para`: same stream and speaker, a short-enough pause, and still under
/// the cap. The source check keeps a meeting's Me / Them turns in separate blocks.
fn continues(para: &Paragraph, segment: &TranscriptSegment) -> bool {
    para.source == segment.source
        && para.speaker == segment.speaker
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

/// App-supplied context for a Markdown meeting export — the bits the pure formatter can't derive from
/// the segments (the shell knows the date, model, and any AI summary). Duration and the participant
/// list ARE derived from the segments here.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct MeetingMeta {
    /// Document title; defaults to "Meeting transcript" when absent.
    pub title: Option<String>,
    /// ISO date/time string, formatted by the shell (so core needs no clock/date dependency).
    pub date: Option<String>,
    /// The transcription model/engine used.
    pub engine: Option<String>,
    /// The transcription language.
    pub language: Option<String>,
    /// An AI-generated summary to lead the document with, when one is available.
    pub summary: Option<String>,
}

/// Renders a structured Markdown meeting document: YAML front-matter (title, date, duration,
/// participants, engine, language), an optional `## Summary`, then a `## Transcript` of diarized,
/// timestamped, Me/Them-labelled paragraphs — human-readable and clean for an LLM/agent to consume.
pub fn format_markdown(segments: &[TranscriptSegment], meta: &MeetingMeta) -> String {
    let paragraphs = group_paragraphs(segments);

    let mut out = front_matter(&paragraphs, meta);

    out.push_str(&format!(
        "# {}\n\n",
        meta.title.as_deref().unwrap_or("Meeting transcript")
    ));

    if let Some(summary) = meta
        .summary
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
    {
        out.push_str("## Summary\n\n");
        out.push_str(summary);
        out.push_str("\n\n");
    }

    out.push_str("## Transcript\n\n");
    for para in &paragraphs {
        let stamp = short_timestamp(para.start);
        match meeting_label(para) {
            Some(who) => out.push_str(&format!("**[{stamp}] {who}:** {}\n\n", para.text)),
            None => out.push_str(&format!("**[{stamp}]** {}\n\n", para.text)),
        }
    }

    out
}

/// The YAML front-matter block — only the fields present/derivable, so a bare file export stays minimal
/// while a full meeting carries its context.
fn front_matter(paragraphs: &[Paragraph], meta: &MeetingMeta) -> String {
    let mut lines: Vec<String> = Vec::new();

    lines.push(format!(
        "title: {}",
        yaml_scalar(meta.title.as_deref().unwrap_or("Meeting transcript"))
    ));
    if let Some(date) = meta.date.as_deref() {
        lines.push(format!("date: {}", yaml_scalar(date)));
    }
    if let Some(duration) = paragraphs.iter().map(|p| p.end).max() {
        lines.push(format!("duration: {}", full_timestamp(duration)));
    }

    let participants = participants(paragraphs);
    if !participants.is_empty() {
        let joined = participants
            .iter()
            .map(|p| yaml_scalar(p))
            .collect::<Vec<_>>()
            .join(", ");
        lines.push(format!("participants: [{joined}]"));
    }
    if let Some(engine) = meta.engine.as_deref() {
        lines.push(format!("engine: {}", yaml_scalar(engine)));
    }
    if let Some(language) = meta.language.as_deref() {
        lines.push(format!("language: {}", yaml_scalar(language)));
    }

    format!("---\n{}\n---\n\n", lines.join("\n"))
}

/// The distinct speaker labels in first-seen order — the meeting's participant list.
fn participants(paragraphs: &[Paragraph]) -> Vec<String> {
    let mut seen: Vec<String> = Vec::new();
    for para in paragraphs {
        if let Some(label) = meeting_label(para) {
            if !seen.contains(&label) {
                seen.push(label);
            }
        }
    }
    seen
}

/// A meeting-aware speaker label for a paragraph: the local mic is "Me"; the system/far end is "Them"
/// (with the diarized speaker number when known); any other source (a file) is just "Speaker N" when
/// diarized, else unlabelled.
fn meeting_label(para: &Paragraph) -> Option<String> {
    match para.source {
        AudioSourceKind::Microphone => Some("Me".to_owned()),
        AudioSourceKind::System => Some(match para.speaker {
            Some(speaker) => format!("Them ({})", speaker_label(speaker)),
            None => "Them".to_owned(),
        }),
        _ => para.speaker.map(speaker_label),
    }
}

/// Escapes a string as a double-quoted YAML scalar, so a colon/bracket in a title or model name can't
/// break the front-matter.
fn yaml_scalar(s: &str) -> String {
    format!("\"{}\"", s.replace('\\', "\\\\").replace('"', "\\\""))
}

/// `H:MM:SS` when there's an hour, else `M:SS` — a compact inline timestamp for the transcript.
fn short_timestamp(d: Duration) -> String {
    let secs = d.as_secs();
    let (h, m, s) = (secs / 3600, (secs % 3600) / 60, secs % 60);
    if h > 0 {
        format!("{h}:{m:02}:{s:02}")
    } else {
        format!("{m}:{s:02}")
    }
}

/// `HH:MM:SS` — the meeting's total length, for the front-matter.
fn full_timestamp(d: Duration) -> String {
    let secs = d.as_secs();
    format!(
        "{:02}:{:02}:{:02}",
        secs / 3600,
        (secs % 3600) / 60,
        secs % 60
    )
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
        assert_eq!(ExportFormat::from_name("md"), Some(ExportFormat::Markdown));
        assert_eq!(
            ExportFormat::from_name("Markdown"),
            Some(ExportFormat::Markdown)
        );
        assert_eq!(ExportFormat::Markdown.extension(), "md");
        assert_eq!(ExportFormat::from_name("docx"), None);
    }

    /// A segment from a specific stream (mic / system / file), for the Me-Them meeting-label tests.
    fn seg_src(
        text: &str,
        start_ms: u64,
        end_ms: u64,
        source: AudioSourceKind,
    ) -> TranscriptSegment {
        TranscriptSegment::new(
            0,
            text,
            Duration::from_millis(start_ms)..Duration::from_millis(end_ms),
            source,
        )
    }

    #[test]
    fn markdown_has_front_matter_summary_and_a_transcript() {
        let segs = vec![seg("hello there", 0, 2_000)];
        let meta = MeetingMeta {
            title: Some("Standup".to_owned()),
            date: Some("2026-06-04".to_owned()),
            engine: Some("Apple on-device speech".to_owned()),
            language: Some("en".to_owned()),
            summary: Some("Quick sync.".to_owned()),
        };
        let md = format_markdown(&segs, &meta);

        assert!(md.starts_with("---\n"), "opens with YAML front-matter");
        assert!(md.contains("title: \"Standup\""));
        assert!(md.contains("date: \"2026-06-04\""));
        assert!(md.contains("duration: 00:00:02"));
        assert!(md.contains("engine: \"Apple on-device speech\""));
        assert!(md.contains("# Standup"));
        assert!(md.contains("## Summary\n\nQuick sync.\n\n"));
        assert!(md.contains("## Transcript\n\n**[0:00]** hello there\n\n"));
    }

    #[test]
    fn markdown_labels_me_and_them_by_source() {
        let segs = vec![
            seg_src("morning", 0, 1_000, AudioSourceKind::Microphone),
            seg_src("hi back", 2_000, 3_000, AudioSourceKind::System),
        ];
        let md = format_markdown(&segs, &MeetingMeta::default());

        assert!(md.contains("participants: [\"Me\", \"Them\"]"));
        assert!(md.contains("**[0:00] Me:** morning"));
        assert!(md.contains("**[0:02] Them:** hi back"));
    }

    #[test]
    fn markdown_tags_the_diarized_far_end_speaker() {
        let mut s = seg_src("over here", 0, 1_000, AudioSourceKind::System);
        s.speaker = Some(SpeakerId(1));
        let md = format_markdown(&[s], &MeetingMeta::default());
        // System + a diarized speaker → "Them (Speaker 2)" (SpeakerId is 0-based, label 1-based).
        assert!(
            md.contains("**[0:00] Them (Speaker 2):** over here"),
            "{md}"
        );
    }

    #[test]
    fn a_meeting_keeps_me_and_them_in_separate_blocks() {
        // Same speaker (None) but different streams must not merge into one paragraph.
        let segs = vec![
            seg_src("a", 0, 500, AudioSourceKind::Microphone),
            seg_src("b", 600, 1_000, AudioSourceKind::System),
        ];
        assert_eq!(group_paragraphs(&segs).len(), 2);
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
