//! Turns complete utterances into transcript segments via an ASR engine.
//!
//! Owns the (slow) [`AsrEngine`] so it can run on its own thread, draining utterances produced by
//! the [`Segmenter`](crate::segmenter::Segmenter) without ever stalling the capture path.

use std::time::Duration;

use wisp_audio::{normalize_for_asr, TARGET_SAMPLE_RATE};
use wisp_core::diarize::Diarizer;
use wisp_core::engine::AsrEngine;
use wisp_core::error::Result;
use wisp_core::transcript::{AudioSourceKind, SegmentStatus, TranscriptSegment};

use crate::segmenter::Utterance;

/// Transcribes utterances into transcript segments, tagging each with the source kind.
///
/// Two entry points: the File pipeline uses [`transcribe`](Self::transcribe) (one segment per
/// engine sentence, sequential ids — granular for SRT/VTT export); the live session uses
/// [`transcribe_utterance`](Self::transcribe_utterance) (one merged segment with a caller-supplied
/// id and status, so a provisional `Partial` and its `Final` share an id and update in place).
pub struct Transcriber {
    engine: Box<dyn AsrEngine>,
    source_kind: AudioSourceKind,
    next_id: u64,
    diarizer: Option<Box<dyn Diarizer>>,
    /// Recent finalized text, fed back to the engine as a biasing prompt so names/terms stay
    /// consistent across a live session (see [`transcribe_utterance`](Self::transcribe_utterance)).
    context: String,
    /// Whether to maintain rolling context at all (toggled by [`ROLLING_CONTEXT_ENV`]).
    context_enabled: bool,
}

impl Transcriber {
    /// Env var toggling cross-utterance rolling context — `off`/`0`/`false`/`no` disables it.
    /// Defaults on: feeding recent finals back as a prompt keeps proper nouns and terminology
    /// consistent. Disable it if a session's domain confuses the bias.
    pub const ROLLING_CONTEXT_ENV: &'static str = "WISP_LIVE_CONTEXT";

    /// Creates a transcriber that tags segments as coming from `source_kind`.
    pub fn new(engine: Box<dyn AsrEngine>, source_kind: AudioSourceKind) -> Self {
        Self {
            engine,
            source_kind,
            next_id: 0,
            diarizer: None,
            context: String::new(),
            context_enabled: rolling_context_enabled(),
        }
    }

    /// Attaches a live diarizer that labels each utterance's segments with their speaker.
    pub fn with_diarizer(mut self, diarizer: Box<dyn Diarizer>) -> Self {
        self.diarizer = Some(diarizer);
        self
    }

    /// Overrides the [`ROLLING_CONTEXT_ENV`](Self::ROLLING_CONTEXT_ENV)-derived default, mainly for
    /// tests that need a deterministic setting regardless of the environment.
    pub fn with_rolling_context(mut self, enabled: bool) -> Self {
        self.context_enabled = enabled;
        self
    }

    /// The kind this transcriber tags File segments with, and the default a single-stream live session
    /// ([`spawn_live`](crate::session::Session::spawn_live)) labels its audio. A multi-stream session
    /// overrides it per utterance via [`transcribe_utterance`](Self::transcribe_utterance).
    pub fn source_kind(&self) -> AudioSourceKind {
        self.source_kind
    }

    /// Transcribes one complete `utterance` into final segments (one per engine sentence).
    ///
    /// Empty / punctuation-only recognitions are dropped (ASR hallucinations on near-silence).
    /// Surviving segments get a sequential id, the configured source kind, `Final` status, and
    /// their times offset by the utterance's start. The engine is reset afterwards. Used by the
    /// File pipeline, where per-sentence granularity feeds SRT/VTT export.
    pub fn transcribe(&mut self, utterance: &Utterance) -> Result<Vec<TranscriptSegment>> {
        let audio = normalize_for_asr(&utterance.audio, TARGET_SAMPLE_RATE);
        let result = self.engine.transcribe(&audio, TARGET_SAMPLE_RATE)?;

        let mut segments = Vec::new();
        for mut segment in result.segments {
            if !segment.text.chars().any(char::is_alphanumeric) {
                continue;
            }
            segment.id = self.next_id;
            segment.source = self.source_kind;
            segment.status = SegmentStatus::Final;
            segment.start += utterance.start;
            segment.end += utterance.start;
            self.next_id += 1;
            segments.push(segment);
        }

        // Live diarization: label every segment of this utterance with its speaker. Best-effort —
        // a failure leaves the speaker unset rather than dropping the segment.
        if !segments.is_empty() {
            if let Some(diarizer) = &mut self.diarizer {
                if let Ok(speaker) = diarizer.identify(&utterance.audio, TARGET_SAMPLE_RATE) {
                    for segment in &mut segments {
                        segment.speaker = Some(speaker);
                    }
                }
            }
        }

        self.engine.reset();
        Ok(segments)
    }

    /// Transcribes `utterance` into a single segment carrying the caller's `id` and `status`.
    ///
    /// All of the engine's sub-segments are merged into one line (text concatenated, span = first
    /// start .. last end) so a multi-sentence utterance stays one updatable row that a `Partial`
    /// can refine and a `Final` can replace in place. Empty / punctuation-only recognitions return
    /// `None` (ASR hallucinations on near-silence). Diarization runs **only for `Final`** — a
    /// partial stays unlabelled, both for speed and because the speaker can still change before the
    /// utterance closes. The engine is reset afterwards so the next decode starts clean.
    pub fn transcribe_utterance(
        &mut self,
        id: u64,
        kind: AudioSourceKind,
        utterance: &Utterance,
        status: SegmentStatus,
    ) -> Result<Option<TranscriptSegment>> {
        let audio = normalize_for_asr(&utterance.audio, TARGET_SAMPLE_RATE);
        let result = self.engine.transcribe(&audio, TARGET_SAMPLE_RATE)?;
        self.engine.reset();

        let Some(mut segment) = merge_segments(result.segments) else {
            return Ok(None);
        };

        segment.id = id;
        segment.source = kind;
        segment.status = status;
        segment.start += utterance.start;
        segment.end += utterance.start;

        if status == SegmentStatus::Final {
            if let Some(diarizer) = &mut self.diarizer {
                if let Ok(speaker) = diarizer.identify(&utterance.audio, TARGET_SAMPLE_RATE) {
                    segment.speaker = Some(speaker);
                }
            }

            // Bias the next utterance (its partials and final) toward this line's names and terms.
            if self.context_enabled {
                self.push_context(&segment.text);
                self.engine.set_context(&self.context);
            }
        }

        Ok(Some(segment))
    }

    /// Appends a finalized line to the rolling context, trimmed to the last [`CONTEXT_CHARS`] chars
    /// on a char boundary so the bias stays a bounded prompt and can't snowball.
    fn push_context(&mut self, text: &str) {
        if !self.context.is_empty() {
            self.context.push(' ');
        }
        self.context.push_str(text);

        let overflow = self.context.chars().count().saturating_sub(CONTEXT_CHARS);
        if overflow > 0 {
            self.context = self.context.chars().skip(overflow).collect();
        }
    }
}

/// Most chars of recent finalized text to keep as rolling context. Bounded so it stays a light,
/// prompt-sized hint (whisper caps an initial prompt around here) that biases without snowballing.
const CONTEXT_CHARS: usize = 224;

/// Whether rolling context is on, per [`Transcriber::ROLLING_CONTEXT_ENV`]; defaults on.
fn rolling_context_enabled() -> bool {
    !matches!(
        std::env::var(Transcriber::ROLLING_CONTEXT_ENV)
            .ok()
            .as_deref(),
        Some("off" | "0" | "false" | "no")
    )
}

/// Collapses an engine's (possibly multi-sentence) segments into one updatable line, or `None` if
/// there's no alphanumeric content. Text and word timings are concatenated in order; the span
/// covers the earliest start to the latest end. The first segment carries over any other fields.
fn merge_segments(segments: Vec<TranscriptSegment>) -> Option<TranscriptSegment> {
    let mut text = String::new();
    let mut words = Vec::new();
    let mut span: Option<(Duration, Duration)> = None;
    for s in &segments {
        text.push_str(&s.text);
        words.extend(s.words.iter().cloned());
        span = Some(match span {
            None => (s.start, s.end),
            Some((start, end)) => (start.min(s.start), end.max(s.end)),
        });
    }

    if !text.chars().any(char::is_alphanumeric) {
        return None;
    }

    let (start, end) = span.unwrap_or((Duration::ZERO, Duration::ZERO));
    let mut merged = segments.into_iter().next()?;
    merged.text = text;
    merged.words = words;
    merged.start = start;
    merged.end = end;
    Some(merged)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};
    use wisp_core::engine::{EngineInfo, TranscriptionResult};
    use wisp_core::testing::MockAsrEngine;

    /// An engine that returns canned text per `transcribe` call and records every `set_context`
    /// string, so a test can assert what rolling context the transcriber fed back.
    struct RecordingEngine {
        texts: Vec<String>,
        idx: usize,
        contexts: Arc<Mutex<Vec<String>>>,
    }

    impl AsrEngine for RecordingEngine {
        fn info(&self) -> EngineInfo {
            EngineInfo {
                name: "recording".to_owned(),
                streaming: false,
            }
        }
        fn transcribe(&mut self, _audio: &[f32], _sample_rate: u32) -> Result<TranscriptionResult> {
            let text = self.texts.get(self.idx).cloned().unwrap_or_default();
            self.idx += 1;
            Ok(TranscriptionResult {
                segments: vec![TranscriptSegment::new(
                    0,
                    text,
                    Duration::ZERO..Duration::from_millis(10),
                    AudioSourceKind::File,
                )],
            })
        }
        fn set_context(&mut self, recent_text: &str) {
            self.contexts.lock().unwrap().push(recent_text.to_owned());
        }
    }

    fn recording(texts: &[&str]) -> (Box<RecordingEngine>, Arc<Mutex<Vec<String>>>) {
        let contexts = Arc::new(Mutex::new(Vec::new()));
        let engine = Box::new(RecordingEngine {
            texts: texts.iter().map(|s| (*s).to_owned()).collect(),
            idx: 0,
            contexts: Arc::clone(&contexts),
        });
        (engine, contexts)
    }

    fn utterance(start_ms: u64) -> Utterance {
        Utterance {
            audio: vec![0.1; 1_600],
            start: Duration::from_millis(start_ms),
        }
    }

    fn canned(text: &str) -> TranscriptionResult {
        TranscriptionResult {
            segments: vec![TranscriptSegment::new(
                999,
                text,
                Duration::ZERO..Duration::from_millis(50),
                AudioSourceKind::File,
            )],
        }
    }

    #[test]
    fn overrides_id_source_status_and_offsets_times() {
        let engine = Box::new(MockAsrEngine::new(vec![canned("hello")]));
        let mut transcriber = Transcriber::new(engine, AudioSourceKind::Microphone);

        let segment = transcriber
            .transcribe_utterance(
                7,
                AudioSourceKind::Microphone,
                &utterance(500),
                SegmentStatus::Final,
            )
            .unwrap()
            .expect("a non-empty recognition yields a segment");

        assert_eq!(segment.id, 7);
        assert_eq!(segment.text, "hello");
        assert_eq!(segment.source, AudioSourceKind::Microphone);
        assert_eq!(segment.status, SegmentStatus::Final);
        assert_eq!(segment.start, Duration::from_millis(500));
        assert_eq!(segment.end, Duration::from_millis(550));
    }

    #[test]
    fn shares_one_id_across_a_partial_then_its_final() {
        // The capture thread reuses an id so a provisional partial and its final update one row.
        let engine = Box::new(MockAsrEngine::new(vec![
            canned("hi the"),
            canned("hi there"),
        ]));
        let mut transcriber = Transcriber::new(engine, AudioSourceKind::Microphone);

        let partial = transcriber
            .transcribe_utterance(
                3,
                AudioSourceKind::Microphone,
                &utterance(0),
                SegmentStatus::Partial,
            )
            .unwrap()
            .unwrap();
        let committed = transcriber
            .transcribe_utterance(
                3,
                AudioSourceKind::Microphone,
                &utterance(0),
                SegmentStatus::Final,
            )
            .unwrap()
            .unwrap();

        assert_eq!((partial.id, partial.status), (3, SegmentStatus::Partial));
        assert_eq!((committed.id, committed.status), (3, SegmentStatus::Final));
        assert_eq!(committed.text, "hi there");
    }

    #[test]
    fn drops_punctuation_only_recognition() {
        let engine = Box::new(MockAsrEngine::new(vec![canned("？")]));
        let mut transcriber = Transcriber::new(engine, AudioSourceKind::Microphone);

        let segment = transcriber
            .transcribe_utterance(
                0,
                AudioSourceKind::Microphone,
                &utterance(0),
                SegmentStatus::Final,
            )
            .unwrap();
        assert!(segment.is_none());
    }

    #[test]
    fn merges_a_multi_sentence_utterance_into_one_line() {
        let engine = Box::new(MockAsrEngine::new(vec![TranscriptionResult {
            segments: vec![
                TranscriptSegment::new(
                    1,
                    "Hello world.",
                    Duration::ZERO..Duration::from_millis(500),
                    AudioSourceKind::File,
                ),
                TranscriptSegment::new(
                    2,
                    " How are you?",
                    Duration::from_millis(500)..Duration::from_millis(1_200),
                    AudioSourceKind::File,
                ),
            ],
        }]));
        let mut transcriber = Transcriber::new(engine, AudioSourceKind::Microphone);

        let segment = transcriber
            .transcribe_utterance(
                0,
                AudioSourceKind::Microphone,
                &utterance(100),
                SegmentStatus::Final,
            )
            .unwrap()
            .unwrap();

        assert_eq!(segment.text, "Hello world. How are you?");
        // Span = earliest start .. latest end, each offset by the utterance start (100 ms).
        assert_eq!(segment.start, Duration::from_millis(100));
        assert_eq!(segment.end, Duration::from_millis(1_300));
    }

    #[test]
    fn final_labels_the_segment_with_its_speaker() {
        use wisp_core::diarize::Diarizer;
        use wisp_core::transcript::SpeakerId;

        struct FixedSpeaker(u32);
        impl Diarizer for FixedSpeaker {
            fn identify(&mut self, _audio: &[f32], _sample_rate: u32) -> Result<SpeakerId> {
                Ok(SpeakerId(self.0))
            }
        }

        let engine = Box::new(MockAsrEngine::new(vec![canned("hi")]));
        let mut transcriber = Transcriber::new(engine, AudioSourceKind::Microphone)
            .with_diarizer(Box::new(FixedSpeaker(3)));

        let segment = transcriber
            .transcribe_utterance(
                0,
                AudioSourceKind::Microphone,
                &utterance(0),
                SegmentStatus::Final,
            )
            .unwrap()
            .unwrap();
        assert_eq!(segment.speaker, Some(SpeakerId(3)));
    }

    #[test]
    fn partial_skips_diarization() {
        use wisp_core::diarize::Diarizer;
        use wisp_core::transcript::SpeakerId;

        struct FixedSpeaker;
        impl Diarizer for FixedSpeaker {
            fn identify(&mut self, _audio: &[f32], _sample_rate: u32) -> Result<SpeakerId> {
                Ok(SpeakerId(9))
            }
        }

        let engine = Box::new(MockAsrEngine::new(vec![canned("hi")]));
        let mut transcriber = Transcriber::new(engine, AudioSourceKind::Microphone)
            .with_diarizer(Box::new(FixedSpeaker));

        let segment = transcriber
            .transcribe_utterance(
                0,
                AudioSourceKind::Microphone,
                &utterance(0),
                SegmentStatus::Partial,
            )
            .unwrap()
            .unwrap();
        assert!(
            segment.speaker.is_none(),
            "a partial stays unlabelled — the speaker can still change before it closes"
        );
    }

    #[test]
    fn transcribe_assigns_sequential_ids_and_offsets_times() {
        let engine = Box::new(MockAsrEngine::new(vec![canned("one"), canned("two")]));
        let mut transcriber = Transcriber::new(engine, AudioSourceKind::System);

        let first = transcriber.transcribe(&utterance(0)).unwrap();
        let second = transcriber.transcribe(&utterance(100)).unwrap();

        assert_eq!((first[0].id, first[0].status), (0, SegmentStatus::Final));
        assert_eq!(first[0].source, AudioSourceKind::System);
        assert_eq!(second[0].id, 1);
        assert_eq!(second[0].start, Duration::from_millis(100));
    }

    #[test]
    fn transcribe_drops_punctuation_only_recognition() {
        let engine = Box::new(MockAsrEngine::new(vec![canned("？")]));
        let mut transcriber = Transcriber::new(engine, AudioSourceKind::Microphone);

        assert!(transcriber.transcribe(&utterance(0)).unwrap().is_empty());
    }

    #[test]
    fn rolling_context_accumulates_finals_and_skips_partials() {
        let (engine, contexts) = recording(&["hello Acme", "ignored partial", "second line"]);
        let mut t =
            Transcriber::new(engine, AudioSourceKind::Microphone).with_rolling_context(true);

        t.transcribe_utterance(
            0,
            AudioSourceKind::Microphone,
            &utterance(0),
            SegmentStatus::Final,
        )
        .unwrap();
        t.transcribe_utterance(
            1,
            AudioSourceKind::Microphone,
            &utterance(0),
            SegmentStatus::Partial,
        )
        .unwrap();
        t.transcribe_utterance(
            2,
            AudioSourceKind::Microphone,
            &utterance(0),
            SegmentStatus::Final,
        )
        .unwrap();

        // set_context fires only on finals, accumulating recent text; the partial never contributes.
        assert_eq!(
            *contexts.lock().unwrap(),
            vec!["hello Acme".to_owned(), "hello Acme second line".to_owned()]
        );
    }

    #[test]
    fn rolling_context_is_capped_to_a_bounded_window() {
        let long = "x".repeat(CONTEXT_CHARS + 50);
        let (engine, contexts) = recording(&[&long]);
        let mut t =
            Transcriber::new(engine, AudioSourceKind::Microphone).with_rolling_context(true);

        t.transcribe_utterance(
            0,
            AudioSourceKind::Microphone,
            &utterance(0),
            SegmentStatus::Final,
        )
        .unwrap();

        let seen = contexts.lock().unwrap();
        assert_eq!(seen.len(), 1);
        assert_eq!(
            seen[0].chars().count(),
            CONTEXT_CHARS,
            "context is trimmed to the cap so it can't snowball"
        );
    }

    #[test]
    fn rolling_context_disabled_never_reaches_the_engine() {
        let (engine, contexts) = recording(&["hello", "world"]);
        let mut t =
            Transcriber::new(engine, AudioSourceKind::Microphone).with_rolling_context(false);

        t.transcribe_utterance(
            0,
            AudioSourceKind::Microphone,
            &utterance(0),
            SegmentStatus::Final,
        )
        .unwrap();
        t.transcribe_utterance(
            1,
            AudioSourceKind::Microphone,
            &utterance(0),
            SegmentStatus::Final,
        )
        .unwrap();

        assert!(contexts.lock().unwrap().is_empty());
    }

    /// Renaming this constant silently breaks anyone who pinned the rolling-context toggle via env.
    #[test]
    fn rolling_context_env_var_name_is_pinned() {
        assert_eq!(Transcriber::ROLLING_CONTEXT_ENV, "WISP_LIVE_CONTEXT");
    }

    /// The env var is process-global, so exercise its whole lifecycle in one serial test.
    #[test]
    fn rolling_context_enabled_defaults_on_and_parses_off_values() {
        let env = Transcriber::ROLLING_CONTEXT_ENV;

        std::env::remove_var(env);
        assert!(rolling_context_enabled(), "unset → on");

        for off in ["off", "0", "false", "no"] {
            std::env::set_var(env, off);
            assert!(!rolling_context_enabled(), "{off} disables");
        }

        std::env::set_var(env, "on");
        assert!(rolling_context_enabled(), "any other value → on");

        std::env::remove_var(env);
    }
}
