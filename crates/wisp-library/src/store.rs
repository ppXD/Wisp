//! The SQLite-backed store: persistence, retrieval, and full-text search over past meetings.

use std::collections::HashSet;
use std::path::Path;

use rusqlite::{Connection, OptionalExtension};
use wisp_core::export::MeetingMeta;
use wisp_core::transcript::{AudioSourceKind, SegmentStatus, TranscriptSegment};

use crate::record::{Meeting, MeetingSummary, SearchHit, Segment};
use crate::Result;

/// On-disk schema version, bumped on schema changes (drives migration via `PRAGMA user_version`).
const SCHEMA_VERSION: i64 = 1;

/// Characters of transcript kept as a list preview.
const PREVIEW_CHARS: usize = 160;

/// Characters of the matching segment shown as a snippet for a LIKE-fallback hit. FTS hits use
/// SQLite's own `snippet()` instead.
const LIKE_SNIPPET_CHARS: usize = 120;

/// Initial schema. `segment_fts` is an external-content FTS5 index (it stores no copy of the text,
/// keeping the database small); the triggers keep it in sync with `segment`. The `trigram` tokenizer
/// gives substring matching that also works for CJK, where word boundaries aren't whitespace.
const SCHEMA_V1: &str = "\
CREATE TABLE meeting (
    id            TEXT PRIMARY KEY,
    title         TEXT NOT NULL,
    started_at_ms INTEGER NOT NULL,
    duration_ms   INTEGER NOT NULL,
    language      TEXT,
    engine        TEXT,
    summary       TEXT,
    segment_count INTEGER NOT NULL
);
CREATE INDEX meeting_started_at ON meeting (started_at_ms DESC);

CREATE TABLE segment (
    id         INTEGER PRIMARY KEY,
    meeting_id TEXT NOT NULL REFERENCES meeting (id) ON DELETE CASCADE,
    idx        INTEGER NOT NULL,
    start_ms   INTEGER NOT NULL,
    end_ms     INTEGER NOT NULL,
    speaker    INTEGER,
    source     TEXT NOT NULL,
    text       TEXT NOT NULL
);
CREATE INDEX segment_meeting ON segment (meeting_id, idx);

CREATE VIRTUAL TABLE segment_fts USING fts5 (
    text,
    content = 'segment',
    content_rowid = 'id',
    tokenize = 'trigram'
);
CREATE TRIGGER segment_ai AFTER INSERT ON segment BEGIN
    INSERT INTO segment_fts (rowid, text) VALUES (new.id, new.text);
END;
CREATE TRIGGER segment_ad AFTER DELETE ON segment BEGIN
    INSERT INTO segment_fts (segment_fts, rowid, text) VALUES ('delete', old.id, old.text);
END;
";

/// A handle to the meeting knowledge base. Open once and reuse across queries.
pub struct Library {
    conn: Connection,
}

impl Library {
    /// Opens (creating if absent) the library database at `path`, applying schema migrations.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        Self::init(Connection::open(path)?)
    }

    /// Opens a fresh in-memory library — for tests.
    pub fn open_in_memory() -> Result<Self> {
        Self::init(Connection::open_in_memory()?)
    }

    fn init(conn: Connection) -> Result<Self> {
        conn.execute_batch("PRAGMA foreign_keys = ON;")?;
        let mut lib = Self { conn };
        lib.migrate()?;
        Ok(lib)
    }

    fn migrate(&mut self) -> Result<()> {
        let version: i64 = self
            .conn
            .pragma_query_value(None, "user_version", |r| r.get(0))?;
        if version < 1 {
            self.conn.execute_batch(SCHEMA_V1)?;
        }
        self.conn
            .pragma_update(None, "user_version", SCHEMA_VERSION)?;
        Ok(())
    }

    /// Persists a finished meeting — its metadata and finalized transcript segments — and indexes it
    /// for full-text search, in one transaction. `id` is caller-generated (the crate stays clock- and
    /// randomness-free) and `started_at_ms` is the session start in epoch milliseconds. Saving an
    /// existing `id` replaces it. Partial and blank segments are skipped; the duration is the latest
    /// segment end.
    pub fn save_meeting(
        &mut self,
        id: &str,
        meta: &MeetingMeta,
        started_at_ms: i64,
        segments: &[TranscriptSegment],
    ) -> Result<()> {
        let finals: Vec<&TranscriptSegment> = segments
            .iter()
            .filter(|s| s.status == SegmentStatus::Final && !s.text.trim().is_empty())
            .collect();

        let duration_ms = finals
            .iter()
            .map(|s| s.end.as_millis() as i64)
            .max()
            .unwrap_or(0);
        let title = meta
            .title
            .clone()
            .unwrap_or_else(|| "Untitled meeting".to_owned());

        let tx = self.conn.transaction()?;
        tx.execute("DELETE FROM meeting WHERE id = ?1", [id])?;
        tx.execute(
            "INSERT INTO meeting
                 (id, title, started_at_ms, duration_ms, language, engine, summary, segment_count)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            rusqlite::params![
                id,
                title,
                started_at_ms,
                duration_ms,
                meta.language,
                meta.engine,
                meta.summary,
                finals.len() as i64,
            ],
        )?;
        insert_segments(&tx, id, &finals)?;
        tx.commit()?;
        Ok(())
    }

    /// Every meeting, newest first, each with a short transcript preview — for the Library list.
    pub fn list_meetings(&self) -> Result<Vec<MeetingSummary>> {
        let mut stmt = self.conn.prepare(
            "SELECT m.id, m.title, m.started_at_ms, m.duration_ms, m.language, m.engine,
                    (SELECT group_concat(text, ' ')
                       FROM (SELECT text FROM segment WHERE meeting_id = m.id ORDER BY idx LIMIT 6))
             FROM meeting m
             ORDER BY m.started_at_ms DESC",
        )?;
        let rows = stmt
            .query_map([], |r| {
                let preview: Option<String> = r.get(6)?;
                Ok(MeetingSummary {
                    id: r.get(0)?,
                    title: r.get(1)?,
                    started_at_ms: r.get(2)?,
                    duration_ms: r.get(3)?,
                    language: r.get(4)?,
                    engine: r.get(5)?,
                    preview: truncate_chars(&preview.unwrap_or_default(), PREVIEW_CHARS),
                })
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        Ok(rows)
    }

    /// One meeting's metadata plus its segments in order, or `None` if no such meeting.
    pub fn get_meeting(&self, id: &str) -> Result<Option<(Meeting, Vec<Segment>)>> {
        let meeting = self
            .conn
            .query_row(
                "SELECT id, title, started_at_ms, duration_ms, language, engine, summary, segment_count
                 FROM meeting WHERE id = ?1",
                [id],
                |r| {
                    Ok(Meeting {
                        id: r.get(0)?,
                        title: r.get(1)?,
                        started_at_ms: r.get(2)?,
                        duration_ms: r.get(3)?,
                        language: r.get(4)?,
                        engine: r.get(5)?,
                        summary: r.get(6)?,
                        segment_count: r.get(7)?,
                    })
                },
            )
            .optional()?;
        let Some(meeting) = meeting else {
            return Ok(None);
        };

        let mut stmt = self.conn.prepare(
            "SELECT idx, start_ms, end_ms, speaker, source, text
             FROM segment WHERE meeting_id = ?1 ORDER BY idx",
        )?;
        let segments = stmt
            .query_map([id], |r| {
                Ok(Segment {
                    idx: r.get(0)?,
                    start_ms: r.get(1)?,
                    end_ms: r.get(2)?,
                    speaker: r.get(3)?,
                    source: r.get(4)?,
                    text: r.get(5)?,
                })
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        Ok(Some((meeting, segments)))
    }

    /// Searches transcripts for `query`, returning one [`SearchHit`] per matching meeting (its
    /// best-scoring snippet), most relevant first, capped at `limit`. A blank query yields no hits.
    ///
    /// Tokens of 3+ characters use the ranked FTS5 trigram index; shorter terms — common in CJK,
    /// where a word may be one or two characters — fall back to a LIKE substring scan so they still
    /// match (semantic search, added later, covers this case more thoroughly).
    pub fn search(&self, query: &str, limit: usize) -> Result<Vec<SearchHit>> {
        let tokens: Vec<&str> = query.split_whitespace().collect();

        if tokens.is_empty() {
            return Ok(Vec::new());
        }

        if tokens.iter().all(|t| t.chars().count() >= 3) {
            self.search_fts(&tokens, limit)
        } else {
            self.search_like(&tokens, limit)
        }
    }

    /// Ranked full-text search over the FTS5 trigram index (BM25 score + SQLite snippets).
    fn search_fts(&self, tokens: &[&str], limit: usize) -> Result<Vec<SearchHit>> {
        // Quote each token so arbitrary punctuation can't break FTS5 query syntax; AND-combined.
        let match_query = tokens
            .iter()
            .map(|t| format!("\"{}\"", t.replace('"', "\"\"")))
            .collect::<Vec<_>>()
            .join(" ");

        let mut stmt = self.conn.prepare(
            "SELECT s.meeting_id, m.title, m.started_at_ms,
                    snippet(segment_fts, 0, '«', '»', '…', 12),
                    bm25(segment_fts)
             FROM segment_fts
             JOIN segment s ON s.id = segment_fts.rowid
             JOIN meeting m ON m.id = s.meeting_id
             WHERE segment_fts MATCH ?1
             ORDER BY bm25(segment_fts)",
        )?;
        let hits = stmt
            .query_map(rusqlite::params![match_query], |r| {
                Ok(SearchHit {
                    meeting_id: r.get(0)?,
                    title: r.get(1)?,
                    started_at_ms: r.get(2)?,
                    snippet: r.get(3)?,
                    score: r.get(4)?,
                })
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        Ok(best_per_meeting(hits, limit))
    }

    /// Unranked substring fallback for tokens the trigram index can't handle (under 3 chars): every
    /// token must appear, case-insensitively, in a segment. Slower than FTS, but a personal library
    /// is small and this path is only hit by short queries.
    fn search_like(&self, tokens: &[&str], limit: usize) -> Result<Vec<SearchHit>> {
        let predicate = (1..=tokens.len())
            .map(|n| format!("s.text LIKE ?{n} ESCAPE '\\'"))
            .collect::<Vec<_>>()
            .join(" AND ");
        let sql = format!(
            "SELECT s.meeting_id, m.title, m.started_at_ms, s.text
             FROM segment s
             JOIN meeting m ON m.id = s.meeting_id
             WHERE {predicate}
             ORDER BY m.started_at_ms DESC, s.idx",
        );
        let patterns: Vec<String> = tokens
            .iter()
            .map(|t| format!("%{}%", like_escape(t)))
            .collect();

        let mut stmt = self.conn.prepare(&sql)?;
        let hits = stmt
            .query_map(rusqlite::params_from_iter(&patterns), |r| {
                let text: String = r.get(3)?;
                Ok(SearchHit {
                    meeting_id: r.get(0)?,
                    title: r.get(1)?,
                    started_at_ms: r.get(2)?,
                    snippet: truncate_chars(&text, LIKE_SNIPPET_CHARS),
                    score: 0.0,
                })
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        Ok(best_per_meeting(hits, limit))
    }

    /// Deletes a meeting and its segments + search index. Returns whether a meeting existed.
    pub fn delete_meeting(&self, id: &str) -> Result<bool> {
        let affected = self
            .conn
            .execute("DELETE FROM meeting WHERE id = ?1", [id])?;
        Ok(affected > 0)
    }

    /// Number of meetings stored.
    pub fn count(&self) -> Result<i64> {
        Ok(self
            .conn
            .query_row("SELECT count(*) FROM meeting", [], |r| r.get(0))?)
    }
}

/// Maps an [`AudioSourceKind`] to the stored source label.
fn source_label(source: AudioSourceKind) -> &'static str {
    match source {
        AudioSourceKind::Microphone => "mic",
        AudioSourceKind::System => "system",
        AudioSourceKind::File => "file",
        _ => "unknown",
    }
}

/// Inserts a meeting's finalized segments into the open transaction (the FTS index follows via the
/// insert trigger). Factored out of [`Library::save_meeting`] so that method reads as a flat pipeline.
fn insert_segments(
    tx: &rusqlite::Transaction,
    meeting_id: &str,
    finals: &[&TranscriptSegment],
) -> rusqlite::Result<()> {
    let mut stmt = tx.prepare(
        "INSERT INTO segment (meeting_id, idx, start_ms, end_ms, speaker, source, text)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
    )?;
    for (idx, seg) in finals.iter().enumerate() {
        stmt.execute(rusqlite::params![
            meeting_id,
            idx as i64,
            seg.start.as_millis() as i64,
            seg.end.as_millis() as i64,
            seg.speaker.map(|s| i64::from(s.0)),
            source_label(seg.source),
            seg.text.trim(),
        ])?;
    }
    Ok(())
}

/// Escapes LIKE wildcards (`%`, `_`, `\`) so a user's literal characters aren't treated as patterns
/// (paired with `ESCAPE '\'` in the query).
fn like_escape(token: &str) -> String {
    token
        .replace('\\', "\\\\")
        .replace('%', "\\%")
        .replace('_', "\\_")
}

/// Collapses score-ordered per-segment hits to the best hit per meeting, capped at `limit`. The
/// first row seen for a meeting is its best, since the input is already ordered by score.
fn best_per_meeting(hits: Vec<SearchHit>, limit: usize) -> Vec<SearchHit> {
    let mut seen = HashSet::new();
    let mut out = Vec::new();
    for hit in hits {
        if seen.insert(hit.meeting_id.clone()) {
            out.push(hit);
            if out.len() >= limit {
                break;
            }
        }
    }
    out
}

/// Truncates `s` to at most `max` characters (not bytes), appending `…` when shortened.
fn truncate_chars(s: &str, max: usize) -> String {
    let trimmed = s.trim();
    if trimmed.chars().count() <= max {
        return trimmed.to_owned();
    }
    let mut out: String = trimmed.chars().take(max).collect();
    out.push('…');
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;
    use wisp_core::transcript::SpeakerId;

    fn seg(
        id: u64,
        start_ms: u64,
        end_ms: u64,
        text: &str,
        source: AudioSourceKind,
    ) -> TranscriptSegment {
        TranscriptSegment {
            id,
            text: text.to_owned(),
            start: Duration::from_millis(start_ms),
            end: Duration::from_millis(end_ms),
            status: SegmentStatus::Final,
            source,
            speaker: None,
            confidence: None,
            words: Vec::new(),
            aux_text: None,
        }
    }

    fn meta(title: &str) -> MeetingMeta {
        MeetingMeta {
            title: Some(title.to_owned()),
            date: None,
            engine: Some("whisper".to_owned()),
            language: Some("en".to_owned()),
            summary: None,
        }
    }

    #[test]
    fn save_with_no_final_segments_stores_an_empty_meeting() {
        let mut lib = Library::open_in_memory().unwrap();
        lib.save_meeting("m1", &meta("Silent"), 1000, &[]).unwrap();
        let (m, rows) = lib.get_meeting("m1").unwrap().unwrap();
        assert_eq!(m.segment_count, 0);
        assert_eq!(m.duration_ms, 0);
        assert!(rows.is_empty());
    }

    #[test]
    fn get_missing_meeting_returns_none() {
        let lib = Library::open_in_memory().unwrap();
        assert!(lib.get_meeting("nope").unwrap().is_none());
    }

    #[test]
    fn list_is_empty_with_no_meetings() {
        let lib = Library::open_in_memory().unwrap();
        assert!(lib.list_meetings().unwrap().is_empty());
    }

    #[test]
    fn source_label_covers_every_kind() {
        let mut lib = Library::open_in_memory().unwrap();
        lib.save_meeting(
            "m1",
            &meta("M"),
            0,
            &[
                seg(1, 0, 100, "mic line", AudioSourceKind::Microphone),
                seg(2, 100, 200, "system line", AudioSourceKind::System),
                seg(3, 200, 300, "file line", AudioSourceKind::File),
            ],
        )
        .unwrap();
        let (_, rows) = lib.get_meeting("m1").unwrap().unwrap();
        let sources: Vec<&str> = rows.iter().map(|s| s.source.as_str()).collect();
        assert_eq!(sources, ["mic", "system", "file"]);
    }

    #[test]
    fn search_respects_the_limit() {
        let mut lib = Library::open_in_memory().unwrap();
        for i in 0i64..5 {
            lib.save_meeting(
                &format!("m{i}"),
                &meta("M"),
                1000 + i,
                &[seg(
                    1,
                    0,
                    100,
                    "shared keyword here",
                    AudioSourceKind::Microphone,
                )],
            )
            .unwrap();
        }
        // Five meetings match; the cap holds.
        assert_eq!(lib.search("keyword", 3).unwrap().len(), 3);
    }

    #[test]
    fn search_tolerates_punctuation_without_erroring() {
        let mut lib = Library::open_in_memory().unwrap();
        lib.save_meeting(
            "m1",
            &meta("M"),
            0,
            &[seg(
                1,
                0,
                100,
                "the quarterly budget",
                AudioSourceKind::Microphone,
            )],
        )
        .unwrap();
        // Raw FTS5 special characters in user input must not produce a query-syntax error.
        for q in ["budget (Q3)", "a\" OR b", "foo: bar*", "((("] {
            assert!(lib.search(q, 10).is_ok(), "query {q:?} errored");
        }
    }

    #[test]
    fn search_matches_short_cjk_via_like_fallback() {
        let mut lib = Library::open_in_memory().unwrap();
        lib.save_meeting(
            "m1",
            &meta("会议"),
            0,
            &[seg(
                1,
                0,
                100,
                "我们讨论了预算和排期",
                AudioSourceKind::Microphone,
            )],
        )
        .unwrap();
        // "预算" is two characters — too short for the trigram index, so it routes through LIKE.
        let hits = lib.search("预算", 10).unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].meeting_id, "m1");
        assert!(hits[0].snippet.contains("预算"));
    }

    #[test]
    fn like_fallback_escapes_wildcards() {
        let mut lib = Library::open_in_memory().unwrap();
        lib.save_meeting(
            "lit",
            &meta("M"),
            0,
            &[seg(
                1,
                0,
                100,
                "literal 50% off",
                AudioSourceKind::Microphone,
            )],
        )
        .unwrap();
        lib.save_meeting(
            "oth",
            &meta("M"),
            1,
            &[seg(
                1,
                0,
                100,
                "50 percent saved",
                AudioSourceKind::Microphone,
            )],
        )
        .unwrap();
        // "%" is a LIKE wildcard; escaped, "0%" matches only the literal "0%".
        let hits = lib.search("0%", 10).unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].meeting_id, "lit");
    }

    #[test]
    fn long_transcript_preview_is_truncated_with_ellipsis() {
        let mut lib = Library::open_in_memory().unwrap();
        let long = "word ".repeat(200); // ~1000 chars, far over PREVIEW_CHARS
        lib.save_meeting(
            "m1",
            &meta("M"),
            0,
            &[seg(1, 0, 100, &long, AudioSourceKind::Microphone)],
        )
        .unwrap();
        let preview = lib.list_meetings().unwrap()[0].preview.clone();
        assert!(preview.chars().count() <= PREVIEW_CHARS + 1); // +1 for the ellipsis
        assert!(preview.ends_with('…'));
    }

    #[test]
    fn preview_truncation_is_char_safe_for_cjk() {
        let mut lib = Library::open_in_memory().unwrap();
        let long = "字".repeat(300); // multi-byte chars; truncation must split on a char boundary
        lib.save_meeting(
            "m1",
            &meta("M"),
            0,
            &[seg(1, 0, 100, &long, AudioSourceKind::Microphone)],
        )
        .unwrap();
        let preview = lib.list_meetings().unwrap()[0].preview.clone(); // must not panic
        assert!(preview.chars().count() <= PREVIEW_CHARS + 1);
        assert!(preview.starts_with('字'));
    }

    #[test]
    fn save_then_get_roundtrips_metadata_and_segments() {
        let mut lib = Library::open_in_memory().unwrap();
        let segs = [
            seg(1, 0, 1000, "Hello there", AudioSourceKind::Microphone),
            seg(2, 1000, 2500, "General Kenobi", AudioSourceKind::System),
        ];
        lib.save_meeting("m1", &meta("Standup"), 1000, &segs)
            .unwrap();

        assert_eq!(lib.count().unwrap(), 1);
        let (m, rows) = lib.get_meeting("m1").unwrap().unwrap();
        assert_eq!(m.title, "Standup");
        assert_eq!(m.started_at_ms, 1000);
        assert_eq!(m.duration_ms, 2500); // latest segment end
        assert_eq!(m.segment_count, 2);
        assert_eq!(m.engine.as_deref(), Some("whisper"));
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].text, "Hello there");
        assert_eq!(rows[0].source, "mic");
        assert_eq!(rows[1].source, "system");
    }

    #[test]
    fn save_skips_partial_and_blank_segments() {
        let mut lib = Library::open_in_memory().unwrap();
        let mut partial = seg(1, 0, 500, "in progress", AudioSourceKind::Microphone);
        partial.status = SegmentStatus::Partial;
        let blank = seg(2, 500, 600, "   ", AudioSourceKind::Microphone);
        let good = seg(3, 600, 1200, "committed", AudioSourceKind::Microphone);
        lib.save_meeting("m1", &meta("M"), 0, &[partial, blank, good])
            .unwrap();

        let (m, rows) = lib.get_meeting("m1").unwrap().unwrap();
        assert_eq!(m.segment_count, 1);
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].text, "committed");
    }

    #[test]
    fn save_maps_speaker_id() {
        let mut lib = Library::open_in_memory().unwrap();
        let mut s = seg(1, 0, 1000, "labelled", AudioSourceKind::Microphone);
        s.speaker = Some(SpeakerId(3));
        lib.save_meeting("m1", &meta("M"), 0, &[s]).unwrap();
        let (_, rows) = lib.get_meeting("m1").unwrap().unwrap();
        assert_eq!(rows[0].speaker, Some(3));
    }

    #[test]
    fn list_orders_newest_first_with_preview() {
        let mut lib = Library::open_in_memory().unwrap();
        lib.save_meeting(
            "old",
            &meta("Old"),
            1000,
            &[seg(
                1,
                0,
                1000,
                "old content here",
                AudioSourceKind::Microphone,
            )],
        )
        .unwrap();
        lib.save_meeting(
            "new",
            &meta("New"),
            5000,
            &[seg(
                1,
                0,
                1000,
                "new content here",
                AudioSourceKind::Microphone,
            )],
        )
        .unwrap();

        let list = lib.list_meetings().unwrap();
        assert_eq!(list.len(), 2);
        assert_eq!(list[0].id, "new"); // newest first
        assert_eq!(list[1].id, "old");
        assert!(list[0].preview.contains("new content"));
    }

    #[test]
    fn search_finds_a_term_and_groups_by_meeting() {
        let mut lib = Library::open_in_memory().unwrap();
        lib.save_meeting(
            "m1",
            &meta("Budget"),
            1000,
            &[
                seg(
                    1,
                    0,
                    1000,
                    "we discussed the quarterly budget",
                    AudioSourceKind::Microphone,
                ),
                seg(
                    2,
                    1000,
                    2000,
                    "and the budget again",
                    AudioSourceKind::Microphone,
                ),
            ],
        )
        .unwrap();
        lib.save_meeting(
            "m2",
            &meta("Other"),
            2000,
            &[seg(
                1,
                0,
                1000,
                "unrelated chatter",
                AudioSourceKind::Microphone,
            )],
        )
        .unwrap();

        let hits = lib.search("budget", 10).unwrap();
        assert_eq!(hits.len(), 1); // grouped: one hit for m1, m2 doesn't match
        assert_eq!(hits[0].meeting_id, "m1");
        assert!(hits[0].snippet.contains('«')); // matched term is marked
    }

    #[test]
    fn search_empty_query_returns_nothing() {
        let lib = Library::open_in_memory().unwrap();
        assert!(lib.search("   ", 10).unwrap().is_empty());
    }

    #[test]
    fn search_matches_cjk_substring() {
        let mut lib = Library::open_in_memory().unwrap();
        lib.save_meeting(
            "m1",
            &meta("会议"),
            1000,
            &[seg(
                1,
                0,
                1000,
                "今天讨论了产品路线图和预算",
                AudioSourceKind::Microphone,
            )],
        )
        .unwrap();

        let hits = lib.search("产品路线", 10).unwrap(); // >= 3 chars, trigram substring
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].meeting_id, "m1");
    }

    #[test]
    fn delete_removes_meeting_segments_and_index() {
        let mut lib = Library::open_in_memory().unwrap();
        lib.save_meeting(
            "m1",
            &meta("M"),
            1000,
            &[seg(
                1,
                0,
                1000,
                "deletable budget line",
                AudioSourceKind::Microphone,
            )],
        )
        .unwrap();

        assert!(lib.delete_meeting("m1").unwrap());
        assert_eq!(lib.count().unwrap(), 0);
        assert!(lib.get_meeting("m1").unwrap().is_none());
        assert!(lib.search("budget", 10).unwrap().is_empty()); // FTS cleared by the cascade trigger
        assert!(!lib.delete_meeting("m1").unwrap()); // already gone
    }

    #[test]
    fn resaving_same_id_replaces() {
        let mut lib = Library::open_in_memory().unwrap();
        lib.save_meeting(
            "m1",
            &meta("First"),
            1000,
            &[seg(
                1,
                0,
                1000,
                "first version",
                AudioSourceKind::Microphone,
            )],
        )
        .unwrap();
        lib.save_meeting(
            "m1",
            &meta("Second"),
            2000,
            &[seg(
                1,
                0,
                1000,
                "second version",
                AudioSourceKind::Microphone,
            )],
        )
        .unwrap();

        assert_eq!(lib.count().unwrap(), 1);
        let (m, rows) = lib.get_meeting("m1").unwrap().unwrap();
        assert_eq!(m.title, "Second");
        assert_eq!(rows[0].text, "second version");
        assert!(lib.search("first", 10).unwrap().is_empty()); // old index gone
    }

    #[test]
    fn reopen_is_idempotent() {
        // Migrating an already-migrated DB must be a no-op (no "table already exists").
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("lib.db");
        {
            let mut lib = Library::open(&path).unwrap();
            lib.save_meeting(
                "m1",
                &meta("M"),
                1000,
                &[seg(1, 0, 1000, "persisted", AudioSourceKind::Microphone)],
            )
            .unwrap();
        }
        let lib = Library::open(&path).unwrap(); // second open: user_version is 1, schema skipped
        assert_eq!(lib.count().unwrap(), 1);
    }
}
