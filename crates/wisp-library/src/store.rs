//! The SQLite-backed store: persistence, retrieval, and full-text search over past meetings.

use std::collections::{HashMap, HashSet};
use std::path::Path;

use rusqlite::{Connection, OptionalExtension};
use wisp_core::export::MeetingMeta;
use wisp_core::transcript::{AudioSourceKind, SegmentStatus, TranscriptSegment};

use crate::embed::{self, Embedder};
use crate::record::{Note, NoteSummary, SearchHit, Segment};
use crate::Result;

/// On-disk schema version, bumped on schema changes (drives migration via `PRAGMA user_version`).
const SCHEMA_VERSION: i64 = 2;

/// Characters of transcript kept as a list preview.
const PREVIEW_CHARS: usize = 160;

/// Characters of the matching segment shown as a snippet for a LIKE-fallback hit. FTS hits use
/// SQLite's own `snippet()` instead.
const LIKE_SNIPPET_CHARS: usize = 120;

/// Target size (characters) of a text chunk handed to the embedder — a coherent unit larger than one
/// utterance but small enough to embed precisely.
const CHUNK_CHARS: usize = 512;

/// Reciprocal Rank Fusion damping constant for hybrid search (the conventional default).
const RRF_K: f64 = 60.0;

/// A chunk ready to store: its text plus its embedding serialized to little-endian bytes.
type EmbeddedChunk = (String, Vec<u8>);

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

/// Schema v2 — semantic search. Each note's transcript is chunked and each chunk's embedding vector
/// is stored as a little-endian f32 BLOB. Written only when an [`Embedder`] is configured; the
/// cascade from `meeting` clears a note's chunks on delete or re-save. Added as a separate migration
/// so existing v1 databases gain the table without losing data.
const SCHEMA_V2: &str = "\
CREATE TABLE chunk (
    id         INTEGER PRIMARY KEY,
    meeting_id TEXT NOT NULL REFERENCES meeting (id) ON DELETE CASCADE,
    idx        INTEGER NOT NULL,
    text       TEXT NOT NULL,
    embedding  BLOB NOT NULL
);
CREATE INDEX chunk_meeting ON chunk (meeting_id);
";

/// A handle to the meeting knowledge base. Open once and reuse across queries. With no embedder it
/// is full-text only; configure one via [`Library::set_embedder`] to enable semantic and hybrid
/// search.
pub struct Library {
    conn: Connection,
    embedder: Option<Box<dyn Embedder>>,
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

    /// Sets (or with `None` clears) the embedder used to index and search notes semantically. New
    /// saves embed their chunks; existing notes keep whatever was indexed when they were saved.
    pub fn set_embedder(&mut self, embedder: Option<Box<dyn Embedder>>) {
        self.embedder = embedder;
    }

    fn init(conn: Connection) -> Result<Self> {
        conn.execute_batch("PRAGMA foreign_keys = ON;")?;
        let mut lib = Self {
            conn,
            embedder: None,
        };
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
        if version < 2 {
            self.conn.execute_batch(SCHEMA_V2)?;
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
    pub fn save_note(
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

        // Embed before opening the transaction — inference touches no DB and may be slow.
        let chunks = self.embed_chunks(&finals)?;

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
        if let Some(chunks) = &chunks {
            insert_chunks(&tx, id, chunks)?;
        }
        tx.commit()?;
        Ok(())
    }

    /// Chunks a note's finalized text and embeds each chunk, when an embedder is configured. Each
    /// entry is `(chunk text, serialized vector)`; returns `None` in full-text-only mode so the
    /// caller writes no `chunk` rows.
    fn embed_chunks(&self, finals: &[&TranscriptSegment]) -> Result<Option<Vec<EmbeddedChunk>>> {
        let Some(embedder) = &self.embedder else {
            return Ok(None);
        };

        let texts: Vec<&str> = finals.iter().map(|s| s.text.trim()).collect();
        let chunks = embed::chunk_texts(&texts, CHUNK_CHARS);
        if chunks.is_empty() {
            return Ok(Some(Vec::new()));
        }

        let refs: Vec<&str> = chunks.iter().map(String::as_str).collect();
        let vectors = embedder.embed_passages(&refs)?;

        Ok(Some(
            chunks
                .into_iter()
                .zip(vectors)
                .map(|(text, vector)| (text, embed::to_bytes(&vector)))
                .collect(),
        ))
    }

    /// Every meeting, newest first, each with a short transcript preview — for the Library list.
    pub fn list_notes(&self) -> Result<Vec<NoteSummary>> {
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
                Ok(NoteSummary {
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
    pub fn get_note(&self, id: &str) -> Result<Option<(Note, Vec<Segment>)>> {
        let meeting = self
            .conn
            .query_row(
                "SELECT id, title, started_at_ms, duration_ms, language, engine, summary, segment_count
                 FROM meeting WHERE id = ?1",
                [id],
                |r| {
                    Ok(Note {
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

    /// Semantic search: embeds `query` and ranks notes by the cosine similarity of their chunk
    /// vectors, returning the best-scoring snippet per note (most similar first, capped at `limit`).
    /// Yields nothing when no embedder is configured or the query is blank. A personal library is
    /// small, so this scans every chunk rather than maintaining an approximate index.
    pub fn search_semantic(&self, query: &str, limit: usize) -> Result<Vec<SearchHit>> {
        let Some(embedder) = &self.embedder else {
            return Ok(Vec::new());
        };
        let query = query.trim();
        if query.is_empty() {
            return Ok(Vec::new());
        }

        let qvec = embedder.embed_query(query)?;

        let mut stmt = self.conn.prepare(
            "SELECT c.meeting_id, m.title, m.started_at_ms, c.text, c.embedding
             FROM chunk c JOIN meeting m ON m.id = c.meeting_id",
        )?;
        let mut hits = stmt
            .query_map([], |r| {
                let text: String = r.get(3)?;
                let blob: Vec<u8> = r.get(4)?;
                Ok(SearchHit {
                    meeting_id: r.get(0)?,
                    title: r.get(1)?,
                    started_at_ms: r.get(2)?,
                    snippet: truncate_chars(&text, LIKE_SNIPPET_CHARS),
                    score: f64::from(embed::dot(&qvec, &embed::from_bytes(&blob))),
                })
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?;

        // Drop non-matches (orthogonal / negatively-correlated chunks), then rank by similarity
        // (higher is better, unlike BM25) and keep the best chunk per note.
        hits.retain(|h| h.score > 0.0);
        hits.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        Ok(best_per_meeting(hits, limit))
    }

    /// Hybrid search: fuses the full-text and semantic rankings with Reciprocal Rank Fusion — the
    /// most accurate mode, since lexical and semantic matches reinforce each other. Falls back to
    /// plain [`Library::search`] when no embedder is configured. The fused RRF value becomes the
    /// score; the snippet prefers the highlighted full-text one.
    pub fn search_hybrid(&self, query: &str, limit: usize) -> Result<Vec<SearchHit>> {
        if self.embedder.is_none() {
            return self.search(query, limit);
        }

        let pool = limit.max(20);
        let fts = self.search(query, pool)?;
        let semantic = self.search_semantic(query, pool)?;

        let rankings: [Vec<String>; 2] = [
            fts.iter().map(|h| h.meeting_id.clone()).collect(),
            semantic.iter().map(|h| h.meeting_id.clone()).collect(),
        ];
        let fused = embed::rrf_fuse(&rankings, RRF_K);

        // FTS chained last so it overwrites the semantic entry for a shared note — its «»-highlighted
        // snippet is the better one to show.
        let by_id: HashMap<&str, &SearchHit> = semantic
            .iter()
            .chain(fts.iter())
            .map(|h| (h.meeting_id.as_str(), h))
            .collect();

        Ok(fused
            .into_iter()
            .take(limit)
            .filter_map(|(id, score)| {
                by_id.get(id.as_str()).map(|h| SearchHit {
                    meeting_id: id,
                    title: h.title.clone(),
                    started_at_ms: h.started_at_ms,
                    snippet: h.snippet.clone(),
                    score,
                })
            })
            .collect())
    }

    /// Deletes a meeting and its segments + search index. Returns whether a meeting existed.
    pub fn delete_note(&self, id: &str) -> Result<bool> {
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
/// insert trigger). Factored out of [`Library::save_note`] so that method reads as a flat pipeline.
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

/// Inserts a note's embedded chunks (text + serialized vector) into the open transaction. Factored
/// out of [`Library::save_note`] to keep that method a flat pipeline.
fn insert_chunks(
    tx: &rusqlite::Transaction,
    meeting_id: &str,
    chunks: &[EmbeddedChunk],
) -> rusqlite::Result<()> {
    let mut stmt =
        tx.prepare("INSERT INTO chunk (meeting_id, idx, text, embedding) VALUES (?1, ?2, ?3, ?4)")?;
    for (idx, (text, embedding)) in chunks.iter().enumerate() {
        stmt.execute(rusqlite::params![meeting_id, idx as i64, text, embedding])?;
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

    /// Deterministic test embedder: each whitespace-delimited token bumps one (hashed) dimension and
    /// the vector is L2-normalized. Texts sharing tokens get similar vectors — enough to exercise
    /// storage, scoring, ranking, and fusion without loading a real model.
    struct HashingEmbedder {
        dim: usize,
    }

    impl HashingEmbedder {
        // Symmetric here (no passage/query prefix), so both trait methods share this.
        fn vec(&self, t: &str) -> Vec<f32> {
            let mut v = vec![0f32; self.dim];
            for tok in t.split_whitespace() {
                let h = tok
                    .bytes()
                    .fold(0usize, |a, b| a.wrapping_mul(31).wrapping_add(b as usize));
                v[h % self.dim] += 1.0;
            }
            let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                for x in &mut v {
                    *x /= norm;
                }
            }
            v
        }
    }

    impl Embedder for HashingEmbedder {
        fn dim(&self) -> usize {
            self.dim
        }

        fn embed_passages(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
            Ok(texts.iter().map(|t| self.vec(t)).collect())
        }

        fn embed_query(&self, text: &str) -> Result<Vec<f32>> {
            Ok(self.vec(text))
        }
    }

    /// An in-memory library with the deterministic test embedder configured.
    fn lib_with_embedder() -> Library {
        let mut lib = Library::open_in_memory().unwrap();
        lib.set_embedder(Some(Box::new(HashingEmbedder { dim: 64 })));
        lib
    }

    #[test]
    fn save_with_no_final_segments_stores_an_empty_meeting() {
        let mut lib = Library::open_in_memory().unwrap();
        lib.save_note("m1", &meta("Silent"), 1000, &[]).unwrap();
        let (m, rows) = lib.get_note("m1").unwrap().unwrap();
        assert_eq!(m.segment_count, 0);
        assert_eq!(m.duration_ms, 0);
        assert!(rows.is_empty());
    }

    #[test]
    fn get_missing_meeting_returns_none() {
        let lib = Library::open_in_memory().unwrap();
        assert!(lib.get_note("nope").unwrap().is_none());
    }

    #[test]
    fn list_is_empty_with_no_meetings() {
        let lib = Library::open_in_memory().unwrap();
        assert!(lib.list_notes().unwrap().is_empty());
    }

    #[test]
    fn source_label_covers_every_kind() {
        let mut lib = Library::open_in_memory().unwrap();
        lib.save_note(
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
        let (_, rows) = lib.get_note("m1").unwrap().unwrap();
        let sources: Vec<&str> = rows.iter().map(|s| s.source.as_str()).collect();
        assert_eq!(sources, ["mic", "system", "file"]);
    }

    #[test]
    fn search_respects_the_limit() {
        let mut lib = Library::open_in_memory().unwrap();
        for i in 0i64..5 {
            lib.save_note(
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
        lib.save_note(
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
        lib.save_note(
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
        lib.save_note(
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
        lib.save_note(
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
        lib.save_note(
            "m1",
            &meta("M"),
            0,
            &[seg(1, 0, 100, &long, AudioSourceKind::Microphone)],
        )
        .unwrap();
        let preview = lib.list_notes().unwrap()[0].preview.clone();
        assert!(preview.chars().count() <= PREVIEW_CHARS + 1); // +1 for the ellipsis
        assert!(preview.ends_with('…'));
    }

    #[test]
    fn preview_truncation_is_char_safe_for_cjk() {
        let mut lib = Library::open_in_memory().unwrap();
        let long = "字".repeat(300); // multi-byte chars; truncation must split on a char boundary
        lib.save_note(
            "m1",
            &meta("M"),
            0,
            &[seg(1, 0, 100, &long, AudioSourceKind::Microphone)],
        )
        .unwrap();
        let preview = lib.list_notes().unwrap()[0].preview.clone(); // must not panic
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
        lib.save_note("m1", &meta("Standup"), 1000, &segs).unwrap();

        assert_eq!(lib.count().unwrap(), 1);
        let (m, rows) = lib.get_note("m1").unwrap().unwrap();
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
        lib.save_note("m1", &meta("M"), 0, &[partial, blank, good])
            .unwrap();

        let (m, rows) = lib.get_note("m1").unwrap().unwrap();
        assert_eq!(m.segment_count, 1);
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].text, "committed");
    }

    #[test]
    fn save_maps_speaker_id() {
        let mut lib = Library::open_in_memory().unwrap();
        let mut s = seg(1, 0, 1000, "labelled", AudioSourceKind::Microphone);
        s.speaker = Some(SpeakerId(3));
        lib.save_note("m1", &meta("M"), 0, &[s]).unwrap();
        let (_, rows) = lib.get_note("m1").unwrap().unwrap();
        assert_eq!(rows[0].speaker, Some(3));
    }

    #[test]
    fn list_orders_newest_first_with_preview() {
        let mut lib = Library::open_in_memory().unwrap();
        lib.save_note(
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
        lib.save_note(
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

        let list = lib.list_notes().unwrap();
        assert_eq!(list.len(), 2);
        assert_eq!(list[0].id, "new"); // newest first
        assert_eq!(list[1].id, "old");
        assert!(list[0].preview.contains("new content"));
    }

    #[test]
    fn search_finds_a_term_and_groups_by_meeting() {
        let mut lib = Library::open_in_memory().unwrap();
        lib.save_note(
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
        lib.save_note(
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
        lib.save_note(
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
        lib.save_note(
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

        assert!(lib.delete_note("m1").unwrap());
        assert_eq!(lib.count().unwrap(), 0);
        assert!(lib.get_note("m1").unwrap().is_none());
        assert!(lib.search("budget", 10).unwrap().is_empty()); // FTS cleared by the cascade trigger
        assert!(!lib.delete_note("m1").unwrap()); // already gone
    }

    #[test]
    fn resaving_same_id_replaces() {
        let mut lib = Library::open_in_memory().unwrap();
        lib.save_note(
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
        lib.save_note(
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
        let (m, rows) = lib.get_note("m1").unwrap().unwrap();
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
            lib.save_note(
                "m1",
                &meta("M"),
                1000,
                &[seg(1, 0, 1000, "persisted", AudioSourceKind::Microphone)],
            )
            .unwrap();
        }
        let lib = Library::open(&path).unwrap(); // second open: user_version is 2, schema skipped
        assert_eq!(lib.count().unwrap(), 1);
    }

    #[test]
    fn semantic_search_is_empty_without_an_embedder() {
        let mut lib = Library::open_in_memory().unwrap();
        lib.save_note(
            "m1",
            &meta("M"),
            0,
            &[seg(1, 0, 100, "budget review", AudioSourceKind::Microphone)],
        )
        .unwrap();
        assert!(lib.search_semantic("budget", 10).unwrap().is_empty());
    }

    #[test]
    fn save_without_an_embedder_indexes_no_chunks() {
        // Saved full-text-only, then an embedder is attached: there are no chunks to find, so
        // semantic search stays empty until the note is re-saved with embedding on.
        let mut lib = Library::open_in_memory().unwrap();
        lib.save_note(
            "m1",
            &meta("M"),
            0,
            &[seg(1, 0, 100, "budget review", AudioSourceKind::Microphone)],
        )
        .unwrap();
        lib.set_embedder(Some(Box::new(HashingEmbedder { dim: 256 })));
        assert!(lib.search_semantic("budget", 10).unwrap().is_empty());
    }

    #[test]
    fn semantic_search_finds_and_ranks_notes_by_content() {
        let mut lib = lib_with_embedder();
        lib.save_note(
            "budget",
            &meta("Budget"),
            1,
            &[seg(
                1,
                0,
                100,
                "the annual budget report",
                AudioSourceKind::Microphone,
            )],
        )
        .unwrap();
        lib.save_note(
            "schedule",
            &meta("Schedule"),
            2,
            &[seg(
                1,
                0,
                100,
                "the weekly schedule planning",
                AudioSourceKind::Microphone,
            )],
        )
        .unwrap();

        // Each query's exact-token note ranks first (it owns that dimension outright).
        assert_eq!(
            lib.search_semantic("budget", 10).unwrap()[0].meeting_id,
            "budget"
        );
        assert_eq!(
            lib.search_semantic("schedule", 10).unwrap()[0].meeting_id,
            "schedule"
        );
    }

    #[test]
    fn semantic_search_groups_a_multi_chunk_note_to_one_hit() {
        let mut lib = lib_with_embedder();
        // Two long segments exceed CHUNK_CHARS together, so they store as two chunks of one note.
        let long = "keyword ".repeat(60);
        lib.save_note(
            "m1",
            &meta("M"),
            0,
            &[
                seg(1, 0, 100, &long, AudioSourceKind::Microphone),
                seg(2, 100, 200, &long, AudioSourceKind::System),
            ],
        )
        .unwrap();
        // Both chunks match, but the result collapses to a single hit for the note.
        assert_eq!(lib.search_semantic("keyword", 10).unwrap().len(), 1);
    }

    #[test]
    fn hybrid_search_returns_the_matching_note() {
        let mut lib = lib_with_embedder();
        lib.save_note(
            "budget",
            &meta("Budget"),
            1,
            &[seg(
                1,
                0,
                100,
                "the annual budget report",
                AudioSourceKind::Microphone,
            )],
        )
        .unwrap();
        lib.save_note(
            "schedule",
            &meta("Schedule"),
            2,
            &[seg(
                1,
                0,
                100,
                "the weekly schedule planning",
                AudioSourceKind::Microphone,
            )],
        )
        .unwrap();

        // Full-text and semantic both surface the budget note for "budget", so it fuses to the top.
        assert_eq!(
            lib.search_hybrid("budget", 10).unwrap()[0].meeting_id,
            "budget"
        );
    }

    #[test]
    fn hybrid_search_without_an_embedder_falls_back_to_full_text() {
        let mut lib = Library::open_in_memory().unwrap();
        lib.save_note(
            "m1",
            &meta("M"),
            0,
            &[seg(
                1,
                0,
                100,
                "the annual budget report",
                AudioSourceKind::Microphone,
            )],
        )
        .unwrap();
        // No embedder → identical to the full-text search.
        let hybrid = lib.search_hybrid("budget", 10).unwrap();
        assert_eq!(hybrid.len(), 1);
        assert_eq!(hybrid[0].meeting_id, "m1");
    }

    #[test]
    fn delete_removes_a_notes_chunks() {
        let mut lib = lib_with_embedder();
        lib.save_note(
            "m1",
            &meta("M"),
            0,
            &[seg(1, 0, 100, "budget review", AudioSourceKind::Microphone)],
        )
        .unwrap();
        assert_eq!(lib.search_semantic("budget", 10).unwrap().len(), 1);

        assert!(lib.delete_note("m1").unwrap());
        assert!(lib.search_semantic("budget", 10).unwrap().is_empty()); // chunks gone via cascade
    }

    #[test]
    fn migration_adds_the_chunk_table_to_a_v1_database() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("lib.db");

        // Build a v1 database by hand: the original schema, a stored note, user_version = 1.
        {
            let conn = Connection::open(&path).unwrap();
            conn.execute_batch(SCHEMA_V1).unwrap();
            conn.execute(
                "INSERT INTO meeting (id, title, started_at_ms, duration_ms, segment_count)
                 VALUES ('old', 'Old', 1000, 0, 0)",
                [],
            )
            .unwrap();
            conn.pragma_update(None, "user_version", 1).unwrap();
        }

        // Opening runs the v1→v2 migration: the chunk table appears and existing data survives.
        let mut lib = Library::open(&path).unwrap();
        assert_eq!(lib.count().unwrap(), 1);

        lib.set_embedder(Some(Box::new(HashingEmbedder { dim: 256 })));
        lib.save_note(
            "new",
            &meta("New"),
            2000,
            &[seg(1, 0, 100, "budget review", AudioSourceKind::Microphone)],
        )
        .unwrap();
        // The new note's chunk is searchable — proving the chunk table exists and works.
        assert_eq!(lib.search_semantic("budget", 10).unwrap().len(), 1);
    }
}
