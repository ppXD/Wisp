//! Word-level alignment: turning an engine's per-token timings into [`Word`]s.
//!
//! ASR engines time their output per decoder *token*, which is finer than a word (an English word
//! can span several sub-word tokens) and language-dependent (CJK scripts have no spaces, so a
//! token is roughly one character). This module groups those raw tokens into words so subtitle
//! export can carry word-level timing. The logic lives in the core, independent of any engine, so
//! it is fully unit-tested.

use std::time::Duration;

use crate::transcript::Word;

/// One decoder token with its time span, as produced by an engine's token timestamps.
#[derive(Debug, Clone, PartialEq)]
pub struct TokenTiming {
    /// Raw token text, including any leading space the engine emits at word starts.
    pub text: String,
    /// Token start offset from the clip start.
    pub start: Duration,
    /// Token end offset from the clip start.
    pub end: Duration,
}

impl TokenTiming {
    /// Convenience constructor.
    pub fn new(text: impl Into<String>, start: Duration, end: Duration) -> Self {
        Self {
            text: text.into(),
            start,
            end,
        }
    }
}

/// Group raw decoder `tokens` into words.
///
/// A new word begins at a leading space (space-delimited languages), at any CJK token (each is its
/// own word, since CJK has no word spaces), or when crossing out of CJK. Whisper's special tokens
/// (`[_BEG_]`, timestamp markers — they start with `[`) and empty tokens are dropped. Each word
/// keeps the leading spacing of its first token so a line can be reconstructed verbatim.
pub fn merge_tokens_into_words(tokens: &[TokenTiming]) -> Vec<Word> {
    let mut words: Vec<Word> = Vec::new();
    for token in tokens {
        if token.text.is_empty() || token.text.starts_with('[') {
            continue;
        }

        let starts_new_word = token.text.starts_with(char::is_whitespace)
            || starts_with_cjk(&token.text)
            || words.last().is_none_or(|w| ends_with_cjk(&w.text));

        if starts_new_word {
            words.push(Word {
                text: token.text.clone(),
                start: token.start,
                end: token.end,
            });
        } else {
            let word = words
                .last_mut()
                .expect("non-empty: starts_new_word covers the empty case");
            word.text.push_str(&token.text);
            word.end = token.end;
        }
    }
    words
}

/// Build [`Word`]s from an engine's per-token output: `tokens[i]` is the i-th decoder token's text
/// and `timestamps[i]` its start offset (seconds) from the clip start. Each token's end is the next
/// token's start (the last runs to `clip_secs`). SentencePiece (`▁`, U+2581) and GPT-2 (`Ġ`, U+0120)
/// word-boundary markers at a token's start are normalized to a leading space so word splitting
/// fires, then [`merge_tokens_into_words`] groups sub-word tokens and isolates CJK characters.
///
/// Returns no words when `tokens`/`timestamps` are empty or their lengths disagree — i.e. the engine
/// didn't emit per-token timings — so the caller can fall back to a word-less segment.
pub fn words_from_token_timestamps(
    tokens: &[String],
    timestamps: &[f32],
    clip_secs: f32,
) -> Vec<Word> {
    if tokens.is_empty() || tokens.len() != timestamps.len() {
        return Vec::new();
    }

    let timings: Vec<TokenTiming> = tokens
        .iter()
        .enumerate()
        .map(|(i, text)| {
            let start = timestamps[i].max(0.0);
            let end = timestamps
                .get(i + 1)
                .copied()
                .unwrap_or(clip_secs)
                .max(start);
            TokenTiming::new(normalize_word_marker(text), secs(start), secs(end))
        })
        .collect();

    merge_tokens_into_words(&timings)
}

/// Replace a leading SentencePiece (`▁`) or GPT-2 (`Ġ`) space marker with an ASCII space so the
/// word-boundary detection in [`merge_tokens_into_words`] treats the token as a new word. Other
/// tokens (sub-word continuations, CJK characters) are returned unchanged.
fn normalize_word_marker(token: &str) -> String {
    match token
        .strip_prefix('\u{2581}')
        .or_else(|| token.strip_prefix('\u{0120}'))
    {
        Some(rest) => format!(" {rest}"),
        None => token.to_owned(),
    }
}

/// Seconds (clamped non-negative; `NaN` → 0) as a [`Duration`], so a stray timestamp can't panic
/// [`Duration::from_secs_f32`].
fn secs(s: f32) -> Duration {
    Duration::from_secs_f32(s.max(0.0))
}

/// Whether the first non-space character of `s` is CJK.
fn starts_with_cjk(s: &str) -> bool {
    s.trim_start().chars().next().is_some_and(is_cjk)
}

/// Whether the last character of `s` is CJK.
fn ends_with_cjk(s: &str) -> bool {
    s.chars().next_back().is_some_and(is_cjk)
}

/// Whether `c` belongs to a CJK script (so it stands alone as a word).
fn is_cjk(c: char) -> bool {
    matches!(c,
        '\u{3040}'..='\u{30FF}'   // Hiragana + Katakana
        | '\u{3400}'..='\u{4DBF}' // CJK Extension A
        | '\u{4E00}'..='\u{9FFF}' // CJK Unified Ideographs
        | '\u{AC00}'..='\u{D7AF}' // Hangul syllables
        | '\u{F900}'..='\u{FAFF}' // CJK Compatibility Ideographs
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tok(text: &str, start_ms: u64, end_ms: u64) -> TokenTiming {
        TokenTiming::new(
            text,
            Duration::from_millis(start_ms),
            Duration::from_millis(end_ms),
        )
    }

    #[test]
    fn splits_space_delimited_words() {
        let words = merge_tokens_into_words(&[tok(" Hello", 0, 500), tok(" world", 500, 1_000)]);
        assert_eq!(words.len(), 2);
        assert_eq!(words[0].text.trim(), "Hello");
        assert_eq!(words[0].start, Duration::from_millis(0));
        assert_eq!(words[1].text.trim(), "world");
        assert_eq!(words[1].end, Duration::from_millis(1_000));
    }

    #[test]
    fn merges_subword_tokens_into_one_word() {
        // No leading space on the continuation token ⇒ it joins the current word.
        let words = merge_tokens_into_words(&[tok(" trans", 0, 300), tok("cription", 300, 600)]);
        assert_eq!(words.len(), 1);
        assert_eq!(words[0].text.trim(), "transcription");
        assert_eq!(words[0].start, Duration::from_millis(0));
        assert_eq!(words[0].end, Duration::from_millis(600));
    }

    #[test]
    fn each_cjk_token_is_its_own_word() {
        let words = merge_tokens_into_words(&[tok("你", 0, 200), tok("好", 200, 400)]);
        assert_eq!(words.len(), 2);
        assert_eq!(words[0].text, "你");
        assert_eq!(words[1].text, "好");
        assert_eq!(words[1].start, Duration::from_millis(200));
    }

    #[test]
    fn separates_at_cjk_latin_boundary() {
        let words = merge_tokens_into_words(&[tok(" OK", 0, 300), tok("好", 300, 500)]);
        assert_eq!(words.len(), 2);
        assert_eq!(words[0].text.trim(), "OK");
        assert_eq!(words[1].text, "好");
    }

    #[test]
    fn drops_special_and_empty_tokens() {
        let words =
            merge_tokens_into_words(&[tok("[_BEG_]", 0, 0), tok("", 0, 0), tok(" Hi", 0, 400)]);
        assert_eq!(words.len(), 1);
        assert_eq!(words[0].text.trim(), "Hi");
    }

    #[test]
    fn empty_input_yields_no_words() {
        assert!(merge_tokens_into_words(&[]).is_empty());
    }

    #[test]
    fn words_from_tokens_derive_ends_from_next_start_and_clip_end() {
        // SentencePiece "▁" marks word starts; each token's end is the next token's start, and the
        // final token runs to the clip end.
        let tokens = vec!["\u{2581}hello".to_owned(), "\u{2581}world".to_owned()];
        let timestamps = vec![0.0_f32, 0.5];
        let words = words_from_token_timestamps(&tokens, &timestamps, 1.0);

        assert_eq!(words.len(), 2);
        assert_eq!(words[0].text.trim(), "hello");
        assert_eq!(words[0].start, Duration::from_secs_f32(0.0));
        assert_eq!(
            words[0].end,
            Duration::from_secs_f32(0.5),
            "end = next token's start"
        );
        assert_eq!(words[1].text.trim(), "world");
        assert_eq!(
            words[1].end,
            Duration::from_secs_f32(1.0),
            "last word runs to the clip end"
        );
    }

    #[test]
    fn words_from_tokens_merge_subword_pieces() {
        // A continuation token with no leading marker joins the current word.
        let tokens = vec!["\u{2581}trans".to_owned(), "cription".to_owned()];
        let timestamps = vec![0.0_f32, 0.3];
        let words = words_from_token_timestamps(&tokens, &timestamps, 0.6);

        assert_eq!(words.len(), 1);
        assert_eq!(words[0].text.trim(), "transcription");
        assert_eq!(words[0].start, Duration::from_secs_f32(0.0));
        assert_eq!(words[0].end, Duration::from_secs_f32(0.6));
    }

    #[test]
    fn words_from_tokens_handle_gpt2_marker_and_cjk() {
        // GPT-2 "Ġ" marker for a Latin word; CJK characters are each their own word regardless.
        let latin = words_from_token_timestamps(&["\u{0120}hi".to_owned()], &[0.0], 0.4);
        assert_eq!(latin.len(), 1);
        assert_eq!(latin[0].text.trim(), "hi");

        let cjk =
            words_from_token_timestamps(&["你".to_owned(), "好".to_owned()], &[0.0, 0.2], 0.4);
        assert_eq!(cjk.len(), 2);
        assert_eq!(cjk[0].text, "你");
        assert_eq!(cjk[1].text, "好");
        assert_eq!(cjk[1].start, Duration::from_secs_f32(0.2));
    }

    #[test]
    fn words_from_tokens_are_empty_when_timings_missing_or_mismatched() {
        assert!(words_from_token_timestamps(&[], &[], 1.0).is_empty());
        // An engine that returned text but no timestamps (length mismatch) yields no words, so the
        // caller keeps a word-less segment instead of fabricating spans.
        assert!(words_from_token_timestamps(&["hi".to_owned()], &[], 1.0).is_empty());
    }
}
