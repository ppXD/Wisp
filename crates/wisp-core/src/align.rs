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
}
