//! SHA-256 helpers for verifying downloaded model files.

use std::fmt::Write as _;
use std::fs::File;
use std::io::Read;
use std::path::Path;

use sha2::{Digest, Sha256};
use wisp_core::error::{Result, WispError};

/// Returns the lowercase hex SHA-256 of `bytes`.
pub fn sha256_hex(bytes: &[u8]) -> String {
    to_hex(Sha256::digest(bytes).as_slice())
}

/// Returns the lowercase hex SHA-256 of the file at `path`, hashing it in a streaming fashion so
/// that multi-gigabyte models are never held in memory at once.
pub fn sha256_file(path: &Path) -> Result<String> {
    let mut file = File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 64 * 1024];

    loop {
        let read = file.read(&mut buf)?;
        if read == 0 {
            break;
        }
        hasher.update(&buf[..read]);
    }

    Ok(to_hex(hasher.finalize().as_slice()))
}

/// Verifies that the file at `path` hashes to `expected` (case-insensitive hex).
pub fn verify_file(path: &Path, expected: &str) -> Result<()> {
    let actual = sha256_file(path)?;
    if actual.eq_ignore_ascii_case(expected) {
        return Ok(());
    }

    Err(WispError::Model(format!(
        "checksum mismatch for {}: expected {expected}, got {actual}",
        path.display()
    )))
}

fn to_hex(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        let _ = write!(out, "{byte:02x}");
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    // Known-answer vectors from FIPS 180-2.
    const ABC_SHA256: &str = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";
    const EMPTY_SHA256: &str = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";

    #[test]
    fn sha256_of_abc_matches_known_vector() {
        assert_eq!(sha256_hex(b"abc"), ABC_SHA256);
    }

    #[test]
    fn sha256_of_empty_matches_known_vector() {
        assert_eq!(sha256_hex(b""), EMPTY_SHA256);
    }

    #[test]
    fn verify_file_accepts_match_and_rejects_mismatch() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("f.bin");
        std::fs::write(&path, b"abc").unwrap();

        verify_file(&path, ABC_SHA256).unwrap();
        // Case-insensitive comparison.
        verify_file(&path, &ABC_SHA256.to_uppercase()).unwrap();
        assert!(verify_file(&path, "deadbeef").is_err());
    }
}
