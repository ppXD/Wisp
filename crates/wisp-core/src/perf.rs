//! Shared runtime performance policy for the ASR engines.
//!
//! Pure on purpose — the hardware probe (physical-core count) is supplied by the caller, so the
//! policy unit-tests cleanly with no machine dependency. The native engine crates pass
//! `num_cpus::get_physical()`.

/// Env var overriding the ASR worker-thread count (e.g. `WISP_ASR_THREADS=6`). Pinned by a test, so
/// renaming it is a deliberate, visible change for anyone who tuned it.
pub const ASR_THREADS_ENV: &str = "WISP_ASR_THREADS";

/// Worker-thread count for CPU-bound ASR inference (onnxruntime intra-op, whisper.cpp decode), given
/// an optional [`ASR_THREADS_ENV`] override and the machine's detected `physical_cores`.
///
/// Defaults to the physical-core count — the sweet spot for compute-bound inference, since scaling
/// onto hyperthreads adds scheduling contention without throughput. A positive override wins; the
/// result is never below 1 (a single-threaded engine still runs).
pub fn asr_threads(override_var: Option<String>, physical_cores: usize) -> usize {
    if let Some(n) = override_var
        .and_then(|v| v.trim().parse::<usize>().ok())
        .filter(|&n| n >= 1)
    {
        return n;
    }
    physical_cores.max(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn env_var_name_is_pinned() {
        // Renaming this silently breaks anyone who pinned a thread count via env. Pin the literal.
        assert_eq!(ASR_THREADS_ENV, "WISP_ASR_THREADS");
    }

    #[test]
    fn defaults_to_physical_cores() {
        assert_eq!(asr_threads(None, 6), 6);
    }

    #[test]
    fn positive_override_wins_and_is_trimmed() {
        assert_eq!(asr_threads(Some("  4 ".to_owned()), 16), 4);
    }

    #[test]
    fn zero_negative_or_invalid_override_falls_back_to_physical() {
        assert_eq!(asr_threads(Some("0".to_owned()), 8), 8);
        assert_eq!(asr_threads(Some("-2".to_owned()), 8), 8);
        assert_eq!(asr_threads(Some("eight".to_owned()), 8), 8);
    }

    #[test]
    fn never_below_one_even_with_zero_cores() {
        assert_eq!(asr_threads(None, 0), 1);
    }
}
