//! Small concurrency helpers shared across the app.

use std::sync::mpsc;
use std::thread;
use std::time::Duration;

/// Runs `f` on a worker thread and returns its result, or `None` if it does not finish within
/// `timeout`.
///
/// On timeout the worker is **detached** — left to finish on its own and self-clean — and the caller
/// is unblocked regardless. Use this to bound an operation that can wedge on a flaky OS resource (a
/// stuck audio teardown, a hung capture daemon) so it can never hang the calling command forever.
/// The happy path costs one thread spawn and returns the instant `f` completes.
pub fn run_within<T, F>(timeout: Duration, f: F) -> Option<T>
where
    T: Send + 'static,
    F: FnOnce() -> T + Send + 'static,
{
    let (tx, rx) = mpsc::channel();

    thread::spawn(move || {
        let _ = tx.send(f());
    });

    rx.recv_timeout(timeout).ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;

    #[test]
    fn returns_the_result_when_work_finishes_in_time() {
        assert_eq!(run_within(Duration::from_secs(5), || 7), Some(7));
    }

    #[test]
    fn returns_none_when_work_exceeds_the_timeout() {
        let result = run_within(Duration::from_millis(40), || {
            thread::sleep(Duration::from_millis(500));
            7
        });

        assert_eq!(result, None);
    }

    #[test]
    fn detaches_the_worker_on_timeout_so_it_still_completes() {
        // The worker outlives the timed-out call; it must keep running, not be cancelled.
        let done = Arc::new(AtomicBool::new(false));
        let done_in_worker = Arc::clone(&done);

        let result = run_within(Duration::from_millis(40), move || {
            thread::sleep(Duration::from_millis(150));
            done_in_worker.store(true, Ordering::SeqCst);
        });
        assert_eq!(result, None);

        thread::sleep(Duration::from_millis(300));
        assert!(
            done.load(Ordering::SeqCst),
            "detached worker should have finished"
        );
    }
}
