//! A bounded audio-frame channel that drops the oldest frame when full.
//!
//! Capture callbacks produce frames in real time, but the transcription consumer can briefly
//! fall behind (an engine pass is synchronous). An unbounded queue would then grow without
//! limit — latency and memory climb until the stream appears stuck. This channel caps the
//! backlog and drops the **oldest** frame on overflow, so capture stays real-time (dropping
//! stale audio instead of stalling). [`FrameSender::close`] wakes a blocked receiver so a
//! stopped or errored source ends cleanly rather than hanging.

use std::collections::VecDeque;
use std::sync::{Arc, Condvar, Mutex};

use crate::audio::AudioFrame;

struct Shared {
    state: Mutex<State>,
    available: Condvar,
    capacity: usize,
}

struct State {
    frames: VecDeque<AudioFrame>,
    closed: bool,
}

/// Sending half of a [`frame_channel`]. Cheap to clone; `Send + Sync`.
#[derive(Clone)]
pub struct FrameSender {
    shared: Arc<Shared>,
}

/// Receiving half of a [`frame_channel`].
pub struct FrameReceiver {
    shared: Arc<Shared>,
}

/// Creates a bounded frame channel holding at most `capacity` frames (drop-oldest on overflow).
pub fn frame_channel(capacity: usize) -> (FrameSender, FrameReceiver) {
    let shared = Arc::new(Shared {
        state: Mutex::new(State {
            frames: VecDeque::new(),
            closed: false,
        }),
        available: Condvar::new(),
        capacity: capacity.max(1),
    });
    (
        FrameSender {
            shared: Arc::clone(&shared),
        },
        FrameReceiver { shared },
    )
}

impl FrameSender {
    /// Pushes a frame, dropping the oldest if the buffer is full. A no-op once closed.
    pub fn send(&self, frame: AudioFrame) {
        let mut state = self.shared.state.lock().expect("frame channel poisoned");
        if state.closed {
            return;
        }
        while state.frames.len() >= self.shared.capacity {
            state.frames.pop_front();
        }
        state.frames.push_back(frame);
        self.shared.available.notify_one();
    }

    /// Closes the channel and wakes a blocked receiver (which drains, then returns `None`).
    pub fn close(&self) {
        let mut state = self.shared.state.lock().expect("frame channel poisoned");
        state.closed = true;
        self.shared.available.notify_all();
    }
}

impl FrameReceiver {
    /// Blocks until a frame is available; returns `None` once the channel is closed and drained.
    pub fn recv(&self) -> Option<AudioFrame> {
        let mut state = self.shared.state.lock().expect("frame channel poisoned");
        loop {
            if let Some(frame) = state.frames.pop_front() {
                return Some(frame);
            }
            if state.closed {
                return None;
            }
            state = self
                .shared
                .available
                .wait(state)
                .expect("frame channel poisoned");
        }
    }

    /// Returns a frame if one is immediately available, otherwise `None`; never blocks.
    ///
    /// Unlike [`recv`](Self::recv), a `None` does not distinguish "empty" from "closed and
    /// drained" — this drains whatever is buffered right now without waiting. Used to pull the
    /// latest reference audio without stalling the caller.
    pub fn try_recv(&self) -> Option<AudioFrame> {
        self.shared
            .state
            .lock()
            .expect("frame channel poisoned")
            .frames
            .pop_front()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn frame(value: f32) -> AudioFrame {
        AudioFrame::new(vec![value], 16_000, 1, Duration::ZERO)
    }

    #[test]
    fn delivers_in_order() {
        let (tx, rx) = frame_channel(8);
        tx.send(frame(1.0));
        tx.send(frame(2.0));
        assert_eq!(rx.recv().unwrap().samples, vec![1.0]);
        assert_eq!(rx.recv().unwrap().samples, vec![2.0]);
    }

    #[test]
    fn drops_oldest_when_full() {
        let (tx, rx) = frame_channel(2);
        tx.send(frame(1.0));
        tx.send(frame(2.0));
        tx.send(frame(3.0)); // buffer full → drops the oldest (1.0)
        assert_eq!(rx.recv().unwrap().samples, vec![2.0]);
        assert_eq!(rx.recv().unwrap().samples, vec![3.0]);
    }

    #[test]
    fn close_returns_none_when_drained() {
        let (tx, rx) = frame_channel(4);
        tx.send(frame(1.0));
        tx.close();
        assert_eq!(rx.recv().unwrap().samples, vec![1.0]); // drains remaining
        assert!(rx.recv().is_none()); // then None
    }

    #[test]
    fn try_recv_drains_without_blocking() {
        let (tx, rx) = frame_channel(4);
        assert!(rx.try_recv().is_none()); // empty → None, does not block
        tx.send(frame(1.0));
        tx.send(frame(2.0));
        assert_eq!(rx.try_recv().unwrap().samples, vec![1.0]);
        assert_eq!(rx.try_recv().unwrap().samples, vec![2.0]);
        assert!(rx.try_recv().is_none()); // drained → None again
    }
}
