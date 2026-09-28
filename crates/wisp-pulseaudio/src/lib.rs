//! Linux system-audio capture via the PulseAudio monitor source.
//!
//! Records whatever is playing on the default output — the sink's `.monitor` source — and exposes
//! it as a [`wisp_core::AudioSource`]: the Linux counterpart of the macOS ScreenCaptureKit source
//! and the Windows WASAPI loopback. One click, no virtual device, no setup — on PulseAudio and on
//! PipeWire's `pipewire-pulse` server (the default on modern desktops).
//!
//! The PulseAudio objects live entirely on a dedicated capture thread that feeds frames to a
//! channel; the public handle holds only the receiver + a stop flag and stays `Send` (the same
//! split the Windows WASAPI source uses for its non-`Send` COM objects).

#![cfg(target_os = "linux")]

use std::cell::RefCell;
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Sender};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use libpulse_binding::context::{Context, FlagSet as ContextFlags, State as ContextState};
use libpulse_binding::def::BufferAttr;
use libpulse_binding::mainloop::threaded::Mainloop;
use libpulse_binding::operation::State as OperationState;
use libpulse_binding::sample::{Format, Spec};
use libpulse_binding::stream::{FlagSet as StreamFlags, PeekResult, State as StreamState, Stream};

use wisp_core::audio::{AudioFrame, AudioSource, AudioSourceInfo};
use wisp_core::channel::{frame_channel, FrameReceiver, FrameSender};
use wisp_core::error::{Result, WispError};
use wisp_core::transcript::AudioSourceKind;

/// Bounded capacity of the capture→pipeline frame channel. Drop-oldest on overflow keeps capture
/// real-time if the consumer briefly stalls (same policy as the mic and loopback sources).
const FRAME_CHANNEL_CAPACITY: usize = 1024;

/// The stream's sample spec — the pipeline's native input (16 kHz mono f32), so the server does
/// any resample/downmix once, server-side, and capture hands the engines their exact format.
const SAMPLE_SPEC: Spec = Spec {
    format: Format::F32le,
    rate: 16_000,
    channels: 1,
};
const BYTES_PER_SAMPLE: usize = 4;
const CHANNELS: u16 = SAMPLE_SPEC.channels as u16;

/// How often the capture thread drains the record buffer. Well under the server fragment size, so
/// monitor audio never overflows while keeping latency low.
const POLL: Duration = Duration::from_millis(10);

/// How long `new` waits for the context + stream to come up before reporting unavailable — covers
/// a slow `pipewire-pulse` autospawn without hanging Start forever.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(5);

/// An [`AudioSource`] capturing the default output's monitor through PulseAudio/PipeWire.
pub struct PulseMonitorSource {
    rx: FrameReceiver,
    stop: Arc<AtomicBool>,
    handle: Option<JoinHandle<()>>,
}

impl PulseMonitorSource {
    /// Starts capturing the default sink's monitor. Errors if the sound server can't be reached or
    /// has no output device, so the caller can degrade to mic-only.
    pub fn new() -> Result<Self> {
        let (tx, rx) = frame_channel(FRAME_CHANNEL_CAPACITY);
        let (ready_tx, ready_rx) = mpsc::channel::<std::result::Result<(), String>>();
        let stop = Arc::new(AtomicBool::new(false));
        let stop_for_thread = Arc::clone(&stop);

        let handle = thread::spawn(move || capture_thread(&tx, &stop_for_thread, &ready_tx));

        match ready_rx.recv() {
            Ok(Ok(())) => Ok(Self {
                rx,
                stop,
                handle: Some(handle),
            }),
            // The thread already returned after signalling failure — reap it so the error path
            // doesn't leave a zombie.
            Ok(Err(e)) => {
                let _ = handle.join();
                Err(WispError::Audio(e))
            }
            Err(_) => Err(WispError::Audio(
                "PulseAudio capture thread exited before signalling readiness".to_owned(),
            )),
        }
    }
}

impl AudioSource for PulseMonitorSource {
    fn info(&self) -> AudioSourceInfo {
        AudioSourceInfo {
            kind: AudioSourceKind::System,
            name: "System audio".to_owned(),
        }
    }

    fn next_frame(&mut self) -> Result<Option<AudioFrame>> {
        Ok(self.rx.recv())
    }
}

impl Drop for PulseMonitorSource {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

/// Runs the capture, signalling readiness to `new` once the stream is up, then closes the channel
/// so a blocked receiver ends cleanly when capture stops or fails.
fn capture_thread(
    tx: &FrameSender,
    stop: &AtomicBool,
    ready: &Sender<std::result::Result<(), String>>,
) {
    if let Err(e) = run_capture(tx, stop, ready) {
        // Only reached for a failure *before* readiness was signalled; tell `new` to degrade.
        let _ = ready.send(Err(e));
    }
    tx.close();
}

/// Releases the PulseAudio objects in order: stream, context, mainloop. The context is
/// disconnected under the mainloop lock and the loop stopped unlocked, per libpulse's contract.
struct ServerGuard {
    mainloop: Rc<RefCell<Mainloop>>,
    context: Rc<RefCell<Context>>,
    stream: Option<Rc<RefCell<Stream>>>,
}

impl Drop for ServerGuard {
    fn drop(&mut self) {
        self.mainloop.borrow_mut().lock();
        if let Some(stream) = &self.stream {
            let _ = stream.borrow_mut().disconnect();
        }
        self.context.borrow_mut().disconnect();
        self.mainloop.borrow_mut().unlock();
        self.mainloop.borrow_mut().stop();
    }
}

/// Builds the PulseAudio connection and record stream on this thread, then drains the monitor
/// into `tx` until `stop`. Returns `Err` only for a setup failure before `ready` is signalled.
fn run_capture(
    tx: &FrameSender,
    stop: &AtomicBool,
    ready: &Sender<std::result::Result<(), String>>,
) -> std::result::Result<(), String> {
    let mainloop = Rc::new(RefCell::new(
        Mainloop::new().ok_or("create PulseAudio mainloop")?,
    ));
    let context = Rc::new(RefCell::new(
        Context::new(&*mainloop.borrow(), "Wisp").ok_or("create PulseAudio context")?,
    ));
    context
        .borrow_mut()
        .connect(None, ContextFlags::NOFLAGS, None)
        .map_err(|e| format!("connect to sound server: {e}"))?;
    mainloop
        .borrow_mut()
        .start()
        .map_err(|e| format!("start PulseAudio mainloop: {e}"))?;

    let mut guard = ServerGuard {
        mainloop: Rc::clone(&mainloop),
        context: Rc::clone(&context),
        stream: None,
    };

    wait_for("context", CONNECT_TIMEOUT, || {
        mainloop.borrow_mut().lock();
        let state = context.borrow().get_state();
        mainloop.borrow_mut().unlock();
        match state {
            ContextState::Ready => PollState::Ready,
            ContextState::Failed | ContextState::Terminated => PollState::Failed,
            _ => PollState::Pending,
        }
    })?;

    let monitor = default_sink_monitor(&mainloop, &context)?;

    // pa_stream_new touches the context's object list — lock while the mainloop thread is live.
    mainloop.borrow_mut().lock();
    let stream = Stream::new(
        &mut context.borrow_mut(),
        "meeting audio",
        &SAMPLE_SPEC,
        None,
    );
    mainloop.borrow_mut().unlock();
    let stream = Rc::new(RefCell::new(stream.ok_or("create record stream")?));
    guard.stream = Some(Rc::clone(&stream));

    let attr = buffer_attr();
    mainloop.borrow_mut().lock();
    let connected = stream.borrow_mut().connect_record(
        Some(&monitor),
        Some(&attr),
        StreamFlags::ADJUST_LATENCY,
    );
    mainloop.borrow_mut().unlock();
    connected.map_err(|e| format!("record the monitor source '{monitor}': {e}"))?;

    wait_for("record stream", CONNECT_TIMEOUT, || {
        mainloop.borrow_mut().lock();
        let state = stream.borrow().get_state();
        mainloop.borrow_mut().unlock();
        match state {
            StreamState::Ready => PollState::Ready,
            StreamState::Failed | StreamState::Terminated => PollState::Failed,
            _ => PollState::Pending,
        }
    })?;

    let _ = ready.send(Ok(()));
    let start = Instant::now();

    while !stop.load(Ordering::Relaxed) {
        let mut produced: Vec<f32> = Vec::new();

        mainloop.borrow_mut().lock();
        let mut stream_mut = stream.borrow_mut();
        loop {
            match stream_mut.readable_size() {
                Some(0) | None => break,
                Some(_) => match stream_mut.peek() {
                    Ok(PeekResult::Data(bytes)) => {
                        produced.extend(
                            bytes
                                .as_chunks::<BYTES_PER_SAMPLE>()
                                .0
                                .iter()
                                .map(|c| f32::from_le_bytes(*c)),
                        );
                        let _ = stream_mut.discard();
                    }
                    // A hole is a reported gap in the stream — skip it; the pipeline handles gaps.
                    Ok(PeekResult::Hole(_)) => {
                        let _ = stream_mut.discard();
                    }
                    Ok(PeekResult::Empty) | Err(_) => break,
                },
            }
        }
        drop(stream_mut);
        mainloop.borrow_mut().unlock();

        if !produced.is_empty() {
            tx.send(AudioFrame::new(
                produced,
                SAMPLE_SPEC.rate,
                CHANNELS,
                start.elapsed(),
            ));
        }

        thread::sleep(POLL);
    }

    Ok(())
}

/// Outcome of one poll of an object's state.
enum PollState {
    Ready,
    Pending,
    Failed,
}

/// Polls `state` until it reports ready or failed, or `timeout` elapses. Polling (rather than a
/// state callback + `mainloop.wait`) keeps the wait logic identical for the context, the stream,
/// and the introspection operation, and can't deadlock on a missed signal.
fn wait_for(
    what: &str,
    timeout: Duration,
    mut state: impl FnMut() -> PollState,
) -> std::result::Result<(), String> {
    let deadline = Instant::now() + timeout;
    loop {
        match state() {
            PollState::Ready => return Ok(()),
            PollState::Failed => return Err(format!("{what} failed to start")),
            PollState::Pending if Instant::now() >= deadline => {
                return Err(format!("{what} took over {timeout:?} to start"))
            }
            PollState::Pending => thread::sleep(POLL),
        }
    }
}

/// The record stream's buffering: ~40 ms fragments inside a ~1 s ceiling — small enough to keep
/// transcription latency invisible, large enough that a momentary engine stall never overflows.
fn buffer_attr() -> BufferAttr {
    let bytes_per_sec =
        SAMPLE_SPEC.rate as usize * SAMPLE_SPEC.channels as usize * BYTES_PER_SAMPLE;
    BufferAttr {
        maxlength: bytes_per_sec as u32,       // ~1 s
        tlength: u32::MAX,                     // playback-only field — let the server pick
        prebuf: u32::MAX,                      // playback-only field
        minreq: u32::MAX,                      // playback-only field
        fragsize: (bytes_per_sec / 25) as u32, // 40 ms fragments
    }
}

/// The name of the source monitoring the default output — `<default sink>.monitor` — resolved via
/// the server-info introspection op. Errors when the server has no output device to monitor.
fn default_sink_monitor(
    mainloop: &Rc<RefCell<Mainloop>>,
    context: &Rc<RefCell<Context>>,
) -> std::result::Result<String, String> {
    let sink_name: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));

    mainloop.borrow_mut().lock();
    let op = context.borrow().introspect().get_server_info({
        let sink_name = Arc::clone(&sink_name);
        move |info| {
            if let Some(name) = info.default_sink_name.as_deref() {
                *sink_name.lock().expect("sink name poisoned") = Some(name.to_owned());
            }
        }
    });
    mainloop.borrow_mut().unlock();

    wait_for("server-info query", CONNECT_TIMEOUT, || {
        mainloop.borrow_mut().lock();
        let state = op.get_state();
        mainloop.borrow_mut().unlock();
        match state {
            OperationState::Done => PollState::Ready,
            OperationState::Cancelled => PollState::Failed,
            _ => PollState::Pending,
        }
    })?;

    let sink_name = sink_name.lock().expect("sink name poisoned").clone();
    sink_name
        .map(|name| format!("{name}.monitor"))
        .ok_or_else(|| "sound server reports no default output device".to_owned())
}
