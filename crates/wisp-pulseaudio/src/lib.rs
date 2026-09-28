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

/// While the monitor produces nothing (suspended sink, silent desktop), still emit an empty
/// heartbeat frame at this cadence so a blocked `next_frame` wakes, the session's stop flag gets
/// observed, and teardown doesn't fall back to the teardown-timeout detach — which would leak the
/// live PulseAudio record stream for the rest of the process. `frame_count() == 0` frames carry
/// no audio, so downstream treats them as a tick rather than silence samples.
const HEARTBEAT: Duration = Duration::from_millis(250);

/// How long `new` waits for the context + stream to come up before reporting unavailable — covers
/// a slow `pipewire-pulse` autospawn without hanging Start forever.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(5);

/// Outer bound on `new` — covers the three CONNECT_TIMEOUT waits plus slack. Mirrors the macOS
/// source's STARTUP_TIMEOUT: if libpulse wedges before the first state poll, Start must still
/// return (the caller degrades to mic-only) rather than block the UI forever.
const READY_TIMEOUT: Duration = Duration::from_secs(20);

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

        match ready_rx.recv_timeout(READY_TIMEOUT) {
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
            Err(mpsc::RecvTimeoutError::Timeout) => {
                // Wedged inside libpulse before the first state wait — set the stop flag so the
                // thread exits if it ever unwedges, then detach it (dropping the JoinHandle) and
                // report unavailable. Joining here could block forever, which is the failure
                // mode this bound exists to prevent.
                stop.store(true, Ordering::Relaxed);
                Err(WispError::Audio(
                    "timed out starting PulseAudio capture".to_owned(),
                ))
            }
            Err(mpsc::RecvTimeoutError::Disconnected) => Err(WispError::Audio(
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
    // catch_unwind so a panic inside run_capture still reaches `tx.close()` — otherwise every
    // consumer blocked in `recv` hangs until the pipeline's teardown timeout detaches it.
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        run_capture(tx, stop, ready)
    }));
    match outcome {
        // Only reached for a failure *before* readiness was signalled; tell `new` to degrade.
        Ok(Err(e)) => {
            let _ = ready.send(Err(e));
        }
        Err(_) => {
            eprintln!("wisp: PulseAudio capture thread panicked; ending system audio");
            // If the panic happened before readiness, `new` still needs a verdict.
            let _ = ready.send(Err("capture thread panicked".to_owned()));
        }
        Ok(Ok(())) => {}
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
    // DONT_MOVE: if the monitored source disappears, a server running module-rescue-streams would
    // otherwise rebind this record stream to an arbitrary source — potentially a physical
    // microphone captured under the "System audio" label. Terminating instead matches the other
    // backends' device-loss contract (clean EOF, logged).
    let connected = stream.borrow_mut().connect_record(
        Some(&monitor),
        Some(&attr),
        StreamFlags::ADJUST_LATENCY | StreamFlags::DONT_MOVE,
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
    let mut last_emit = Instant::now();

    while !stop.load(Ordering::Relaxed) {
        let mut produced: Vec<f32> = Vec::new();

        mainloop.borrow_mut().lock();
        // A server restart or stream failure mid-capture ends system audio here — the same
        // contract the WASAPI source has (log, close the channel, let the app degrade).
        let dead = matches!(
            context.borrow().get_state(),
            ContextState::Failed | ContextState::Terminated
        ) || matches!(
            stream.borrow().get_state(),
            StreamState::Failed | StreamState::Terminated
        );
        if dead {
            mainloop.borrow_mut().unlock();
            eprintln!("wisp: PulseAudio capture ended (server or stream went away)");
            break;
        }
        let mut stream_mut = stream.borrow_mut();
        loop {
            match stream_mut.readable_size() {
                Some(0) | None => break,
                Some(_) => match stream_mut.peek() {
                    Ok(PeekResult::Data(bytes)) => {
                        produced.extend(decode_f32le(bytes));
                        if stream_mut.discard().is_err() {
                            break; // don't re-peek the same fragment in a lock-held spin
                        }
                    }
                    // A hole is a reported gap in the stream — skip it; the pipeline handles gaps.
                    Ok(PeekResult::Hole(_)) => {
                        if stream_mut.discard().is_err() {
                            break;
                        }
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
            last_emit = Instant::now();
        } else if last_emit.elapsed() >= HEARTBEAT {
            // Empty keepalive so a blocked receiver can notice stop/close during silence.
            tx.send(AudioFrame::new(
                Vec::new(),
                SAMPLE_SPEC.rate,
                CHANNELS,
                start.elapsed(),
            ));
            last_emit = Instant::now();
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

/// Decodes one peeked fragment — little-endian f32, 4 bytes per sample. A trailing partial
/// sample can't occur (PA fragments are frame-aligned); `as_chunks` drops it if it ever did.
fn decode_f32le(bytes: &[u8]) -> impl Iterator<Item = f32> + '_ {
    bytes
        .as_chunks::<BYTES_PER_SAMPLE>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
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
/// Resolved once at capture start: if the default sink changes mid-meeting (headphones plugged
/// in), the stream keeps recording the original sink's monitor — the same bound-at-init behaviour
/// the WASAPI backend has.
fn default_sink_monitor(
    mainloop: &Rc<RefCell<Mainloop>>,
    context: &Rc<RefCell<Context>>,
) -> std::result::Result<String, String> {
    let sink_name: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));

    mainloop.borrow_mut().lock();
    // pa_context_get_server_info returns NULL on a non-Ready context and the binding asserts on
    // it — the server can die in the gap between the state wait and this call, so re-check under
    // the same lock (the mainloop can't transition the state while we hold it).
    if context.borrow().get_state() != ContextState::Ready {
        mainloop.borrow_mut().unlock();
        return Err("sound server went away before the sink query".to_owned());
    }
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
    monitor_source_name(sink_name.as_deref())
}

/// `<default sink>.monitor` for a server-reported default sink name; an absent or empty default
/// means the host has no output device to capture — the caller degrades to mic-only.
fn monitor_source_name(default_sink: Option<&str>) -> std::result::Result<String, String> {
    default_sink
        .filter(|name| !name.is_empty())
        .map(|name| format!("{name}.monitor"))
        .ok_or_else(|| "sound server reports no default output device".to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn monitor_source_name_appends_monitor() {
        assert_eq!(
            monitor_source_name(Some("alsa_output.pci-0000_00_1f.3.analog-stereo")).unwrap(),
            "alsa_output.pci-0000_00_1f.3.analog-stereo.monitor"
        );
    }

    #[test]
    fn monitor_source_name_rejects_absent_or_empty_default() {
        assert!(monitor_source_name(None).is_err());
        assert!(monitor_source_name(Some("")).is_err());
    }

    #[test]
    fn buffer_attr_is_low_latency_with_headroom() {
        let attr = buffer_attr();
        let bytes_per_sec =
            SAMPLE_SPEC.rate as usize * SAMPLE_SPEC.channels as usize * BYTES_PER_SAMPLE;
        assert_eq!(attr.fragsize, (bytes_per_sec / 25) as u32); // ~40 ms fragments
        assert_eq!(attr.maxlength, bytes_per_sec as u32); // ~1 s ceiling
                                                          // Playback-only fields stay at the "unset" sentinel for a record stream.
        assert_eq!(attr.tlength, u32::MAX);
        assert_eq!(attr.prebuf, u32::MAX);
        assert_eq!(attr.minreq, u32::MAX);
    }

    #[test]
    fn decode_f32le_round_trips_and_drops_partial_tail() {
        let samples = [0.0f32, 0.5, -1.0, f32::MAX];
        let mut bytes = Vec::new();
        for s in samples {
            bytes.extend_from_slice(&s.to_le_bytes());
        }
        bytes.extend_from_slice(&[0xAA, 0xBB]); // unaligned tail — PA never emits one
        let decoded: Vec<f32> = decode_f32le(&bytes).collect();
        assert_eq!(decoded, samples);
    }

    #[test]
    fn wait_for_returns_on_ready_and_failed() {
        assert!(wait_for("test", Duration::from_millis(50), || PollState::Ready).is_ok());
        assert!(wait_for("test", Duration::from_millis(50), || PollState::Failed).is_err());
    }

    #[test]
    fn wait_for_times_out_on_pending() {
        let start = Instant::now();
        let result = wait_for("test", Duration::from_millis(30), || PollState::Pending);
        assert!(result.is_err());
        assert!(start.elapsed() >= Duration::from_millis(30));
    }

    #[test]
    fn wait_for_observes_late_ready() {
        let mut calls = 0;
        let result = wait_for("test", CONNECT_TIMEOUT, || {
            calls += 1;
            if calls > 3 {
                PollState::Ready
            } else {
                PollState::Pending
            }
        });
        assert!(result.is_ok());
    }

    #[test]
    fn constants_match_pipeline_expectations() {
        // The pipeline's native input: 16 kHz mono f32 (the server resamples for us).
        assert_eq!(SAMPLE_SPEC.rate, 16_000);
        assert_eq!(SAMPLE_SPEC.channels, 1);
        assert_eq!(SAMPLE_SPEC.format, Format::F32le);
        assert_eq!(BYTES_PER_SAMPLE, 4);
        // Same capacity/policy as the mic, WASAPI, and ScreenCaptureKit sources.
        assert_eq!(FRAME_CHANNEL_CAPACITY, 1024);
        assert!(POLL < HEARTBEAT);
        assert!(READY_TIMEOUT > CONNECT_TIMEOUT * 3);
    }
}
