//! Background session lifecycle around a [`Pipeline`].

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{self, Sender};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use wisp_audio::{Resampler16k, TARGET_SAMPLE_RATE};
use wisp_core::audio::AudioSource;
use wisp_core::denoise::Denoiser;
use wisp_core::engine::StreamingAsrEngine;
use wisp_core::error::{Result, WispError};
use wisp_core::transcript::{AudioSourceKind, SegmentStatus, TranscriptEvent, TranscriptSegment};

use crate::pipeline::Pipeline;
use crate::segmenter::{Segmenter, Utterance};
use crate::transcriber::Transcriber;

/// A boxed, `Send` consumer of transcript events (e.g. one that forwards them to the UI).
pub type EventSink = Box<dyn FnMut(TranscriptEvent) + Send>;

/// A unit of work handed from the (real-time) capture thread to the (slow) transcription thread.
#[derive(Debug)]
enum Job {
    /// A provisional decode of the still-open utterance, sharing the final's id so the UI updates
    /// one row in place. Dropped if a newer partial or the final supersedes it before it runs.
    Partial(u64, Utterance),
    /// The closed, authoritative utterance.
    Final(u64, Utterance),
}

/// Bounds on how much captured audio to accumulate between provisional partial decodes. The actual
/// interval *adapts to the engine's measured decode time*: a fast Mac gets snappier partials near
/// the floor, a slow one backs off toward the ceiling so it never builds a backlog (stale partials
/// are dropped regardless). [`PARTIAL_DEFAULT`] is used until the first decode is timed.
const PARTIAL_FLOOR: Duration = Duration::from_millis(300);
const PARTIAL_CEIL: Duration = Duration::from_millis(1_500);
const PARTIAL_DEFAULT: Duration = Duration::from_millis(500);

/// An utterance paired with the id its segment will carry — the unit a transcribe pass consumes.
type Decode = (u64, Utterance);

/// A transcription run executing on background thread(s).
///
/// The application's start/stop commands are thin wrappers over this — `start` →
/// [`Session::spawn_live`], `stop` → [`Session::stop`] — keeping the lifecycle logic off the UI
/// layer.
pub struct Session {
    stop: Arc<AtomicBool>,
    handles: Vec<JoinHandle<Result<()>>>,
}

impl Session {
    /// Spawns a synchronous [`Pipeline`] on one background thread, driving `source` and forwarding
    /// events to `sink` until the source ends or [`stop`](Self::stop) is called.
    ///
    /// Segmentation and (slow) transcription share this thread, so it's meant for finite sources
    /// (e.g. a file) where transcription latency doesn't stall live capture. Live microphone /
    /// system audio should use [`spawn_live`](Self::spawn_live).
    pub fn spawn(
        mut pipeline: Pipeline,
        mut source: Box<dyn AudioSource>,
        mut sink: EventSink,
    ) -> Self {
        let stop = Arc::new(AtomicBool::new(false));
        let stop_for_thread = Arc::clone(&stop);

        let handle =
            thread::spawn(move || pipeline.run_until(&mut *source, &mut *sink, &stop_for_thread));

        Self {
            stop,
            handles: vec![handle],
        }
    }

    /// Spawns a *decoupled* live session: a capture+segmentation thread feeds complete utterances
    /// over a queue to a separate transcription thread.
    ///
    /// The capture thread runs at real-time — it converts frames to 16 kHz mono, segments them
    /// with `segmenter`, and hands each finished [`Utterance`] to the queue — so a slow engine
    /// can never stall capture or cause dropped audio mid-sentence. The transcription thread
    /// drains the queue, runs `transcriber`, and forwards segments to `sink`. On stop the capture
    /// thread flushes its buffered utterance and closes the queue; the transcription thread then
    /// finishes the backlog before the session joins.
    /// `denoiser`, when present, cleans every captured frame before segmentation, so both the VAD
    /// and the engine see noise-suppressed audio. It is stateful and runs on the capture thread.
    pub fn spawn_live(
        mut segmenter: Box<dyn Segmenter>,
        mut transcriber: Transcriber,
        mut source: Box<dyn AudioSource>,
        mut sink: EventSink,
        denoiser: Option<Box<dyn Denoiser>>,
    ) -> Self {
        let stop = Arc::new(AtomicBool::new(false));
        let stop_for_capture = Arc::clone(&stop);
        let (job_tx, job_rx) = mpsc::channel::<Job>();

        // Most-recent partial-decode time (ms), shared so the capture thread can pace partials to
        // what this machine actually decodes — snappier on a fast Mac, backing off on a slow one.
        let decode_ms = Arc::new(AtomicU64::new(PARTIAL_DEFAULT.as_millis() as u64));
        let decode_ms_for_capture = Arc::clone(&decode_ms);

        // Transcription thread: drains jobs and forwards segments. Ends when the capture thread
        // drops its sender (stop or source exhausted). A transcription error on one job is logged
        // and skipped rather than killing the session. Each burst is coalesced first so the engine
        // never wastes work on a provisional partial that a newer partial or its final supersedes.
        let transcribe_handle = thread::spawn(move || {
            while let Ok(first) = job_rx.recv() {
                let mut batch = vec![first];
                while let Ok(next) = job_rx.try_recv() {
                    batch.push(next);
                }
                let (finals, partial) = coalesce(batch);

                for (id, utterance) in finals {
                    match transcriber.transcribe_utterance(id, &utterance, SegmentStatus::Final) {
                        Ok(Some(segment)) => sink(TranscriptEvent::Segment(segment)),
                        Ok(None) => {}
                        Err(e) => eprintln!("wisp: transcription error: {e}"),
                    }
                }
                if let Some((id, utterance)) = partial {
                    let started = Instant::now();
                    match transcriber.transcribe_utterance(id, &utterance, SegmentStatus::Partial) {
                        Ok(Some(segment)) => sink(TranscriptEvent::Segment(segment)),
                        Ok(None) => {}
                        Err(e) => eprintln!("wisp: partial transcription error: {e}"),
                    }
                    // Feed the measured decode cost back to the capture thread's pacing.
                    decode_ms.store(started.elapsed().as_millis() as u64, Ordering::Relaxed);
                }
            }
            Ok(())
        });

        // Capture + segmentation thread: never blocks on the engine.
        let capture_handle = thread::spawn(move || {
            run_capture(
                &mut *segmenter,
                &mut *source,
                denoiser,
                &job_tx,
                &decode_ms_for_capture,
                &stop_for_capture,
            )
            // `job_tx` drops here, signalling the transcription thread to finish.
        });

        Self {
            stop,
            handles: vec![capture_handle, transcribe_handle],
        }
    }

    /// Spawns a *streaming* live session: a single thread feeds captured audio chunk-by-chunk to an
    /// online `engine` that emits a growing hypothesis and finalizes on its own endpoint detection.
    ///
    /// Unlike [`spawn_live`](Self::spawn_live) — which VAD-segments utterances and transcribes each
    /// whole on a decoupled thread — a streaming engine decodes incrementally and cheaply per chunk,
    /// so capture and recognition share one thread without stalling. Each non-empty hypothesis is
    /// forwarded as a `Partial` segment the UI updates in place by id; on an endpoint it's sent
    /// `Final` and the next utterance gets a fresh id. `denoiser`, when present, cleans each chunk
    /// first. `kind` tags every segment with which source it came from.
    pub fn spawn_streaming(
        mut engine: Box<dyn StreamingAsrEngine>,
        mut source: Box<dyn AudioSource>,
        mut sink: EventSink,
        denoiser: Option<Box<dyn Denoiser>>,
        kind: AudioSourceKind,
    ) -> Self {
        let stop = Arc::new(AtomicBool::new(false));
        let stop_for_thread = Arc::clone(&stop);

        let handle = thread::spawn(move || {
            run_streaming(
                &mut *engine,
                &mut *source,
                denoiser,
                &mut sink,
                kind,
                &stop_for_thread,
            )
        });

        Self {
            stop,
            handles: vec![handle],
        }
    }

    /// Signals the session to stop and waits for its thread(s) to finish, returning the first
    /// error encountered. Use this for live sources (e.g. a microphone) that never end on their
    /// own.
    pub fn stop(mut self) -> Result<()> {
        self.stop.store(true, Ordering::Relaxed);
        self.take_join()
    }

    /// Waits for the session to finish on its own — e.g. a finite/file source reaching its end —
    /// without signalling stop, returning the first error encountered.
    pub fn join(mut self) -> Result<()> {
        self.take_join()
    }

    fn take_join(&mut self) -> Result<()> {
        let mut first_error = None;
        for handle in std::mem::take(&mut self.handles) {
            let result = handle
                .join()
                .unwrap_or_else(|_| Err(WispError::Engine("transcription thread panicked".into())));
            if let Err(e) = result {
                first_error.get_or_insert(e);
            }
        }
        match first_error {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }
}

/// Capture loop for [`Session::spawn_live`]: pull frames, segment them, and queue closed utterances
/// as `Final` jobs until `stop` is set or the source ends; then flush any buffered utterance.
///
/// Between finals, on an interval that adapts to the engine's measured decode speed (bounded by
/// [`PARTIAL_FLOOR`]/[`PARTIAL_CEIL`] via `decode_ms`), it also queues a `Partial` job of the
/// still-open utterance (when the segmenter exposes one), so the UI can show provisional text before
/// the speaker pauses. The capture thread owns utterance ids: a `Partial` and the `Final` that
/// closes the same utterance share one id, so the UI updates a single row in place.
fn run_capture(
    segmenter: &mut dyn Segmenter,
    source: &mut dyn AudioSource,
    mut denoiser: Option<Box<dyn Denoiser>>,
    job_tx: &Sender<Job>,
    decode_ms: &AtomicU64,
    stop: &AtomicBool,
) -> Result<()> {
    let mut next_id: u64 = 0;
    let mut open_id: Option<u64> = None;
    let mut since_partial = Duration::ZERO;
    let mut resampler = Resampler16k::new();

    while !stop.load(Ordering::Relaxed) {
        match source.next_frame()? {
            Some(frame) => {
                let mono = resampler.process(&frame);
                let mono = match &mut denoiser {
                    Some(d) => d.denoise(&mono, TARGET_SAMPLE_RATE),
                    None => mono,
                };

                for utterance in segmenter.push(&mono, frame.timestamp, frame.duration()) {
                    let id = open_id.take().unwrap_or_else(|| alloc_id(&mut next_id));
                    let _ = job_tx.send(Job::Final(id, utterance));
                    since_partial = Duration::ZERO;
                }

                since_partial += frame.duration();
                if since_partial >= partial_target(decode_ms) {
                    since_partial = Duration::ZERO;
                    if let Some(utterance) = segmenter.partial() {
                        let id = *open_id.get_or_insert_with(|| alloc_id(&mut next_id));
                        let _ = job_tx.send(Job::Partial(id, utterance));
                    }
                }
            }
            None => break,
        }
    }
    for utterance in segmenter.flush() {
        let id = open_id.take().unwrap_or_else(|| alloc_id(&mut next_id));
        let _ = job_tx.send(Job::Final(id, utterance));
    }
    Ok(())
}

/// Streaming capture+recognition loop for [`Session::spawn_streaming`]: pull frames, convert each to
/// 16 kHz mono (optionally denoised), feed it to `engine`, and forward the growing hypothesis as
/// `Partial`/`Final` segments until `stop` is set or the source ends.
///
/// All partials of one utterance share an id with the `Final` that closes it, so the UI updates a
/// single row in place; the engine's own endpoint detection commits the utterance and advances to a
/// fresh id and start time.
fn run_streaming(
    engine: &mut dyn StreamingAsrEngine,
    source: &mut dyn AudioSource,
    mut denoiser: Option<Box<dyn Denoiser>>,
    sink: &mut EventSink,
    kind: AudioSourceKind,
    stop: &AtomicBool,
) -> Result<()> {
    let mut id: u64 = 0;
    let mut utterance_start: Option<Duration> = None;
    let mut resampler = Resampler16k::new();

    while !stop.load(Ordering::Relaxed) {
        let Some(frame) = source.next_frame()? else {
            break;
        };
        let start = *utterance_start.get_or_insert(frame.timestamp);

        let mono = resampler.process(&frame);
        let mono = match &mut denoiser {
            Some(d) => d.denoise(&mono, TARGET_SAMPLE_RATE),
            None => mono,
        };

        let result = engine.accept_waveform(TARGET_SAMPLE_RATE, &mono);
        if !result.text.is_empty() {
            let mut segment = TranscriptSegment::new(
                id,
                &result.text,
                start..(frame.timestamp + frame.duration()),
                kind,
            );
            segment.status = if result.is_endpoint {
                SegmentStatus::Final
            } else {
                SegmentStatus::Partial
            };
            sink(TranscriptEvent::Segment(segment));
        }
        if result.is_endpoint {
            // A committed utterance (one that produced text) advances to a fresh id; an empty
            // endpoint just resets the start so the next utterance reuses this id.
            if !result.text.is_empty() {
                id += 1;
            }
            utterance_start = None;
        }
    }
    Ok(())
}

/// The current partial cadence: the engine's last measured partial-decode time, clamped to the sane
/// [`PARTIAL_FLOOR`]..=[`PARTIAL_CEIL`] bounds — so fast Macs emit partials more often and slow ones
/// back off, but neither runs away.
fn partial_target(decode_ms: &AtomicU64) -> Duration {
    let ms = decode_ms.load(Ordering::Relaxed).clamp(
        PARTIAL_FLOOR.as_millis() as u64,
        PARTIAL_CEIL.as_millis() as u64,
    );
    Duration::from_millis(ms)
}

/// Returns the current id and advances the counter.
fn alloc_id(next_id: &mut u64) -> u64 {
    let id = *next_id;
    *next_id += 1;
    id
}

/// Collapses a drained burst of jobs into the finals to transcribe (in order) and at most one
/// surviving partial — the newest one not followed by a final. A partial that a later final closes,
/// or that a newer partial replaces, is stale and dropped, so the engine only ever decodes the
/// latest provisional view of the open utterance.
fn coalesce(batch: Vec<Job>) -> (Vec<Decode>, Option<Decode>) {
    let mut finals = Vec::new();
    let mut partial = None;
    for job in batch {
        match job {
            Job::Final(id, utterance) => {
                partial = None;
                finals.push((id, utterance));
            }
            Job::Partial(id, utterance) => partial = Some((id, utterance)),
        }
    }
    (finals, partial)
}

impl Drop for Session {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        let _ = self.take_join();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::segmenter::EnergySegmenter;
    use crate::vad::EnergyVad;
    use std::sync::mpsc;
    use std::time::Duration;
    use wisp_core::audio::{AudioFrame, AudioSourceInfo};
    use wisp_core::engine::TranscriptionResult;
    use wisp_core::testing::{MockAsrEngine, MockAudioSource};
    use wisp_core::transcript::{AudioSourceKind, TranscriptSegment};

    fn frame(amp: f32, t_ms: u64) -> AudioFrame {
        AudioFrame::new(vec![amp; 1_600], 16_000, 1, Duration::from_millis(t_ms))
    }

    fn canned(text: &str) -> TranscriptionResult {
        TranscriptionResult {
            segments: vec![TranscriptSegment::new(
                0,
                text,
                Duration::ZERO..Duration::from_millis(50),
                AudioSourceKind::File,
            )],
        }
    }

    fn pipeline(results: Vec<TranscriptionResult>) -> Pipeline {
        Pipeline::new(
            Box::new(MockAsrEngine::new(results)),
            Box::new(EnergyVad::new(0.01)),
            AudioSourceKind::Microphone,
            Duration::from_millis(150),
        )
    }

    #[test]
    fn finite_source_delivers_events_then_finishes() {
        let frames = vec![
            frame(0.5, 0),
            frame(0.5, 100),
            frame(0.0, 200),
            frame(0.0, 300),
        ];
        let source: Box<dyn AudioSource> = Box::new(MockAudioSource::new(frames));

        let (tx, rx) = mpsc::channel();
        let sink: EventSink = Box::new(move |event| {
            let _ = tx.send(event);
        });

        let session = Session::spawn(pipeline(vec![canned("hi")]), source, sink);
        session.join().unwrap();

        let events: Vec<_> = rx.try_iter().collect();
        assert_eq!(events.len(), 1);
        match &events[0] {
            TranscriptEvent::Segment(s) => assert_eq!(s.text, "hi"),
            _ => panic!("unexpected event"),
        }
    }

    #[test]
    fn stop_halts_an_endless_source() {
        struct Endless;
        impl AudioSource for Endless {
            fn info(&self) -> AudioSourceInfo {
                AudioSourceInfo {
                    kind: AudioSourceKind::Microphone,
                    name: "endless".to_owned(),
                }
            }
            fn next_frame(&mut self) -> Result<Option<AudioFrame>> {
                Ok(Some(AudioFrame::new(
                    vec![0.0; 160],
                    16_000,
                    1,
                    Duration::ZERO,
                )))
            }
        }

        let (tx, _rx) = mpsc::channel();
        let sink: EventSink = Box::new(move |event| {
            let _ = tx.send(event);
        });

        // Endless yields silence forever; stop() must break the loop and join cleanly.
        let session = Session::spawn(pipeline(vec![]), Box::new(Endless), sink);
        session.stop().unwrap();
    }

    fn segmenter() -> Box<dyn Segmenter> {
        Box::new(EnergySegmenter::new(
            Box::new(EnergyVad::new(0.01)),
            Duration::from_millis(150),
        ))
    }

    fn transcriber(results: Vec<TranscriptionResult>) -> Transcriber {
        Transcriber::new(
            Box::new(MockAsrEngine::new(results)),
            AudioSourceKind::Microphone,
        )
    }

    #[test]
    fn spawn_live_transcribes_all_queued_utterances_in_order() {
        // Two utterances separated by trailing silence; the decoupled path must transcribe both,
        // in order, with sequential ids — even though the engine runs on a separate thread.
        let frames = vec![
            frame(0.5, 0),
            frame(0.5, 100),
            frame(0.0, 200),
            frame(0.0, 300), // closes utterance 1 (start 0)
            frame(0.5, 400),
            frame(0.0, 500),
            frame(0.0, 600), // closes utterance 2 (start 400)
        ];
        let source: Box<dyn AudioSource> = Box::new(MockAudioSource::new(frames));

        let (tx, rx) = mpsc::channel();
        let sink: EventSink = Box::new(move |event| {
            let _ = tx.send(event);
        });

        let session = Session::spawn_live(
            segmenter(),
            transcriber(vec![canned("one"), canned("two")]),
            source,
            sink,
            None,
        );
        session.join().unwrap();

        let segments: Vec<(u64, String)> = rx
            .try_iter()
            .map(|event| match event {
                TranscriptEvent::Segment(s) => (s.id, s.text),
                _ => panic!("unexpected event"),
            })
            .collect();
        assert_eq!(segments, vec![(0, "one".to_owned()), (1, "two".to_owned())]);
    }

    #[test]
    fn spawn_live_stop_halts_an_endless_source() {
        struct Endless;
        impl AudioSource for Endless {
            fn info(&self) -> AudioSourceInfo {
                AudioSourceInfo {
                    kind: AudioSourceKind::Microphone,
                    name: "endless".to_owned(),
                }
            }
            fn next_frame(&mut self) -> Result<Option<AudioFrame>> {
                Ok(Some(AudioFrame::new(
                    vec![0.0; 160],
                    16_000,
                    1,
                    Duration::ZERO,
                )))
            }
        }

        let (tx, _rx) = mpsc::channel();
        let sink: EventSink = Box::new(move |event| {
            let _ = tx.send(event);
        });

        // The capture thread must observe stop, close the queue, and let both threads join.
        let session = Session::spawn_live(
            segmenter(),
            transcriber(vec![]),
            Box::new(Endless),
            sink,
            None,
        );
        session.stop().unwrap();
    }

    #[test]
    fn spawn_live_denoises_each_captured_frame() {
        use std::sync::atomic::AtomicUsize;

        // Passes audio through unchanged while counting how many frames it cleaned.
        struct CountingDenoiser(Arc<AtomicUsize>);
        impl Denoiser for CountingDenoiser {
            fn denoise(&mut self, audio: &[f32], _sample_rate: u32) -> Vec<f32> {
                self.0.fetch_add(1, Ordering::Relaxed);
                audio.to_vec()
            }
        }

        let calls = Arc::new(AtomicUsize::new(0));
        let frames = vec![
            frame(0.5, 0),
            frame(0.5, 100),
            frame(0.0, 200),
            frame(0.0, 300),
        ];
        let source: Box<dyn AudioSource> = Box::new(MockAudioSource::new(frames));
        let (tx, _rx) = mpsc::channel();
        let sink: EventSink = Box::new(move |event| {
            let _ = tx.send(event);
        });

        let session = Session::spawn_live(
            segmenter(),
            transcriber(vec![canned("x")]),
            source,
            sink,
            Some(Box::new(CountingDenoiser(Arc::clone(&calls)))),
        );
        session.join().unwrap();

        assert_eq!(
            calls.load(Ordering::Relaxed),
            4,
            "the denoiser runs once per captured frame"
        );
    }

    #[test]
    fn spawn_streaming_shares_an_id_across_partials_then_finalizes_per_utterance() {
        use std::collections::VecDeque;
        use wisp_core::engine::{StreamingAsrEngine, StreamingResult};
        use wisp_core::transcript::SegmentStatus;

        // A scripted streaming engine: one result per `accept_waveform` call.
        struct Scripted(VecDeque<StreamingResult>);
        impl StreamingAsrEngine for Scripted {
            fn accept_waveform(&mut self, _rate: u32, _samples: &[f32]) -> StreamingResult {
                self.0.pop_front().unwrap_or(StreamingResult {
                    text: String::new(),
                    is_endpoint: false,
                })
            }
            fn reset(&mut self) {}
        }

        let frames = vec![
            frame(0.5, 0),
            frame(0.5, 100),
            frame(0.5, 200),
            frame(0.5, 300),
        ];
        let source: Box<dyn AudioSource> = Box::new(MockAudioSource::new(frames));
        let results = VecDeque::from(vec![
            StreamingResult {
                text: "he".to_owned(),
                is_endpoint: false,
            },
            StreamingResult {
                text: "hello".to_owned(),
                is_endpoint: false,
            },
            StreamingResult {
                text: "hello.".to_owned(),
                is_endpoint: true,
            },
            StreamingResult {
                text: "next".to_owned(),
                is_endpoint: false,
            },
        ]);

        let (tx, rx) = mpsc::channel();
        let sink: EventSink = Box::new(move |event| {
            let _ = tx.send(event);
        });

        let session = Session::spawn_streaming(
            Box::new(Scripted(results)),
            source,
            sink,
            None,
            AudioSourceKind::Microphone,
        );
        session.join().unwrap();

        let segs: Vec<(u64, String, SegmentStatus)> = rx
            .try_iter()
            .map(|e| match e {
                TranscriptEvent::Segment(s) => (s.id, s.text, s.status),
                _ => panic!("unexpected event"),
            })
            .collect();
        assert_eq!(
            segs,
            vec![
                (0, "he".to_owned(), SegmentStatus::Partial),
                (0, "hello".to_owned(), SegmentStatus::Partial),
                (0, "hello.".to_owned(), SegmentStatus::Final),
                (1, "next".to_owned(), SegmentStatus::Partial),
            ],
            "partials share the utterance id, the endpoint finalizes it, and the next utterance gets a fresh id"
        );
    }

    fn utt(len: usize) -> Utterance {
        Utterance {
            audio: vec![0.1; len],
            start: Duration::ZERO,
        }
    }

    #[test]
    fn coalesce_keeps_only_the_newest_partial() {
        let (finals, partial) = coalesce(vec![Job::Partial(0, utt(1)), Job::Partial(0, utt(2))]);
        assert!(finals.is_empty());
        assert_eq!(
            partial.unwrap().1.audio.len(),
            2,
            "the older partial is dropped"
        );
    }

    #[test]
    fn coalesce_drops_a_partial_superseded_by_its_final() {
        let (finals, partial) = coalesce(vec![Job::Partial(0, utt(1)), Job::Final(0, utt(3))]);
        assert_eq!(finals.len(), 1);
        assert_eq!(finals[0].0, 0);
        assert!(
            partial.is_none(),
            "the final supersedes the provisional partial"
        );
    }

    #[test]
    fn coalesce_keeps_a_partial_that_follows_the_last_final() {
        let (finals, partial) = coalesce(vec![Job::Final(0, utt(3)), Job::Partial(1, utt(1))]);
        assert_eq!(finals.len(), 1);
        assert_eq!(
            partial.unwrap().0,
            1,
            "the next utterance's partial survives"
        );
    }

    #[test]
    fn coalesce_preserves_final_order() {
        let (finals, partial) = coalesce(vec![Job::Final(0, utt(1)), Job::Final(1, utt(2))]);
        let ids: Vec<u64> = finals.iter().map(|(id, _)| *id).collect();
        assert_eq!(ids, vec![0, 1]);
        assert!(partial.is_none());
    }

    #[test]
    fn capture_emits_a_partial_then_a_final_sharing_one_id() {
        // Never closes on push, always exposes an in-progress partial, closes on flush (stream end)
        // — so we can observe the partial cadence and the shared id deterministically.
        struct OpenSegmenter {
            partial_len: usize,
            final_len: usize,
        }
        impl Segmenter for OpenSegmenter {
            fn push(&mut self, _m: &[f32], _t: Duration, _d: Duration) -> Vec<Utterance> {
                Vec::new()
            }
            fn flush(&mut self) -> Vec<Utterance> {
                vec![Utterance {
                    audio: vec![0.3; self.final_len],
                    start: Duration::ZERO,
                }]
            }
            fn partial(&self) -> Option<Utterance> {
                Some(Utterance {
                    audio: vec![0.2; self.partial_len],
                    start: Duration::ZERO,
                })
            }
        }

        // Seven 100 ms frames cross the 600 ms partial interval exactly once before the source ends.
        let frames: Vec<AudioFrame> = (0u64..7).map(|i| frame(0.5, i * 100)).collect();
        let mut source = MockAudioSource::new(frames);
        let mut seg = OpenSegmenter {
            partial_len: 800,
            final_len: 1_600,
        };

        let (tx, rx) = mpsc::channel();
        let stop = AtomicBool::new(false);
        // No transcribe thread here, so the decode time stays at the default (500 ms cadence).
        let decode_ms = AtomicU64::new(PARTIAL_DEFAULT.as_millis() as u64);
        run_capture(&mut seg, &mut source, None, &tx, &decode_ms, &stop).unwrap();
        drop(tx);

        let jobs: Vec<Job> = rx.try_iter().collect();
        assert_eq!(
            jobs.len(),
            2,
            "one partial (one interval) then the closing final"
        );
        match &jobs[0] {
            Job::Partial(id, u) => {
                assert_eq!(*id, 0);
                assert_eq!(u.audio.len(), 800);
            }
            other => panic!("expected a partial first, got {other:?}"),
        }
        match &jobs[1] {
            Job::Final(id, u) => {
                assert_eq!(*id, 0, "the final reuses the open partial's id");
                assert_eq!(u.audio.len(), 1_600);
            }
            other => panic!("expected a final second, got {other:?}"),
        }
    }

    #[test]
    fn partial_target_clamps_decode_time_to_bounds() {
        // A fast machine (decode under the floor) is held at the floor — no partial spam.
        assert_eq!(partial_target(&AtomicU64::new(50)), PARTIAL_FLOOR);
        // A typical decode passes through unchanged.
        assert_eq!(
            partial_target(&AtomicU64::new(420)),
            Duration::from_millis(420)
        );
        // A slow machine (decode over the ceiling) is capped so it still tries periodically.
        assert_eq!(partial_target(&AtomicU64::new(9_000)), PARTIAL_CEIL);
    }
}
