# Wisp — Project Guide for Claude

Local, real-time, cross-platform meeting transcription. On-device, private, model-swappable.
Built in small PRs toward an MVP.

## What Wisp is

Captures the microphone (and, later, system/meeting audio), runs on-device ASR, and shows a
live, timestamped, per-speaker transcript. No cloud.

**MVP success criteria**

1. User picks/downloads a transcription model from a catalog (the "install different model" core).
2. App captures the microphone and shows live partial → final transcript with timestamps.
3. Audio sources, ASR engines, diarizers, and model stores all sit behind narrow traits, so
   system-audio capture, speaker diarization, echo cancellation, and alternative engines
   (Parakeet/WhisperKit) are added later as new trait impls with **zero breaking changes**.

**Post-MVP phases** (architecture already accommodates them): system/loopback capture
(WASAPI / PipeWire / ScreenCaptureKit) → unified dual-stream timeline → echo cancellation
(WebRTC AEC3 — the loopback stream is the far-end reference) → per-speaker diarization
(windowed offline sherpa-onnx + incremental clustering) → export (SRT/VTT) → optional LLM cleanup.

## Locked stack (all Rust, no Python)

- **Shell:** Tauri v2 (Rust core + web UI).
- **Engine:** sherpa-onnx via `sherpa-rs` (VAD + ASR + diarization in one C++ lib). Models:
  Whisper large-v3-turbo (default) + Parakeet (low-latency streaming); pluggable.
- **Audio:** `cpal` (mic) + per-OS loopback; `webrtc-audio-processing` (AEC3) for echo.
- **Diarization:** pyannote-seg-3.0 + 3D-Speaker ONNX (redistributed by k2-fsa; no HF token).

`whisper-rs` (tazz4843) is archived — do **not** depend on it. Use sherpa-rs / direct FFI /
Tauri sidecar.

## Repo layout

- `crates/wisp-core` — pure Rust, **no native deps**: traits, domain types, pipeline logic,
  errors, config. The testable heart; most logic lives here.
- `crates/wisp-engines` — ASR engine impls (sherpa-rs adapter) behind core traits (feature-gated).
- `crates/wisp-audio` — audio source impls (mic, per-OS loopback) behind core traits.
- `crates/wisp-app` (`src-tauri`) — thin Tauri shell; wires core + engines + audio to the UI.
- `ui/` — web frontend.

Crates are added incrementally; only what exists today is committed.

## Architecture rules (load-bearing)

1. **Generic & pluggable (ISP).** Capabilities sit behind narrow traits: `AudioSource`,
   `AsrEngine`, `Diarizer`, `ModelStore`. A trait must make sense for every implementor — start
   narrow; widening is cheap, narrowing breaks every impl.
2. **No breaking changes.** Never change a trait / public signature without searching all usages,
   updating every impl + caller, and getting a full `cargo build` / `cargo test` green. New
   behavior = new impl or a new method with a default — not a changed signature.
3. **Logic in the core, shells stay thin** (mirrors the global guide's Rule 16). Tauri command
   handlers and background tasks are dispatchers; real work lives in `wisp-core` services that
   take inputs and return outputs (functional, no hidden state beyond a single pipeline run).
4. **Functions are the unit of readability.** Short, named, single-responsibility. The pipeline
   reads as a flat sequence of named calls.

## Testing (test-first, three tiers — non-negotiable)

Red → Green → Refactor. Define cases first, write failing tests, minimal code to green, refactor.

- **Unit:** pure logic — trait contracts (via mock impls), domain types, model-descriptor
  parsing, pipeline state transitions, VAD/segmentation and diarization clustering with
  **synthetic** audio/embeddings.
- **Integration:** real-but-isolated — model registry against temp dirs + a mock HTTP server,
  checksum verification (incl. the mismatch path), engine adapter against a tiny fixture WAV.
- **E2E:** full pipeline against real hardware (mic/GUI). **Skip-guarded** (`if no audio device
  { return }`) so `cargo test` is clean on headless/CI machines. Run these on a real Mac.

Carry-overs from the global guide:

- **Env-var overrides** (model dir, download base URL, feature flags) live behind an env var
  whose name is a `pub const`, **pinned by a test** asserting the literal string (Rule 8).
- **Mirrored logic needs a drift-detector test** (Rule 12.5). Prefer driving the real production
  type over a mock when that path is under test (Rule 12.4).
- Allocate test dirs/ports uniquely; clean up every artifact (Rule 12.2 / 12.3).

**Verification reality:** in a headless agent environment only unit + integration tests run.
Anything needing a microphone, a display, or a multi-GB model download is hardware-gated and
must be run on the user's Mac. State which tier a test is when you write it.

## Dev workflow — small PRs

One concern per PR. Branch → implement + tests → `cargo fmt --all` → `cargo clippy --all-targets
--all-features -- -D warnings` → `cargo test --all` (all green locally) → push → open PR → CI
green → squash merge.

CI (`.github/workflows/ci.yml`): fmt + clippy on ubuntu, tests on ubuntu + macos. Native-dep
crates add their per-OS system libs to CI when introduced (e.g. `libasound2-dev` for `cpal`).

## PR & commit conventions (global Rule 13 & 15 — strict)

- **Imperative title**, < 72 chars, no trailing period, **no** bracketed `[Phase]` / `[PRn]`
  progress markers. Body = `## Summary` (the why) + `## Test plan` (checkable). Self-contained.
- **Never** add an AI / `Co-Authored-By: Claude` trailer. **Never** pass `--author` or modify git
  config. Commit as the repo owner. Reference issues as `#N`.
- Commits: imperative subject, body explains the why, not the what.
