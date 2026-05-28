# Wisp

Local, real-time, cross-platform meeting transcription — on-device, private, model-swappable.

Wisp captures your microphone (and, in later phases, system/meeting audio) and turns speech into
a live, timestamped, per-speaker transcript entirely on your machine. No cloud, no upload.

> **Status:** early development, built in small PRs toward an MVP.
> See [`CLAUDE.md`](./CLAUDE.md) for architecture and contribution conventions.

## Why Wisp

- **Local & private** — audio never leaves the device.
- **Real-time** — live partial → finalized transcript with timestamps.
- **Model-swappable** — install/choose different models (Whisper, Parakeet, …) from a catalog.
- **Generic & pluggable** — audio sources, ASR engines, and diarizers sit behind narrow traits,
  so new capabilities (system-audio capture, speaker diarization, echo cancellation, alternative
  engines) slot in without breaking changes.

## Stack

- **Shell:** Tauri v2 (Rust core + web UI)
- **Engine:** sherpa-onnx via `sherpa-rs` (VAD + ASR + diarization, no Python); Whisper / Parakeet models
- **Audio:** `cpal` (mic) + per-OS loopback (WASAPI / PipeWire / ScreenCaptureKit); WebRTC AEC3 for echo
- All inference runs on-device; models are pulled from Hugging Face on demand.

## Development

Requires Rust (stable) and Node.

```sh
cargo test --all --all-features                       # run the test suite
cargo clippy --all-targets --all-features -- -D warnings   # lint
cargo fmt --all -- --check                            # formatting
```

GUI run instructions land with the Tauri app crate.

## License

[MIT](./LICENSE)
