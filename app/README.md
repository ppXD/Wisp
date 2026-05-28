# Wisp — desktop app

Tauri v2 + SvelteKit shell for Wisp: local, real-time, on-device meeting transcription.

## Run (development)

```sh
npm install
npm run tauri dev
```

Press **Start**, allow the microphone, and speak. Transcription runs on-device via the
sherpa-onnx **SenseVoice** engine (multilingual: Chinese, English, Japanese, Korean, Cantonese).

## Model

The engine loads the SenseVoice model from the app data directory:

```
~/Library/Application Support/com.wisp.desktop/models/sense-voice/
  ├── model.int8.onnx
  └── tokens.txt
```

Download it once (~1 GB):

```sh
DEST="$HOME/Library/Application Support/com.wisp.desktop/models/sense-voice"
mkdir -p "$DEST" && cd "$(mktemp -d)"
curl -fL -o m.tar.bz2 \
  https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2
tar xjf m.tar.bz2
cp sherpa-onnx-sense-voice-*/model.int8.onnx sherpa-onnx-sense-voice-*/tokens.txt "$DEST"/
```

If the model is missing, **Start** returns a clear error naming the expected path.

## Architecture

The `src-tauri` shell is thin: `start_session` / `stop_session` dispatch to
`wisp_pipeline::Session`, which drives `MicSource` → VAD → `SenseVoiceEngine` and streams
`transcript://segment` events to this UI. Swapping the engine (e.g. to Whisper or Parakeet) is
a one-line change behind `wisp_core::AsrEngine`.
