//! The built-in model catalog Wisp offers out of the box.
//!
//! Checksums are currently unpinned (empty) — downloads are size-known and the store warns when
//! fetching an unpinned file. Pinning real SHA-256 values is a follow-up.

use wisp_core::model::{ModelDescriptor, ModelFamily, ModelFile, ModelId, Quant};

/// Hugging Face mirror hosting individual SenseVoice files (no auth needed).
const SENSE_VOICE_BASE: &str =
    "https://huggingface.co/csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/resolve/main";

/// Hugging Face repos hosting the sherpa-onnx Whisper ONNX exports (no auth needed).
const WHISPER_LARGE_V3_BASE: &str =
    "https://huggingface.co/csukuangfj/sherpa-onnx-whisper-large-v3/resolve/main";
const WHISPER_MEDIUM_BASE: &str =
    "https://huggingface.co/csukuangfj/sherpa-onnx-whisper-medium/resolve/main";

/// Hugging Face repo hosting the streaming (online) Zipformer transducer, bilingual zh+en (no auth).
const STREAMING_ZIPFORMER_BASE: &str =
    "https://huggingface.co/csukuangfj/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/resolve/main";

/// Hugging Face repo hosting the sherpa-onnx FunASR Paraformer (offline, zh+en) ONNX export (no auth).
const PARAFORMER_ZH_BASE: &str =
    "https://huggingface.co/csukuangfj/sherpa-onnx-paraformer-zh-2024-03-09/resolve/main";

/// Hugging Face repo hosting the sherpa-onnx NVIDIA NeMo Parakeet TDT (offline, English) export (no auth).
const PARAKEET_BASE: &str =
    "https://huggingface.co/csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8/resolve/main";

/// Hugging Face repo hosting the whisper.cpp GGUF models (no auth needed). Also hosts the optional
/// Core ML encoder archives (see [`crate::coreml`]).
pub(crate) const WHISPER_CPP_BASE: &str =
    "https://huggingface.co/ggerganov/whisper.cpp/resolve/main";

/// Hugging Face repos hosting the sherpa-onnx diarization models (no auth needed): a pyannote
/// speaker-segmentation model plus interchangeable speaker-embedding models.
const PYANNOTE_SEG_BASE: &str =
    "https://huggingface.co/csukuangfj/sherpa-onnx-pyannote-segmentation-3-0/resolve/main";
const SPEAKER_EMB_BASE: &str =
    "https://huggingface.co/csukuangfj/speaker-embedding-models/resolve/main";

/// GitHub release hosting the sherpa-onnx speech-enhancement (denoiser) models (no auth needed).
const GTCRN_BASE: &str =
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/speech-enhancement-models";

/// All models Wisp offers in the picker.
pub fn builtin_catalog() -> Vec<ModelDescriptor> {
    vec![
        whisper_turbo_q8(),
        whisper_turbo_q5(),
        whisper_turbo_full(),
        whisper_large_v3_gpu(),
        sense_voice_int8(),
        sense_voice_fp32(),
        paraformer_zh(),
        parakeet_en(),
        streaming_zipformer_zh_en(),
        apple_speech(),
        whisper_large_v3(),
        whisper_medium(),
        whisper_tiny(),
        whisper_base(),
        whisper_small(),
        whisper_medium_gpu(),
    ]
}

/// FunASR Paraformer (offline, Chinese + English) via sherpa-onnx — non-autoregressive, fast on the
/// CPU, with timestamps. A strong Mandarin-focused alternative to SenseVoice. int8 (~0.22 GB).
fn paraformer_zh() -> ModelDescriptor {
    ModelDescriptor {
        id: ModelId("paraformer-zh".to_owned()),
        family: ModelFamily::Paraformer,
        quant: Quant::Q8,
        display_name: "Paraformer · zh + en · int8 (fast)".to_owned(),
        files: vec![
            ModelFile {
                name: "model.int8.onnx".to_owned(),
                url: format!("{PARAFORMER_ZH_BASE}/model.int8.onnx"),
                sha256: String::new(),
                size_bytes: 227_330_205,
            },
            ModelFile {
                name: "tokens.txt".to_owned(),
                url: format!("{PARAFORMER_ZH_BASE}/tokens.txt"),
                sha256: String::new(),
                size_bytes: 75_354,
            },
        ],
        languages: vec!["zh".to_owned(), "en".to_owned()],
        description:
            "FunASR Paraformer (Chinese + English) via sherpa-onnx — a non-autoregressive recognizer \
             that runs fast on the CPU and emits timestamps. Often stronger on Mandarin than \
             SenseVoice. Offline (not for live captions). int8 (~0.22 GB)."
                .to_owned(),
    }
}

/// NVIDIA NeMo Parakeet TDT (offline, English) via sherpa-onnx — among the most accurate open English
/// recognizers. An encoder/decoder/joiner transducer, run on the CPU. int8 (~0.63 GB).
fn parakeet_en() -> ModelDescriptor {
    ModelDescriptor {
        id: ModelId("parakeet-en".to_owned()),
        family: ModelFamily::Parakeet,
        quant: Quant::Q8,
        display_name: "Parakeet · English · int8 (accurate)".to_owned(),
        files: vec![
            ModelFile {
                name: "encoder.int8.onnx".to_owned(),
                url: format!("{PARAKEET_BASE}/encoder.int8.onnx"),
                sha256: String::new(),
                size_bytes: 652_184_296,
            },
            ModelFile {
                name: "decoder.int8.onnx".to_owned(),
                url: format!("{PARAKEET_BASE}/decoder.int8.onnx"),
                sha256: String::new(),
                size_bytes: 7_257_753,
            },
            ModelFile {
                name: "joiner.int8.onnx".to_owned(),
                url: format!("{PARAKEET_BASE}/joiner.int8.onnx"),
                sha256: String::new(),
                size_bytes: 1_739_080,
            },
            ModelFile {
                name: "tokens.txt".to_owned(),
                url: format!("{PARAKEET_BASE}/tokens.txt"),
                sha256: String::new(),
                size_bytes: 9_384,
            },
        ],
        languages: vec!["en".to_owned()],
        description:
            "NVIDIA NeMo Parakeet TDT 0.6B (English only) via sherpa-onnx — among the most accurate \
             open English recognizers, run on the CPU. Offline; English audio only. int8 (~0.63 GB)."
                .to_owned(),
    }
}

/// whisper.cpp tiny (GPU/Metal) — the smallest, fastest Whisper size; lowest accuracy. q5 (~32 MB).
fn whisper_tiny() -> ModelDescriptor {
    ModelDescriptor {
        id: ModelId("whisper-tiny".to_owned()),
        family: ModelFamily::WhisperCpp,
        quant: Quant::Q5,
        display_name: "Whisper tiny · GPU (Metal) · q5".to_owned(),
        files: vec![ModelFile {
            name: "ggml-tiny-q5_1.bin".to_owned(),
            url: format!("{WHISPER_CPP_BASE}/ggml-tiny-q5_1.bin"),
            sha256: String::new(),
            size_bytes: 32_152_673,
        }],
        languages: whisper_languages(),
        description:
            "Whisper tiny on the GPU (Metal) — the smallest, fastest size. ~99 languages but markedly \
             lower accuracy; best for quick drafts or very low-resource machines. q5 (~32 MB)."
                .to_owned(),
    }
}

/// whisper.cpp base (GPU/Metal) — a small, fast size; modest accuracy. q5 (~60 MB).
fn whisper_base() -> ModelDescriptor {
    ModelDescriptor {
        id: ModelId("whisper-base".to_owned()),
        family: ModelFamily::WhisperCpp,
        quant: Quant::Q5,
        display_name: "Whisper base · GPU (Metal) · q5".to_owned(),
        files: vec![ModelFile {
            name: "ggml-base-q5_1.bin".to_owned(),
            url: format!("{WHISPER_CPP_BASE}/ggml-base-q5_1.bin"),
            sha256: String::new(),
            size_bytes: 59_707_625,
        }],
        languages: whisper_languages(),
        description:
            "Whisper base on the GPU (Metal) — small and fast, with modest accuracy. ~99 languages. \
             A step up from tiny for light workloads. q5 (~60 MB)."
                .to_owned(),
    }
}

/// whisper.cpp small (GPU/Metal) — a balanced size; good accuracy at low cost. q5 (~190 MB).
fn whisper_small() -> ModelDescriptor {
    ModelDescriptor {
        id: ModelId("whisper-small".to_owned()),
        family: ModelFamily::WhisperCpp,
        quant: Quant::Q5,
        display_name: "Whisper small · GPU (Metal) · q5".to_owned(),
        files: vec![ModelFile {
            name: "ggml-small-q5_1.bin".to_owned(),
            url: format!("{WHISPER_CPP_BASE}/ggml-small-q5_1.bin"),
            sha256: String::new(),
            size_bytes: 190_085_487,
        }],
        languages: whisper_languages(),
        description:
            "Whisper small on the GPU (Metal) — a good balance of speed and accuracy at a low memory \
             cost. ~99 languages. q5 (~190 MB)."
                .to_owned(),
    }
}

/// whisper.cpp medium (GPU/Metal) — high accuracy, heavier than small, lighter than large. q5 (~539 MB).
fn whisper_medium_gpu() -> ModelDescriptor {
    ModelDescriptor {
        id: ModelId("whisper-medium-gpu".to_owned()),
        family: ModelFamily::WhisperCpp,
        quant: Quant::Q5,
        display_name: "Whisper medium · GPU (Metal) · q5".to_owned(),
        files: vec![ModelFile {
            name: "ggml-medium-q5_0.bin".to_owned(),
            url: format!("{WHISPER_CPP_BASE}/ggml-medium-q5_0.bin"),
            sha256: String::new(),
            size_bytes: 539_212_467,
        }],
        languages: whisper_languages(),
        description:
            "Whisper medium on the GPU (Metal) — high accuracy across ~99 languages, between small and \
             large-v3. A strong middle option when large-v3 is heavier than you need. q5 (~539 MB)."
                .to_owned(),
    }
}

/// Apple's on-device `SpeechAnalyzer` / `SpeechTranscriber` (macOS 26+). Downloads nothing of ours —
/// the OS owns the recogniser and fetches the per-language asset itself on first use — so it has no
/// files. Hidden off macOS by `family_runnable`; the app gates it further on the macOS-26 runtime check.
fn apple_speech() -> ModelDescriptor {
    ModelDescriptor {
        id: ModelId("apple-speech".to_owned()),
        family: ModelFamily::AppleSpeech,
        quant: Quant::Other("os".to_owned()),
        display_name: "Apple on-device speech · macOS 26 · streaming".to_owned(),
        files: Vec::new(),
        languages: vec![
            "yue".to_owned(),
            "zh".to_owned(),
            "en".to_owned(),
            "ja".to_owned(),
            "ko".to_owned(),
        ],
        description:
            "Apple's built-in on-device recogniser (SpeechAnalyzer, macOS 26+) — zero download, very \
             low power, and fully private. Streams words as you speak with the newest system models \
             (a step beyond older Apple dictation). The OS fetches each language pack on first use."
                .to_owned(),
    }
}

/// Languages every Whisper model covers well (a subset surfaced in the UI).
fn whisper_languages() -> Vec<String> {
    ["yue", "zh", "en", "ja"]
        .iter()
        .map(|s| (*s).to_owned())
        .collect()
}

fn whisper_turbo_q5() -> ModelDescriptor {
    ModelDescriptor {
        id: ModelId("whisper-large-v3-turbo-q5".to_owned()),
        family: ModelFamily::WhisperCpp,
        quant: Quant::Q5,
        display_name: "Whisper large-v3-turbo · GPU (Metal) · q5".to_owned(),
        files: vec![ModelFile {
            name: "ggml-large-v3-turbo-q5_0.bin".to_owned(),
            url: format!("{WHISPER_CPP_BASE}/ggml-large-v3-turbo-q5_0.bin"),
            sha256: String::new(),
            size_bytes: 574_041_195,
        }],
        languages: whisper_languages(),
        description:
            "Whisper large-v3-turbo on the GPU (Metal) — real Cantonese (yue) + ~99 languages. A \
             lighter, faster alternative to the q8 default, at slightly lower precision. \
             q5-quantized (~0.55 GB)."
                .to_owned(),
    }
}

fn whisper_turbo_q8() -> ModelDescriptor {
    ModelDescriptor {
        id: ModelId("whisper-large-v3-turbo-q8".to_owned()),
        family: ModelFamily::WhisperCpp,
        quant: Quant::Q8,
        display_name: "Whisper large-v3-turbo · GPU (Metal) · q8".to_owned(),
        files: vec![ModelFile {
            name: "ggml-large-v3-turbo-q8_0.bin".to_owned(),
            url: format!("{WHISPER_CPP_BASE}/ggml-large-v3-turbo-q8_0.bin"),
            sha256: String::new(),
            size_bytes: 874_188_075,
        }],
        languages: whisper_languages(),
        description:
            "Whisper large-v3-turbo on the GPU (Metal) at q8 precision — real Cantonese (yue) + \
             ~99 languages, very accurate yet fast because it runs on the GPU. The recommended \
             default. ~0.85 GB."
                .to_owned(),
    }
}

fn whisper_large_v3_gpu() -> ModelDescriptor {
    ModelDescriptor {
        id: ModelId("whisper-large-v3-gpu".to_owned()),
        family: ModelFamily::WhisperCpp,
        quant: Quant::Q5,
        display_name: "Whisper large-v3 · GPU (Metal) · q5 (most accurate)".to_owned(),
        files: vec![ModelFile {
            name: "ggml-large-v3-q5_0.bin".to_owned(),
            url: format!("{WHISPER_CPP_BASE}/ggml-large-v3-q5_0.bin"),
            sha256: String::new(),
            size_bytes: 1_081_140_203,
        }],
        languages: whisper_languages(),
        description:
            "The full Whisper large-v3 (32-layer decoder, not distilled) on the GPU (Metal) — the \
             most accurate option, best for files with the Accurate mode. Slower than turbo; \
             ~1.1 GB."
                .to_owned(),
    }
}

fn whisper_turbo_full() -> ModelDescriptor {
    ModelDescriptor {
        id: ModelId("whisper-large-v3-turbo".to_owned()),
        family: ModelFamily::WhisperCpp,
        quant: Quant::F16,
        display_name: "Whisper large-v3-turbo · GPU (Metal) · full".to_owned(),
        files: vec![ModelFile {
            name: "ggml-large-v3-turbo.bin".to_owned(),
            // Declared size is a safe under-estimate (the store rejects a download shorter than this).
            url: format!("{WHISPER_CPP_BASE}/ggml-large-v3-turbo.bin"),
            sha256: String::new(),
            size_bytes: 1_550_000_000,
        }],
        languages: whisper_languages(),
        description:
            "Full-precision large-v3-turbo on the GPU (Metal) — the most accurate turbo, a touch \
             slower and larger than q8. Real Cantonese (yue) + ~99 languages. ~1.6 GB."
                .to_owned(),
    }
}

fn sense_voice_languages() -> Vec<String> {
    ["zh", "en", "ja", "ko", "yue"]
        .iter()
        .map(|s| (*s).to_owned())
        .collect()
}

fn sense_voice_tokens() -> ModelFile {
    ModelFile {
        name: "tokens.txt".to_owned(),
        url: format!("{SENSE_VOICE_BASE}/tokens.txt"),
        sha256: String::new(),
        size_bytes: 315_894,
    }
}

fn sense_voice_int8() -> ModelDescriptor {
    ModelDescriptor {
        id: ModelId("sense-voice".to_owned()),
        family: ModelFamily::SenseVoice,
        quant: Quant::Q8,
        display_name: "SenseVoice · multilingual · int8 (fast)".to_owned(),
        files: vec![
            ModelFile {
                name: "model.int8.onnx".to_owned(),
                url: format!("{SENSE_VOICE_BASE}/model.int8.onnx"),
                sha256: String::new(),
                size_bytes: 239_233_841,
            },
            sense_voice_tokens(),
        ],
        languages: sense_voice_languages(),
        description:
            "Small multilingual model (~234M params), int8-quantized. Fast and light on memory, \
             especially strong for Chinese, Cantonese, Japanese, and Korean."
                .to_owned(),
    }
}

fn sense_voice_fp32() -> ModelDescriptor {
    ModelDescriptor {
        id: ModelId("sense-voice-fp32".to_owned()),
        family: ModelFamily::SenseVoice,
        quant: Quant::F32,
        display_name: "SenseVoice · multilingual · fp32 (most accurate)".to_owned(),
        files: vec![
            ModelFile {
                name: "model.onnx".to_owned(),
                url: format!("{SENSE_VOICE_BASE}/model.onnx"),
                sha256: String::new(),
                size_bytes: 937_617_178,
            },
            sense_voice_tokens(),
        ],
        languages: sense_voice_languages(),
        description:
            "Small multilingual model (~234M params), full precision. Highest accuracy, but a \
             much larger download and a little slower than int8."
                .to_owned(),
    }
}

fn whisper_large_v3() -> ModelDescriptor {
    ModelDescriptor {
        id: ModelId("whisper-large-v3".to_owned()),
        family: ModelFamily::Whisper,
        quant: Quant::Q8,
        display_name: "Whisper large-v3 · 99+ languages · int8".to_owned(),
        files: vec![
            ModelFile {
                name: "large-v3-encoder.int8.onnx".to_owned(),
                url: format!("{WHISPER_LARGE_V3_BASE}/large-v3-encoder.int8.onnx"),
                sha256: String::new(),
                size_bytes: 766_671_985,
            },
            ModelFile {
                name: "large-v3-decoder.int8.onnx".to_owned(),
                url: format!("{WHISPER_LARGE_V3_BASE}/large-v3-decoder.int8.onnx"),
                sha256: String::new(),
                size_bytes: 1_008_265_203,
            },
            ModelFile {
                name: "large-v3-tokens.txt".to_owned(),
                url: format!("{WHISPER_LARGE_V3_BASE}/large-v3-tokens.txt"),
                sha256: String::new(),
                size_bytes: 816_730,
            },
        ],
        languages: vec![
            "yue".to_owned(),
            "zh".to_owned(),
            "en".to_owned(),
            "ja".to_owned(),
        ],
        description:
            "OpenAI Whisper large-v3 (int8) — the most accurate, broadest option, ~99 languages \
             including Cantonese (yue). Large (~1.8 GB) and noticeably slower than SenseVoice."
                .to_owned(),
    }
}

fn whisper_medium() -> ModelDescriptor {
    ModelDescriptor {
        id: ModelId("whisper-medium".to_owned()),
        family: ModelFamily::Whisper,
        quant: Quant::Q8,
        display_name: "Whisper medium · 99+ languages · int8".to_owned(),
        files: vec![
            ModelFile {
                name: "medium-encoder.int8.onnx".to_owned(),
                url: format!("{WHISPER_MEDIUM_BASE}/medium-encoder.int8.onnx"),
                sha256: String::new(),
                size_bytes: 374_196_283,
            },
            ModelFile {
                name: "medium-decoder.int8.onnx".to_owned(),
                url: format!("{WHISPER_MEDIUM_BASE}/medium-decoder.int8.onnx"),
                sha256: String::new(),
                size_bytes: 571_059_257,
            },
            ModelFile {
                name: "medium-tokens.txt".to_owned(),
                url: format!("{WHISPER_MEDIUM_BASE}/medium-tokens.txt"),
                sha256: String::new(),
                size_bytes: 816_730,
            },
        ],
        languages: vec![
            "yue".to_owned(),
            "zh".to_owned(),
            "en".to_owned(),
            "ja".to_owned(),
        ],
        description:
            "OpenAI Whisper medium (int8) — a middle ground: ~99 languages, smaller and faster \
             than large-v3 with somewhat lower accuracy (~0.95 GB)."
                .to_owned(),
    }
}

/// The streaming (online) Zipformer transducer — emits words as you speak instead of waiting for a
/// pause, so live transcription feels realtime. CPU-friendly on every platform.
fn streaming_zipformer_zh_en() -> ModelDescriptor {
    ModelDescriptor {
        id: ModelId("streaming-zipformer-zh-en".to_owned()),
        family: ModelFamily::StreamingTransducer,
        quant: Quant::Q8,
        display_name: "Streaming Zipformer · zh + en · low latency".to_owned(),
        files: vec![
            ModelFile {
                name: "encoder-epoch-99-avg-1.int8.onnx".to_owned(),
                url: format!("{STREAMING_ZIPFORMER_BASE}/encoder-epoch-99-avg-1.int8.onnx"),
                sha256: String::new(),
                size_bytes: 181_895_032,
            },
            ModelFile {
                name: "decoder-epoch-99-avg-1.int8.onnx".to_owned(),
                url: format!("{STREAMING_ZIPFORMER_BASE}/decoder-epoch-99-avg-1.int8.onnx"),
                sha256: String::new(),
                size_bytes: 13_091_040,
            },
            ModelFile {
                name: "joiner-epoch-99-avg-1.int8.onnx".to_owned(),
                url: format!("{STREAMING_ZIPFORMER_BASE}/joiner-epoch-99-avg-1.int8.onnx"),
                sha256: String::new(),
                size_bytes: 3_228_404,
            },
            ModelFile {
                name: "tokens.txt".to_owned(),
                url: format!("{STREAMING_ZIPFORMER_BASE}/tokens.txt"),
                sha256: String::new(),
                size_bytes: 56_317,
            },
        ],
        languages: vec!["zh".to_owned(), "en".to_owned()],
        description:
            "Streaming Zipformer transducer (Chinese + English) — emits words as you speak \
             (~200-300 ms latency) and runs in real time on the CPU. Lower accuracy than Whisper \
             but far snappier; best for live captions (~0.2 GB)."
                .to_owned(),
    }
}

/// Speaker-diarization models Wisp offers (segmentation + embedding). Kept separate from
/// [`builtin_catalog`] because they are not ASR engines and never appear in the model picker; the
/// File mode downloads one only when "Identify speakers" is turned on.
pub fn diarization_models() -> Vec<ModelDescriptor> {
    vec![diarization_standard(), diarization_accurate()]
}

/// The shared pyannote speaker-segmentation model (finds *when* speech happens and overlaps).
/// Stored under a fixed name so the engine can load it regardless of the embedding choice.
fn pyannote_segmentation() -> ModelFile {
    ModelFile {
        name: "segmentation.onnx".to_owned(),
        url: format!("{PYANNOTE_SEG_BASE}/model.onnx"),
        sha256: String::new(),
        size_bytes: 5_992_913,
    }
}

fn diarization_standard() -> ModelDescriptor {
    ModelDescriptor {
        id: ModelId("diarize-campplus".to_owned()),
        family: ModelFamily::Diarization,
        quant: Quant::F32,
        display_name: "Speaker ID · CAM++ · standard (fast)".to_owned(),
        files: vec![
            pyannote_segmentation(),
            ModelFile {
                name: "embedding.onnx".to_owned(),
                url: format!(
                    "{SPEAKER_EMB_BASE}/3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx"
                ),
                sha256: String::new(),
                size_bytes: 28_281_138,
            },
        ],
        languages: Vec::new(),
        description:
            "Identifies who speaks when, using CAM++ voice embeddings with pyannote segmentation. \
             Fast and light (~33 MB) with great everyday accuracy — the recommended default."
                .to_owned(),
    }
}

fn diarization_accurate() -> ModelDescriptor {
    ModelDescriptor {
        id: ModelId("diarize-eres2net-large".to_owned()),
        family: ModelFamily::Diarization,
        quant: Quant::F32,
        display_name: "Speaker ID · ERes2Net-large · most accurate".to_owned(),
        files: vec![
            pyannote_segmentation(),
            ModelFile {
                name: "embedding.onnx".to_owned(),
                url: format!(
                    "{SPEAKER_EMB_BASE}/3dspeaker_speech_eres2net_large_sv_zh-cn_3dspeaker_16k.onnx"
                ),
                sha256: String::new(),
                size_bytes: 116_058_710,
            },
        ],
        languages: Vec::new(),
        description:
            "Highest speaker-separation accuracy, using the larger ERes2Net voice embeddings. \
             Bigger download (~122 MB) and a little slower; best when speakers sound alike."
                .to_owned(),
    }
}

/// Speech-denoiser models Wisp offers (downloadable, on top of the built-in RNNoise). Kept separate
/// from [`builtin_catalog`] because they are not ASR engines and never appear in the model picker;
/// File mode downloads one only when the matching "Reduce noise" strength is chosen.
pub fn denoise_models() -> Vec<ModelDescriptor> {
    vec![denoise_gtcrn()]
}

fn denoise_gtcrn() -> ModelDescriptor {
    ModelDescriptor {
        id: ModelId("denoise-gtcrn".to_owned()),
        family: ModelFamily::Denoise,
        quant: Quant::F32,
        display_name: "Noise reduction · GTCRN · balanced".to_owned(),
        files: vec![ModelFile {
            name: "gtcrn_simple.onnx".to_owned(),
            url: format!("{GTCRN_BASE}/gtcrn_simple.onnx"),
            sha256: String::new(),
            size_bytes: 535_638,
        }],
        languages: Vec::new(),
        description:
            "GTCRN neural denoiser — removes real-world background noise (traffic, café, room tone), \
             not just steady hum. Tiny (~0.5 MB) and 16 kHz-native; the balanced choice between the \
             light built-in RNNoise and the strongest models."
                .to_owned(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn catalog_has_distinct_ids_and_files() {
        let catalog = builtin_catalog();
        assert_eq!(catalog.len(), 16);

        let ids: std::collections::HashSet<_> = catalog.iter().map(|d| &d.id).collect();
        assert_eq!(ids.len(), catalog.len(), "model ids must be distinct");

        for descriptor in &catalog {
            // sherpa families ship ONNX + tokens; whisper.cpp ships a single GGUF `.bin`; Apple
            // on-device speech is OS-provided and ships nothing of ours.
            match descriptor.family {
                ModelFamily::AppleSpeech => {
                    assert!(descriptor.files.is_empty(), "OS-provided: no files of ours");
                    assert_eq!(descriptor.total_size_bytes(), 0, "nothing to download");
                }
                ModelFamily::WhisperCpp => {
                    assert!(descriptor.files.iter().any(|f| f.name.ends_with(".bin")));
                    assert!(descriptor.total_size_bytes() > 0);
                }
                _ => {
                    assert!(descriptor.files.iter().any(|f| f.name.ends_with(".onnx")));
                    assert!(descriptor
                        .files
                        .iter()
                        .any(|f| f.name.ends_with("tokens.txt")));
                    assert!(descriptor.total_size_bytes() > 0);
                }
            }
            assert!(!descriptor.description.is_empty());
        }
    }

    #[test]
    fn default_model_is_the_accuracy_first_turbo_q8() {
        // New installs default to the first catalog entry. Pin it: turbo-q8 is the accuracy-first
        // pick that's still real-time on the GPU (large-v3 is more accurate but too slow to default
        // for live). A reorder should be a conscious, reviewed change.
        assert_eq!(
            builtin_catalog()[0].id,
            ModelId("whisper-large-v3-turbo-q8".to_owned())
        );
    }

    #[test]
    fn diarization_models_are_segmentation_plus_embedding_pairs() {
        let models = diarization_models();
        assert_eq!(models.len(), 2);

        let ids: std::collections::HashSet<_> = models.iter().map(|d| &d.id).collect();
        assert_eq!(ids.len(), models.len(), "diarization ids must be distinct");

        for descriptor in &models {
            assert_eq!(descriptor.family, ModelFamily::Diarization);
            assert!(descriptor
                .files
                .iter()
                .any(|f| f.name == "segmentation.onnx"));
            assert!(descriptor.files.iter().any(|f| f.name == "embedding.onnx"));
            assert!(descriptor.files.iter().all(|f| f.size_bytes > 0));
            assert!(!descriptor.description.is_empty());
        }
    }

    #[test]
    fn diarization_models_are_absent_from_the_picker_catalog() {
        assert!(builtin_catalog()
            .iter()
            .all(|d| d.family != ModelFamily::Diarization));
    }

    #[test]
    fn denoise_models_are_single_onnx_files_outside_the_picker() {
        let models = denoise_models();
        assert!(!models.is_empty());

        let ids: std::collections::HashSet<_> = models.iter().map(|d| &d.id).collect();
        assert_eq!(ids.len(), models.len(), "denoise ids must be distinct");

        for descriptor in &models {
            assert_eq!(descriptor.family, ModelFamily::Denoise);
            assert!(descriptor.files.iter().any(|f| f.name.ends_with(".onnx")));
            assert!(descriptor.files.iter().all(|f| f.size_bytes > 0));
            assert!(!descriptor.description.is_empty());
        }

        // Never shown in the ASR model picker.
        assert!(builtin_catalog()
            .iter()
            .all(|d| d.family != ModelFamily::Denoise));
    }
}
