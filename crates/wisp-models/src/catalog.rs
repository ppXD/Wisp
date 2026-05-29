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

/// Hugging Face repo hosting the whisper.cpp GGUF models (no auth needed).
const WHISPER_CPP_BASE: &str = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main";

/// All models Wisp offers in the picker.
pub fn builtin_catalog() -> Vec<ModelDescriptor> {
    vec![
        whisper_turbo_q5(),
        whisper_turbo_q8(),
        whisper_large_v3_gpu(),
        sense_voice_int8(),
        sense_voice_fp32(),
        whisper_large_v3(),
        whisper_medium(),
    ]
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
            "Whisper large-v3-turbo on the GPU (Metal) — real Cantonese (yue) + ~99 languages, \
             near-large-v3 accuracy but far faster because it runs on the GPU instead of the CPU. \
             Recommended. q5-quantized (~0.55 GB)."
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
            "Same large-v3-turbo on the GPU (Metal), at higher q8 precision — fast and very \
             accurate. Larger download (~0.85 GB)."
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
            "Small multilingual model (~234M params), int8-quantized. Fast and light on memory \
             with great everyday accuracy — the recommended default."
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn catalog_has_distinct_ids_and_files() {
        let catalog = builtin_catalog();
        assert_eq!(catalog.len(), 7);

        let ids: std::collections::HashSet<_> = catalog.iter().map(|d| &d.id).collect();
        assert_eq!(ids.len(), catalog.len(), "model ids must be distinct");

        for descriptor in &catalog {
            // sherpa families ship ONNX + tokens; whisper.cpp ships a single GGUF `.bin`.
            match descriptor.family {
                ModelFamily::WhisperCpp => {
                    assert!(descriptor.files.iter().any(|f| f.name.ends_with(".bin")));
                }
                _ => {
                    assert!(descriptor.files.iter().any(|f| f.name.ends_with(".onnx")));
                    assert!(descriptor
                        .files
                        .iter()
                        .any(|f| f.name.ends_with("tokens.txt")));
                }
            }
            assert!(!descriptor.description.is_empty());
            assert!(descriptor.total_size_bytes() > 0);
        }
    }
}
