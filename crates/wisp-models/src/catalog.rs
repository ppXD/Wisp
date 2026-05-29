//! The built-in model catalog Wisp offers out of the box.
//!
//! Checksums are currently unpinned (empty) — downloads are size-known and the store warns when
//! fetching an unpinned file. Pinning real SHA-256 values is a follow-up.

use wisp_core::model::{ModelDescriptor, ModelFamily, ModelFile, ModelId, Quant};

/// Hugging Face mirror hosting individual SenseVoice files (no auth needed).
const SENSE_VOICE_BASE: &str =
    "https://huggingface.co/csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/resolve/main";

/// All models Wisp offers in the picker.
pub fn builtin_catalog() -> Vec<ModelDescriptor> {
    vec![sense_voice_int8(), sense_voice_fp32()]
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn catalog_has_distinct_ids_and_files() {
        let catalog = builtin_catalog();
        assert_eq!(catalog.len(), 2);
        assert_ne!(catalog[0].id, catalog[1].id);
        for descriptor in &catalog {
            assert!(descriptor.files.iter().any(|f| f.name.ends_with(".onnx")));
            assert!(descriptor.files.iter().any(|f| f.name == "tokens.txt"));
            assert!(descriptor.total_size_bytes() > 0);
        }
    }
}
