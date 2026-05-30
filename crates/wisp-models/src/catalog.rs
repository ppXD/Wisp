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
