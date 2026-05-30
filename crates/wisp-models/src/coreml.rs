//! Optional Core ML (Apple Neural Engine) encoder assets for the whisper.cpp models.
//!
//! whisper.cpp auto-loads a `<base>-encoder.mlmodelc` directory sitting next to a model and runs the
//! encoder on the Neural Engine. These archives are large (~1.1 GB) and only help Apple-Silicon
//! Macs, so they're downloaded on demand, never bundled.

use wisp_core::model::{ModelDescriptor, ModelFamily};

use crate::catalog::WHISPER_CPP_BASE;

/// A downloadable Core ML encoder for a whisper.cpp model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CoremlAsset {
    /// URL of the `.mlmodelc.zip` archive.
    pub url: String,
    /// Directory the archive unzips to — the exact name whisper.cpp derives from the model and
    /// auto-loads next to it.
    pub dir_name: String,
    /// Archive size in bytes, for the download progress bar.
    pub size_bytes: u64,
}

/// The Core ML encoder asset for `descriptor`, if it's a whisper.cpp model that has one.
///
/// SenseVoice and sherpa-Whisper models run on ONNX and have no Core ML encoder, so they return
/// `None`. So does a whisper.cpp model whose encoder size we haven't pinned (no asset offered rather
/// than a progress bar with an unknown total).
pub fn coreml_asset(descriptor: &ModelDescriptor) -> Option<CoremlAsset> {
    if descriptor.family != ModelFamily::WhisperCpp {
        return None;
    }
    let bin = descriptor.files.iter().find(|f| f.name.ends_with(".bin"))?;
    let dir_name = coreml_encoder_dir_name(&bin.name);
    let size_bytes = coreml_size(&dir_name)?;
    Some(CoremlAsset {
        url: format!("{WHISPER_CPP_BASE}/{dir_name}.zip"),
        dir_name,
        size_bytes,
    })
}

/// The Core ML encoder directory name whisper.cpp derives from a model `.bin` filename.
///
/// Mirrors `whisper_get_coreml_path_encoder` in `vendor/whisper.cpp/src/whisper.cpp`: drop the
/// extension, drop a trailing `-qX_Y` quantization tag, append `-encoder.mlmodelc`. Quantization is
/// the decoder's; the Core ML encoder is shared across a model's quants. (Drift: pinned by tests
/// against the catalog's real `.bin` names — update both together if whisper.cpp changes.)
fn coreml_encoder_dir_name(bin_filename: &str) -> String {
    let mut s = bin_filename
        .rsplit_once('.')
        .map(|(stem, _ext)| stem)
        .unwrap_or(bin_filename)
        .to_owned();

    if let Some(pos) = s.rfind('-') {
        let tag = &s.as_bytes()[pos..];
        if tag.len() == 5 && tag[1] == b'q' && tag[3] == b'_' {
            s.truncate(pos);
        }
    }

    s.push_str("-encoder.mlmodelc");
    s
}

/// Pinned download sizes (Hugging Face `content-length`) for the encoders we offer, so the UI can
/// show a real progress bar before the download starts. An unknown name means no asset is offered.
fn coreml_size(dir_name: &str) -> Option<u64> {
    match dir_name {
        "ggml-large-v3-turbo-encoder.mlmodelc" => Some(1_173_393_014),
        "ggml-large-v3-encoder.mlmodelc" => Some(1_175_711_232),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtin_catalog;
    use wisp_core::model::ModelId;

    #[test]
    fn encoder_dir_name_mirrors_whisper_cpp_derivation() {
        // turbo q8 and q5 share one encoder (quantization is the decoder's).
        assert_eq!(
            coreml_encoder_dir_name("ggml-large-v3-turbo-q8_0.bin"),
            "ggml-large-v3-turbo-encoder.mlmodelc"
        );
        assert_eq!(
            coreml_encoder_dir_name("ggml-large-v3-turbo-q5_0.bin"),
            "ggml-large-v3-turbo-encoder.mlmodelc"
        );
        assert_eq!(
            coreml_encoder_dir_name("ggml-large-v3-q5_0.bin"),
            "ggml-large-v3-encoder.mlmodelc"
        );
        // A non-quantized name keeps its stem.
        assert_eq!(
            coreml_encoder_dir_name("ggml-base.bin"),
            "ggml-base-encoder.mlmodelc"
        );
    }

    #[test]
    fn every_whisper_cpp_catalog_model_has_a_pinned_coreml_asset() {
        // Drift guard: each shipped whisper.cpp model resolves to an asset (URL + pinned size), and
        // non-whisper.cpp models never do.
        for d in builtin_catalog() {
            let asset = coreml_asset(&d);
            match d.family {
                ModelFamily::WhisperCpp => {
                    let asset = asset.expect("whisper.cpp model must offer a Core ML asset");
                    assert!(asset.url.ends_with(".mlmodelc.zip"));
                    assert!(asset.size_bytes > 0);
                }
                _ => assert!(
                    asset.is_none(),
                    "only whisper.cpp models have Core ML encoders"
                ),
            }
        }
    }

    #[test]
    fn turbo_quants_resolve_to_the_same_encoder_url() {
        let by_id = |id: &str| {
            builtin_catalog()
                .into_iter()
                .find(|d| d.id == ModelId(id.to_owned()))
                .and_then(|d| coreml_asset(&d))
                .unwrap()
        };
        assert_eq!(
            by_id("whisper-large-v3-turbo-q8").url,
            by_id("whisper-large-v3-turbo-q5").url
        );
    }
}
