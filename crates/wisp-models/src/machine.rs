//! Machine-aware engine/model selection.
//!
//! Auto-detecting the host and picking the model that best uses it — the highest accuracy the
//! machine can run in real time — is the cross-platform heart of "use the best engine for this Mac
//! / PC". The detection (which accelerator, how much memory) is platform-specific and lives in the
//! app shell; the *choice* is pure and lives here, so it's the same logic everywhere and fully
//! testable.

use wisp_core::model::{ModelDescriptor, ModelId};

/// RAM at/above which the higher-precision q8 default is recommended; below it, the lighter q5.
const Q8_MIN_RAM_BYTES: u64 = 16 * 1024 * 1024 * 1024; // 16 GiB

/// The on-device accelerator the ASR engine can run on.
///
/// `#[non_exhaustive]` so new backends can be added without breaking matches.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum Accelerator {
    /// Apple GPU via Metal (whisper.cpp) — Apple Silicon.
    Metal,
    /// NVIDIA GPU via CUDA.
    Cuda,
    /// Cross-vendor GPU via Vulkan.
    Vulkan,
    /// Windows GPU via DirectML.
    DirectMl,
    /// No GPU engine available — CPU (ONNX) only.
    Cpu,
}

/// What the host machine can run, for picking a default model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MachineProfile {
    /// The best accelerator an ASR engine can use here.
    pub accelerator: Accelerator,
    /// Total physical memory in bytes.
    pub ram_bytes: u64,
}

impl MachineProfile {
    /// A profile for the given accelerator and memory.
    pub fn new(accelerator: Accelerator, ram_bytes: u64) -> Self {
        Self {
            accelerator,
            ram_bytes,
        }
    }
}

/// Auto-picks the default ASR model for `profile`: the engine family that best uses the machine's
/// accelerator, sized to its memory — the most accurate setup it can run in real time.
///
/// - **Metal** (Apple Silicon): the GPU whisper.cpp turbo — q8 with memory headroom (≥ 16 GB), the
///   lighter/faster q5 below.
/// - **Anything else** (Windows / Linux, today CPU-bound ONNX): SenseVoice int8 — non-autoregressive,
///   so fast on the CPU, with strong everyday and CJK accuracy. The GPU arms (CUDA/Vulkan/DirectML)
///   route here too until a GPU engine is wired for those platforms, then they grow to it.
///
/// The chosen id is guaranteed to exist in `catalog` (falling back to its first entry), so the
/// caller can always install what's recommended.
pub fn recommended_default_model(profile: &MachineProfile, catalog: &[ModelDescriptor]) -> ModelId {
    let ideal = match profile.accelerator {
        Accelerator::Metal if profile.ram_bytes >= Q8_MIN_RAM_BYTES => "whisper-large-v3-turbo-q8",
        Accelerator::Metal => "whisper-large-v3-turbo-q5",
        Accelerator::Cuda | Accelerator::Vulkan | Accelerator::DirectMl | Accelerator::Cpu => {
            "sense-voice"
        }
    };

    let id = ModelId(ideal.to_owned());
    if catalog.iter().any(|d| d.id == id) {
        id
    } else {
        catalog.first().map(|d| d.id.clone()).unwrap_or(id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtin_catalog;

    const GIB: u64 = 1024 * 1024 * 1024;

    #[test]
    fn metal_picks_turbo_sized_to_memory() {
        let catalog = builtin_catalog();
        // Apple Silicon with headroom → the higher-precision q8.
        assert_eq!(
            recommended_default_model(&MachineProfile::new(Accelerator::Metal, 32 * GIB), &catalog),
            ModelId("whisper-large-v3-turbo-q8".to_owned())
        );
        // The 16 GB boundary → q8.
        assert_eq!(
            recommended_default_model(&MachineProfile::new(Accelerator::Metal, 16 * GIB), &catalog),
            ModelId("whisper-large-v3-turbo-q8".to_owned())
        );
        // A memory-constrained Apple Silicon (e.g. 8 GB base) → the lighter q5.
        assert_eq!(
            recommended_default_model(&MachineProfile::new(Accelerator::Metal, 8 * GIB), &catalog),
            ModelId("whisper-large-v3-turbo-q5".to_owned())
        );
    }

    #[test]
    fn non_metal_picks_the_cross_platform_cpu_baseline() {
        let catalog = builtin_catalog();
        // Windows/Linux without a wired GPU engine → fast non-autoregressive SenseVoice, regardless
        // of accelerator flavour or RAM (it's light).
        for accel in [
            Accelerator::Cpu,
            Accelerator::Cuda,
            Accelerator::Vulkan,
            Accelerator::DirectMl,
        ] {
            assert_eq!(
                recommended_default_model(&MachineProfile::new(accel, 64 * GIB), &catalog),
                ModelId("sense-voice".to_owned()),
                "{accel:?} should pick the cross-platform baseline today"
            );
        }
    }

    #[test]
    fn recommended_ids_always_exist_in_the_catalog() {
        let ids: std::collections::HashSet<_> =
            builtin_catalog().into_iter().map(|d| d.id).collect();
        for accel in [Accelerator::Metal, Accelerator::Cpu, Accelerator::Cuda] {
            for ram in [8u64, 16, 64].map(|g| g * GIB) {
                let id =
                    recommended_default_model(&MachineProfile::new(accel, ram), &builtin_catalog());
                assert!(ids.contains(&id), "{accel:?}/{ram} → {id:?} not in catalog");
            }
        }
    }

    #[test]
    fn falls_back_to_the_first_entry_when_the_ideal_is_absent() {
        // A catalog missing the ideal id still yields something installable (its first entry).
        let only = builtin_catalog()
            .into_iter()
            .filter(|d| d.id != ModelId("sense-voice".to_owned()))
            .collect::<Vec<_>>();
        let picked =
            recommended_default_model(&MachineProfile::new(Accelerator::Cpu, 8 * GIB), &only);
        assert_eq!(picked, only[0].id);
    }
}
