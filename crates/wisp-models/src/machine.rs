//! Machine-aware engine/model selection.
//!
//! Auto-detecting the host and picking the model that best uses it — the highest accuracy the
//! machine can run in real time — is the cross-platform heart of "use the best engine for this Mac
//! / PC". The detection (which accelerator, how much memory) is platform-specific and lives in the
//! app shell; the *choice* is pure and lives here, so it's the same logic everywhere and fully
//! testable.

use wisp_core::model::{ModelDescriptor, ModelFamily, ModelId};

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

/// How powerful the GPU ASR engine is on this host — the signal that decides which Whisper a machine
/// can run *in real time*. Ordered (`None < Entry < Standard < High < Ultra`) so the recommender can
/// gate models by tier. On Apple Silicon it maps the chip class (base / Pro / Max / Ultra); other
/// platforms report `None` until a GPU engine is wired (then they map here too).
///
/// `#[non_exhaustive]` so finer tiers can be inserted without breaking matches.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[non_exhaustive]
pub enum GpuTier {
    /// No usable GPU ASR engine — CPU only.
    None,
    /// Entry GPU — Apple M-series base chip.
    Entry,
    /// Mid GPU — Apple "Pro" chip.
    Standard,
    /// Strong GPU — Apple "Max" chip.
    High,
    /// Top GPU — Apple "Ultra" chip.
    Ultra,
}

/// What the host machine can run, for picking a default model — the accelerator, its power tier, and
/// memory. The recommendation is *computed* from these rather than hard-coded per platform.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MachineProfile {
    /// The best accelerator an ASR engine can use here.
    pub accelerator: Accelerator,
    /// Total physical memory in bytes.
    pub ram_bytes: u64,
    /// The GPU ASR engine's power tier — the real driver of "can this machine run model X live".
    pub gpu_tier: GpuTier,
}

impl MachineProfile {
    /// A profile from just accelerator + memory, inferring a GPU tier from memory as a fallback (used
    /// by tests and any host that can't probe the chip). Real hosts call [`MachineProfile::detailed`]
    /// with the detected chip tier.
    pub fn new(accelerator: Accelerator, ram_bytes: u64) -> Self {
        Self::detailed(
            accelerator,
            ram_bytes,
            inferred_gpu_tier(accelerator, ram_bytes),
        )
    }

    /// A profile with an explicitly detected GPU tier (the chip class on Apple Silicon).
    pub fn detailed(accelerator: Accelerator, ram_bytes: u64, gpu_tier: GpuTier) -> Self {
        Self {
            accelerator,
            ram_bytes,
            gpu_tier,
        }
    }
}

/// A GPU tier inferred from accelerator + memory when the chip can't be probed: Metal hosts step up
/// with RAM (a coarse proxy for chip class), everything else has no GPU engine.
fn inferred_gpu_tier(accelerator: Accelerator, ram_bytes: u64) -> GpuTier {
    const GIB: u64 = 1024 * 1024 * 1024;
    match accelerator {
        Accelerator::Metal if ram_bytes >= 32 * GIB => GpuTier::High,
        Accelerator::Metal if ram_bytes >= 16 * GIB => GpuTier::Standard,
        Accelerator::Metal => GpuTier::Entry,
        _ => GpuTier::None,
    }
}

/// Auto-picks the **Live** default for `profile` — the most accurate Whisper its GPU tier can keep up
/// with in *real time*, computed from the tier rather than hard-coded per platform:
///
/// - **Ultra** GPU → the full large-v3 (real time only on the strongest chips).
/// - **High / Standard** → large-v3-turbo q8 (the everyday GPU sweet spot).
/// - **Entry** → the lighter turbo q5.
/// - **None** (no GPU engine) → SenseVoice — non-autoregressive, so real-time on the CPU.
///
/// As chips get faster the tier rises and the pick follows automatically. The chosen id is guaranteed
/// to exist in `catalog` (falling back to its first entry).
pub fn recommended_default_model(profile: &MachineProfile, catalog: &[ModelDescriptor]) -> ModelId {
    let ideal = match profile.gpu_tier {
        GpuTier::Ultra => "whisper-large-v3-gpu",
        GpuTier::High | GpuTier::Standard => "whisper-large-v3-turbo-q8",
        GpuTier::Entry => "whisper-large-v3-turbo-q5",
        GpuTier::None => "sense-voice",
    };
    resolve(ideal, catalog)
}

/// Auto-picks the **File** model for `profile`, where real-time speed isn't required — the most
/// accurate model the machine can run *at all*, again driven by the GPU tier:
///
/// - **Standard and up** (a real GPU) → the full (non-turbo) large-v3 — most accurate, slow is fine.
/// - **Entry** → large-v3-turbo q8 (the full model is painfully slow on a base chip).
/// - **None** → the ONNX large-v3 — most accurate on the CPU.
///
/// Companion to [`recommended_default_model`]; the chosen id is guaranteed to exist in `catalog`.
pub fn recommended_accurate_model(
    profile: &MachineProfile,
    catalog: &[ModelDescriptor],
) -> ModelId {
    let ideal = match profile.gpu_tier {
        GpuTier::Ultra | GpuTier::High | GpuTier::Standard => "whisper-large-v3-gpu",
        GpuTier::Entry => "whisper-large-v3-turbo-q8",
        GpuTier::None => "whisper-large-v3",
    };
    resolve(ideal, catalog)
}

/// Resolves an ideal id against `catalog`, falling back to its first entry if absent, so the caller
/// can always install what's recommended.
fn resolve(ideal: &str, catalog: &[ModelDescriptor]) -> ModelId {
    let id = ModelId(ideal.to_owned());
    if catalog.iter().any(|d| d.id == id) {
        id
    } else {
        catalog.first().map(|d| d.id.clone()).unwrap_or(id)
    }
}

/// Whether an ASR model of `family` can actually run on a host with `accelerator`, so the picker only
/// offers models this machine can start.
///
/// The GPU whisper.cpp family is the macOS Metal engine and needs [`Accelerator::Metal`]; the
/// CPU-ONNX families (SenseVoice, sherpa Whisper, streaming transducer) run on every platform. Pure,
/// so the per-platform capability lives in one tested place rather than scattered `cfg` checks.
pub fn family_runnable(family: ModelFamily, accelerator: Accelerator) -> bool {
    match family {
        ModelFamily::WhisperCpp => accelerator == Accelerator::Metal,
        _ => true,
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
    fn whisper_cpp_runs_only_on_metal() {
        // The GPU whisper.cpp engine is macOS/Metal only — every other accelerator must hide it.
        assert!(family_runnable(ModelFamily::WhisperCpp, Accelerator::Metal));
        for accel in [
            Accelerator::Cpu,
            Accelerator::Cuda,
            Accelerator::Vulkan,
            Accelerator::DirectMl,
        ] {
            assert!(
                !family_runnable(ModelFamily::WhisperCpp, accel),
                "{accel:?} has no Metal whisper.cpp engine"
            );
        }
    }

    #[test]
    fn cpu_onnx_families_run_on_every_accelerator() {
        // SenseVoice / sherpa Whisper / streaming transducer are CPU ONNX — runnable everywhere.
        for family in [
            ModelFamily::SenseVoice,
            ModelFamily::Whisper,
            ModelFamily::StreamingTransducer,
        ] {
            for accel in [Accelerator::Metal, Accelerator::Cpu, Accelerator::DirectMl] {
                assert!(
                    family_runnable(family, accel),
                    "{family:?} on {accel:?} should run"
                );
            }
        }
    }

    #[test]
    fn recommended_default_is_always_runnable_on_its_own_machine() {
        // The two policies must agree: whatever we recommend for a machine must be runnable on it,
        // so the auto-default is never a model the picker would (correctly) hide.
        let catalog = builtin_catalog();
        for accel in [
            Accelerator::Metal,
            Accelerator::Cpu,
            Accelerator::Cuda,
            Accelerator::Vulkan,
            Accelerator::DirectMl,
        ] {
            for ram in [8u64, 16, 64].map(|g| g * GIB) {
                let profile = MachineProfile::new(accel, ram);
                let id = recommended_default_model(&profile, &catalog);
                let family = catalog
                    .iter()
                    .find(|d| d.id == id)
                    .expect("recommended id is in the catalog")
                    .family;
                assert!(
                    family_runnable(family, accel),
                    "recommended {id:?} ({family:?}) must run on {accel:?}"
                );
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

    #[test]
    fn accurate_picks_the_full_large_v3_per_accelerator() {
        let catalog = builtin_catalog();
        // Metal → the full (non-turbo) large-v3 on the GPU, the most accurate local option.
        assert_eq!(
            recommended_accurate_model(
                &MachineProfile::new(Accelerator::Metal, 32 * GIB),
                &catalog
            ),
            ModelId("whisper-large-v3-gpu".to_owned())
        );
        // Anything else → the ONNX large-v3 (most accurate on the CPU).
        for accel in [Accelerator::Cpu, Accelerator::Cuda, Accelerator::DirectMl] {
            assert_eq!(
                recommended_accurate_model(&MachineProfile::new(accel, 16 * GIB), &catalog),
                ModelId("whisper-large-v3".to_owned()),
                "{accel:?} should pick the most accurate CPU model"
            );
        }
    }

    #[test]
    fn accurate_recommendation_is_in_catalog_and_runnable() {
        let catalog = builtin_catalog();
        let ids: std::collections::HashSet<_> = catalog.iter().map(|d| d.id.clone()).collect();
        for accel in [
            Accelerator::Metal,
            Accelerator::Cpu,
            Accelerator::Cuda,
            Accelerator::Vulkan,
            Accelerator::DirectMl,
        ] {
            let id = recommended_accurate_model(&MachineProfile::new(accel, 16 * GIB), &catalog);
            assert!(ids.contains(&id), "{accel:?} → {id:?} not in catalog");
            let family = catalog.iter().find(|d| d.id == id).unwrap().family;
            assert!(
                family_runnable(family, accel),
                "accurate {id:?} ({family:?}) must run on {accel:?}"
            );
        }
    }

    #[test]
    fn recommendation_climbs_with_the_gpu_tier() {
        let catalog = builtin_catalog();
        let live = |tier| {
            recommended_default_model(
                &MachineProfile::detailed(Accelerator::Metal, 32 * GIB, tier),
                &catalog,
            )
        };
        // Live tracks the chip: an Ultra runs the full large-v3 in real time, a Max/Pro gets turbo-q8,
        // a base chip the lighter q5 — not a fixed per-platform string.
        assert_eq!(
            live(GpuTier::Ultra),
            ModelId("whisper-large-v3-gpu".to_owned())
        );
        assert_eq!(
            live(GpuTier::High),
            ModelId("whisper-large-v3-turbo-q8".to_owned())
        );
        assert_eq!(
            live(GpuTier::Entry),
            ModelId("whisper-large-v3-turbo-q5".to_owned())
        );

        // File reaches higher (no real-time limit): the full large-v3 on a real GPU, turbo on a base.
        let file = |tier| {
            recommended_accurate_model(
                &MachineProfile::detailed(Accelerator::Metal, 32 * GIB, tier),
                &catalog,
            )
        };
        assert_eq!(
            file(GpuTier::High),
            ModelId("whisper-large-v3-gpu".to_owned())
        );
        assert_eq!(
            file(GpuTier::Entry),
            ModelId("whisper-large-v3-turbo-q8".to_owned())
        );
    }

    #[test]
    fn new_infers_a_gpu_tier_from_memory_when_the_chip_is_unknown() {
        // Without a probed chip, a Metal host steps up with RAM; non-Metal reports no GPU engine.
        assert_eq!(
            MachineProfile::new(Accelerator::Metal, 32 * GIB).gpu_tier,
            GpuTier::High
        );
        assert_eq!(
            MachineProfile::new(Accelerator::Metal, 8 * GIB).gpu_tier,
            GpuTier::Entry
        );
        assert_eq!(
            MachineProfile::new(Accelerator::Cpu, 64 * GIB).gpu_tier,
            GpuTier::None
        );
    }
}
