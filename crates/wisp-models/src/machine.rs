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

/// Parses total physical memory (bytes) from the contents of Linux `/proc/meminfo` — the `MemTotal:`
/// line, whose value is in kibibytes (`MemTotal:  16384256 kB`). Returns `None` if that line is
/// absent or unparseable, so the caller falls back to a safe default. Pure, so the parse is tested
/// without a Linux host; the file read itself stays in the platform shell.
pub fn parse_meminfo_total_bytes(meminfo: &str) -> Option<u64> {
    let kib: u64 = meminfo
        .lines()
        .find_map(|line| line.strip_prefix("MemTotal:"))?
        .split_whitespace()
        .next()?
        .parse()
        .ok()?;
    Some(kib * 1024)
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
/// The GPU whisper.cpp family is the macOS Metal engine and needs [`Accelerator::Metal`]; Apple
/// on-device speech is macOS-only too (Metal is the macOS accelerator — the app layers the macOS-26
/// runtime check on top). The CPU-ONNX families (SenseVoice, sherpa Whisper, streaming transducer) run
/// on every platform. Pure, so the per-platform capability lives in one tested place rather than
/// scattered `cfg` checks.
pub fn family_runnable(family: ModelFamily, accelerator: Accelerator) -> bool {
    match family {
        // whisper.cpp runs on the Apple GPU (Metal), or — in a Windows GPU build — on Vulkan with a
        // built-in CPU fallback, so it's offered on Windows too (no GPU required to run, just to
        // accelerate). Apple on-device speech stays macOS/Metal-only.
        ModelFamily::WhisperCpp => {
            accelerator == Accelerator::Metal
                || (cfg!(target_os = "windows") && cfg!(feature = "whisper-vulkan"))
        }
        ModelFamily::AppleSpeech => accelerator == Accelerator::Metal,
        _ => true,
    }
}

/// How a model fits a host — for an honest picker that disables (with a reason) what can't run and
/// flags what runs but heavily, instead of silently hiding either.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelFit {
    /// Runs well on this machine.
    Ready,
    /// Runs, but is large for this machine's memory — selectable, with a caveat.
    Heavy(String),
    /// Can't run on this host at all — disabled in the picker, with the reason.
    Blocked(String),
}

/// A model is "heavy" when its on-disk size times this factor exceeds the host's RAM — a coarse proxy
/// for peak working-set pressure (weights load several times their size in scratch/activations). It
/// only ever flags big models on small-RAM hosts; an ample machine flags nothing.
const HEAVY_RAM_FACTOR: u64 = 6;

/// Assesses how `descriptor` fits a host with `profile`: blocked (the platform can't run this family),
/// heavy (runs but large for the RAM), or ready. Pure — the app layers the Apple-Speech macOS-26
/// runtime check on top, the one capability that can't be read from the profile alone.
pub fn model_fit(descriptor: &ModelDescriptor, profile: &MachineProfile) -> ModelFit {
    if !family_runnable(descriptor.family, profile.accelerator) {
        return ModelFit::Blocked(platform_block_reason(descriptor.family));
    }

    if descriptor
        .total_size_bytes()
        .saturating_mul(HEAVY_RAM_FACTOR)
        > profile.ram_bytes
    {
        return ModelFit::Heavy(heavy_reason(profile.ram_bytes));
    }

    ModelFit::Ready
}

/// Why a family can't run on a host lacking its accelerator — the text shown on the greyed entry.
fn platform_block_reason(family: ModelFamily) -> String {
    match family {
        ModelFamily::WhisperCpp => "Needs a macOS Metal GPU".to_owned(),
        ModelFamily::AppleSpeech => "Needs macOS".to_owned(),
        _ => "Not supported on this machine".to_owned(),
    }
}

/// The caveat shown on a runnable-but-large model.
fn heavy_reason(ram_bytes: u64) -> String {
    const GIB: u64 = 1024 * 1024 * 1024;
    format!("Large for {} GB RAM — may run slowly", ram_bytes / GIB)
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
    fn apple_speech_runs_only_on_metal() {
        // Apple's on-device recogniser is macOS-only (Metal is the macOS accelerator); every other
        // accelerator must hide it. The macOS-26 runtime gate is layered on top in the app.
        assert!(family_runnable(
            ModelFamily::AppleSpeech,
            Accelerator::Metal
        ));
        for accel in [
            Accelerator::Cpu,
            Accelerator::Cuda,
            Accelerator::Vulkan,
            Accelerator::DirectMl,
        ] {
            assert!(
                !family_runnable(ModelFamily::AppleSpeech, accel),
                "{accel:?} is not macOS — no Apple on-device speech"
            );
        }
    }

    #[test]
    fn model_fit_blocks_metal_only_families_off_metal() {
        let catalog = builtin_catalog();
        let cpu = MachineProfile::new(Accelerator::Cpu, 16 * GIB);
        for d in catalog
            .iter()
            .filter(|d| d.family == ModelFamily::WhisperCpp || d.family == ModelFamily::AppleSpeech)
        {
            assert!(
                matches!(model_fit(d, &cpu), ModelFit::Blocked(_)),
                "{:?} should be blocked off Metal",
                d.id
            );
        }
    }

    #[test]
    fn model_fit_is_ready_for_every_runnable_model_on_an_ample_machine() {
        // A 64 GB Apple Silicon Mac runs everything comfortably — nothing blocked, nothing heavy.
        let profile = MachineProfile::new(Accelerator::Metal, 64 * GIB);
        for d in builtin_catalog() {
            assert_eq!(model_fit(&d, &profile), ModelFit::Ready, "{:?}", d.id);
        }
    }

    #[test]
    fn model_fit_flags_big_models_heavy_on_a_small_ram_host_but_not_zero_download_ones() {
        let catalog = builtin_catalog();
        let small = MachineProfile::new(Accelerator::Metal, 8 * GIB);

        let big = catalog
            .iter()
            .find(|d| d.id == ModelId("whisper-large-v3".to_owned()))
            .expect("large-v3 in catalog");
        assert!(
            matches!(model_fit(big, &small), ModelFit::Heavy(_)),
            "a 1.8 GB model is heavy on 8 GB"
        );

        // Apple on-device speech downloads nothing of ours (0 bytes), so it's never "heavy".
        let apple = catalog
            .iter()
            .find(|d| d.family == ModelFamily::AppleSpeech)
            .expect("apple speech in catalog");
        assert_eq!(model_fit(apple, &small), ModelFit::Ready);
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

    #[test]
    fn parses_total_memory_from_proc_meminfo() {
        let meminfo = "MemTotal:       16384256 kB\nMemFree:         1000000 kB\nBuffers: 0 kB\n";
        assert_eq!(
            parse_meminfo_total_bytes(meminfo),
            Some(16_384_256 * 1024),
            "MemTotal kB → bytes"
        );
    }

    #[test]
    fn meminfo_without_a_valid_memtotal_is_none() {
        assert_eq!(parse_meminfo_total_bytes(""), None);
        assert_eq!(parse_meminfo_total_bytes("MemFree: 1000 kB\n"), None);
        assert_eq!(
            parse_meminfo_total_bytes("MemTotal:  not_a_number kB"),
            None
        );
    }
}
