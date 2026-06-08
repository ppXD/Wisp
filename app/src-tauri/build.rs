fn main() {
    #[cfg(target_os = "macos")]
    {
        // screencapturekit pulls in Apple's Swift runtime via a Swift bridge; the binary references
        // `@rpath/libswift_Concurrency.dylib`. macOS ships the Swift runtime under /usr/lib/swift
        // (resolved from the dyld shared cache), so add it to the binary's rpath.
        println!("cargo:rustc-link-arg=-Wl,-rpath,/usr/lib/swift");

        // The sherpa-onnx + onnxruntime dylibs the binary loads via `@rpath` are bundled into the
        // packaged .app's `Contents/Frameworks` (see tauri.conf.json `bundle.macOS.frameworks`);
        // resolve `@rpath` against it so the standalone app finds them.
        println!("cargo:rustc-link-arg=-Wl,-rpath,@executable_path/../Frameworks");
    }

    // On Windows the binary dynamically links the sherpa-onnx + onnxruntime DLLs, which sherpa-rs-sys
    // drops in the cargo target dir at build time. Stage them in a fixed in-crate dir so
    // `tauri.windows.conf.json` can bundle them next to the installed .exe. `resources` paths are
    // checked at *compile* time, so pointing straight at `target/<profile>` would break debug builds
    // and `tauri dev`; a fixed dir is profile-independent.
    #[cfg(target_os = "windows")]
    stage_windows_runtime_libs();

    tauri_build::build()
}

/// Copies the sherpa-onnx + onnxruntime runtime DLLs from the cargo target dir into
/// `windows-runtime/` (next to this build script) so they can be bundled as Tauri resources. A
/// size check skips the copy when already current, so the build script doesn't perpetually re-fire.
#[cfg(target_os = "windows")]
fn stage_windows_runtime_libs() {
    use std::path::Path;

    // The DLLs sherpa-onnx's C API depends on at runtime, as shipped in the prebuilt win-x64-shared
    // package sherpa-rs-sys downloads.
    const DLLS: &[&str] = &[
        "onnxruntime.dll",
        "onnxruntime_providers_shared.dll",
        "sherpa-onnx-c-api.dll",
        "cargs.dll",
    ];

    let target_dir = cargo_target_dir();
    let staged = Path::new(env!("CARGO_MANIFEST_DIR")).join("windows-runtime");
    std::fs::create_dir_all(&staged).expect("create windows-runtime dir");

    for dll in DLLS {
        let src = target_dir.join(dll);
        let dst = staged.join(dll);
        let current = std::fs::metadata(&dst)
            .ok()
            .zip(std::fs::metadata(&src).ok())
            .is_some_and(|(d, s)| d.len() == s.len());
        if !current {
            std::fs::copy(&src, &dst).unwrap_or_else(|e| panic!("stage {}: {e}", src.display()));
        }
    }

    // The whisper.cpp Vulkan build also bundles the Vulkan loader so the one installer runs anywhere:
    // it uses the GPU when a driver is present and falls back to CPU when there's no Vulkan device,
    // instead of failing to launch on a machine with no system loader.
    if std::env::var("CARGO_FEATURE_WHISPER_VULKAN").is_ok() {
        stage_vulkan_loader(&staged);
    }
}

/// Copies the Vulkan loader (`vulkan-1.dll`) from the Vulkan SDK into `windows-runtime/` so the
/// whisper.cpp Vulkan build can bundle it next to the .exe. The SDK ships it under `Bin/`; some
/// layouts use `runtime/x64/`, so check both. Only invoked for the `whisper-vulkan` feature, whose
/// build always has the SDK on `VULKAN_SDK`.
#[cfg(target_os = "windows")]
fn stage_vulkan_loader(staged: &std::path::Path) {
    use std::path::Path;

    let sdk = std::env::var("VULKAN_SDK").expect("VULKAN_SDK is set for the whisper-vulkan build");
    let src = ["Bin/vulkan-1.dll", "runtime/x64/vulkan-1.dll"]
        .iter()
        .map(|rel| Path::new(&sdk).join(rel))
        .find(|p| p.exists())
        .unwrap_or_else(|| panic!("vulkan-1.dll not found under VULKAN_SDK={sdk}"));
    let dst = staged.join("vulkan-1.dll");

    let current = std::fs::metadata(&dst)
        .ok()
        .zip(std::fs::metadata(&src).ok())
        .is_some_and(|(d, s)| d.len() == s.len());
    if !current {
        std::fs::copy(&src, &dst)
            .unwrap_or_else(|e| panic!("stage vulkan-1.dll from {}: {e}", src.display()));
    }
}

/// The cargo `target/<profile>` dir, found by walking up from `OUT_DIR` — the same place
/// sherpa-rs-sys copies its DLLs to.
#[cfg(target_os = "windows")]
fn cargo_target_dir() -> std::path::PathBuf {
    let out_dir = std::path::PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let profile = std::env::var("PROFILE").unwrap();
    let mut dir = out_dir.as_path();
    while let Some(parent) = dir.parent() {
        if parent.ends_with(&profile) {
            return parent.to_path_buf();
        }
        dir = parent;
    }
    panic!(
        "could not find target/{profile} above OUT_DIR {}",
        out_dir.display()
    );
}
