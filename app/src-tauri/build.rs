#[cfg(target_os = "windows")]
#[path = "build_support/windows_runtime.rs"]
mod windows_runtime;

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
    const SHERPA_DLLS: &[&str] = &[
        "onnxruntime.dll",
        "onnxruntime_providers_shared.dll",
        "sherpa-onnx-c-api.dll",
        "cargs.dll",
    ];

    let target_dir = cargo_target_dir();
    let staged = Path::new(env!("CARGO_MANIFEST_DIR")).join("windows-runtime");
    std::fs::create_dir_all(&staged).expect("create windows-runtime dir");

    for dll in SHERPA_DLLS {
        let src = target_dir.join(dll);
        let dst = staged.join(dll);
        windows_runtime::copy_if_changed(&src, &dst)
            .unwrap_or_else(|e| panic!("stage {}: {e}", src.display()));
    }

    stage_msvc_runtime(&staged);
}

/// App-local deployment of the MSVC runtime used to compile whisper.cpp. The runtime installed on a
/// user's machine may be older than the release runner's toolset; MSVC only guarantees compatibility
/// when the runtime is at least as new as the build tools. A mismatch can crash in C++ static
/// initialization before Tauri or WebView2 has started.
#[cfg(target_os = "windows")]
fn stage_msvc_runtime(staged: &std::path::Path) {
    println!("cargo:rerun-if-env-changed=VCToolsRedistDir");
    println!("cargo:rerun-if-changed=msvc-runtime-required.txt");
    let source = msvc_runtime_dir().unwrap_or_else(|| {
        panic!(
            "could not find the x64 MSVC redistributable directory; run Cargo from an MSVC \
             developer shell or install the Visual C++ x64 build tools"
        )
    });

    let required: Vec<_> = include_str!("msvc-runtime-required.txt")
        .lines()
        .map(str::trim)
        .filter(|entry| !entry.is_empty() && !entry.starts_with('#'))
        .map(|entry| entry.split_once(':').map(|(name, _)| name).unwrap_or(entry))
        .collect();
    let staged = staged.join("msvc");
    let runtime_dlls = windows_runtime::sync_runtime_dlls(&source, &staged, &required)
        .unwrap_or_else(|e| panic!("stage MSVC runtime from {}: {e}", source.display()));
    for dll in runtime_dlls {
        println!("cargo:rerun-if-changed={}", dll.display());
    }
}

/// Resolves the current toolset's x64 CRT directory. `ilammy/msvc-dev-cmd` supplies the env var in
/// release CI; `vswhere` keeps ordinary local Cargo builds working outside a Developer shell.
#[cfg(target_os = "windows")]
fn msvc_runtime_dir() -> Option<std::path::PathBuf> {
    std::env::var_os("VCToolsRedistDir")
        .map(std::path::PathBuf::from)
        .and_then(|root| windows_runtime::find_crt_dir(&root))
        .or_else(msvc_runtime_dir_from_vswhere)
}

#[cfg(target_os = "windows")]
fn msvc_runtime_dir_from_vswhere() -> Option<std::path::PathBuf> {
    use std::process::Command;

    let program_files_x86 = std::env::var_os("ProgramFiles(x86)")?;
    let vswhere = std::path::PathBuf::from(program_files_x86)
        .join("Microsoft Visual Studio")
        .join("Installer")
        .join("vswhere.exe");
    let output = Command::new(vswhere)
        .args([
            "-latest",
            "-products",
            "*",
            "-requires",
            "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
            "-property",
            "installationPath",
        ])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }

    let installation = String::from_utf8(output.stdout).ok()?;
    let redist_root = std::path::Path::new(installation.trim())
        .join("VC")
        .join("Redist")
        .join("MSVC");
    windows_runtime::newest_crt_dir(&redist_root)
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
