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
        copy_if_changed(&src, &dst);
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
    println!("cargo:rerun-if-changed=msvc-runtime-dlls.txt");
    let source = msvc_runtime_dir().unwrap_or_else(|| {
        panic!(
            "could not find the x64 MSVC redistributable directory; run Cargo from an MSVC \
             developer shell or install the Visual C++ x64 build tools"
        )
    });

    for entry in include_str!("msvc-runtime-dlls.txt").lines() {
        let entry = entry.trim();
        if entry.is_empty() || entry.starts_with('#') {
            continue;
        }
        let dll = entry
            .split_once(':')
            .map(|(name, _)| name)
            .unwrap_or(entry);
        let src = source.join(dll);
        println!("cargo:rerun-if-changed={}", src.display());
        copy_if_changed(&src, &staged.join(dll));
    }
}

/// Resolves the current toolset's x64 CRT directory. `ilammy/msvc-dev-cmd` supplies the env var in
/// release CI; `vswhere` keeps ordinary local Cargo builds working outside a Developer shell.
#[cfg(target_os = "windows")]
fn msvc_runtime_dir() -> Option<std::path::PathBuf> {
    std::env::var_os("VCToolsRedistDir")
        .map(std::path::PathBuf::from)
        .and_then(|root| find_crt_dir(&root))
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
    newest_crt_dir(&redist_root)
}

#[cfg(target_os = "windows")]
fn newest_crt_dir(redist_root: &std::path::Path) -> Option<std::path::PathBuf> {
    let mut versions: Vec<_> = std::fs::read_dir(redist_root)
        .ok()?
        .filter_map(Result::ok)
        .filter(|entry| entry.file_type().is_ok_and(|kind| kind.is_dir()))
        .map(|entry| entry.path())
        .collect();
    versions.sort_by(|a, b| b.file_name().cmp(&a.file_name()));
    versions.into_iter().find_map(|root| find_crt_dir(&root))
}

#[cfg(target_os = "windows")]
fn find_crt_dir(redist_root: &std::path::Path) -> Option<std::path::PathBuf> {
    let x64 = redist_root.join("x64");
    std::fs::read_dir(x64)
        .ok()?
        .filter_map(Result::ok)
        .find(|entry| {
            entry.file_type().is_ok_and(|kind| kind.is_dir())
                && entry.file_name().to_string_lossy().starts_with("Microsoft.VC")
                && entry.file_name().to_string_lossy().ends_with(".CRT")
                && entry.path().join("msvcp140.dll").is_file()
        })
        .map(|entry| entry.path())
}

#[cfg(target_os = "windows")]
fn copy_if_changed(src: &std::path::Path, dst: &std::path::Path) {
    let source = std::fs::read(src).unwrap_or_else(|e| panic!("read {}: {e}", src.display()));
    let current = std::fs::read(dst).ok();
    if current.as_deref() != Some(source.as_slice()) {
        std::fs::write(dst, source).unwrap_or_else(|e| panic!("stage {}: {e}", src.display()));
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
