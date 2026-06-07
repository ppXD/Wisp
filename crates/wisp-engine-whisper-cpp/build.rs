//! Builds the vendored whisper.cpp and generates Rust bindings for its C API.
//!
//! - **macOS**: Metal + Core ML backend (always).
//! - **Windows**: Vulkan backend — generic GPU across AMD/Intel/NVIDIA, with ggml's built-in CPU
//!   fallback — but only when the `vulkan` feature is on, so the default Windows build stays a no-op
//!   shell and needs no Vulkan SDK.
//! - **Other targets**: a no-op, so the crate is an empty shell.

use std::env;
use std::path::{Path, PathBuf};

fn main() {
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let windows_vulkan = target_os == "windows" && env::var("CARGO_FEATURE_VULKAN").is_ok();

    if target_os != "macos" && !windows_vulkan {
        return;
    }

    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let src = manifest.join("vendor/whisper.cpp");
    assert!(
        src.join("include/whisper.h").exists(),
        "whisper.cpp submodule missing — run `git submodule update --init --recursive`"
    );

    if target_os == "macos" {
        build_macos(&src);
    } else {
        build_windows_vulkan(&src);
    }

    generate_bindings(&src);
}

/// Builds whisper.cpp + ggml as static libs with the Metal backend, embedding the Metal shader
/// library into the binary so nothing extra has to ship at runtime, plus the Core ML encoder path.
fn build_macos(src: &Path) {
    let dst = cmake::Config::new(src)
        .profile("Release")
        // ggml uses std::filesystem (introduced in macOS 10.15). Pin a modern deployment target so
        // the C++ build doesn't inherit a lower one from the embedding app's build environment
        // (Tauri defaults low), which otherwise breaks the build with "unavailable" errors.
        .env("MACOSX_DEPLOYMENT_TARGET", "11.0")
        .define("CMAKE_OSX_DEPLOYMENT_TARGET", "11.0")
        .define("BUILD_SHARED_LIBS", "OFF")
        .define("WHISPER_BUILD_EXAMPLES", "OFF")
        .define("WHISPER_BUILD_TESTS", "OFF")
        .define("WHISPER_BUILD_SERVER", "OFF")
        .define("GGML_METAL", "ON")
        .define("GGML_METAL_EMBED_LIBRARY", "ON")
        .define("GGML_OPENMP", "OFF")
        // Core ML encoder path: when a `*-encoder.mlmodelc` sits next to the model, whisper.cpp runs
        // the encoder on the Apple Neural Engine. ALLOW_FALLBACK keeps Metal working when it's
        // absent, so this is safe even before (or without) the optional Core ML model download.
        .define("WHISPER_COREML", "ON")
        .define("WHISPER_COREML_ALLOW_FALLBACK", "ON")
        .build();

    // The static libs land in the install prefix and/or the build tree — search both so we're
    // robust to whichever ones whisper.cpp's install rules actually export.
    let build = dst.join("build");
    for dir in [
        dst.join("lib"),
        build.join("src"),
        build.join("ggml/src"),
        build.join("ggml/src/ggml-metal"),
        build.join("ggml/src/ggml-blas"),
    ] {
        println!("cargo:rustc-link-search=native={}", dir.display());
    }

    for lib in [
        "whisper",
        // The Core ML encoder lib (Objective-C++); `whisper` references it, so it follows here.
        "whisper.coreml",
        "ggml",
        "ggml-cpu",
        "ggml-metal",
        "ggml-blas",
        "ggml-base",
    ] {
        println!("cargo:rustc-link-lib=static={lib}");
    }

    // C++ runtime + the Apple frameworks the Metal, Accelerate, and Core ML backends need.
    println!("cargo:rustc-link-lib=c++");
    for framework in [
        "CoreML",
        "Metal",
        "MetalKit",
        "Foundation",
        "Accelerate",
        "CoreFoundation",
        "CoreGraphics",
        "QuartzCore",
    ] {
        println!("cargo:rustc-link-lib=framework={framework}");
    }
}

/// Builds whisper.cpp + ggml as static libs with the Vulkan backend (generic GPU; ggml falls back to
/// CPU when no Vulkan device is present). The Vulkan headers + loader import lib come from the Vulkan
/// SDK at build time (CI installs it; `VULKAN_SDK` points at it); at runtime `vulkan-1.dll` ships with
/// the GPU driver, so nothing extra has to be bundled.
fn build_windows_vulkan(src: &Path) {
    let mut config = cmake::Config::new(src);
    config
        .profile("Release")
        .define("BUILD_SHARED_LIBS", "OFF")
        .define("WHISPER_BUILD_EXAMPLES", "OFF")
        .define("WHISPER_BUILD_TESTS", "OFF")
        .define("WHISPER_BUILD_SERVER", "OFF")
        .define("GGML_VULKAN", "ON")
        .define("GGML_OPENMP", "OFF");

    // Point CMake's `find_package(Vulkan)` straight at the installed SDK. Auto-detection can miss it
    // on the CI runner even with VULKAN_SDK set, so pre-fill the include dir and loader import lib;
    // the glslc shader compiler is still found via VULKAN_SDK.
    if let Ok(sdk) = env::var("VULKAN_SDK") {
        config.define("Vulkan_INCLUDE_DIR", format!("{sdk}/Include"));
        config.define("Vulkan_LIBRARY", format!("{sdk}/Lib/vulkan-1.lib"));
    }

    let dst = config.build();

    // MSVC is multi-config, so the libs land under per-config (`Release`) subdirs of the build tree
    // and/or the install prefix — search the likely spots so linking is robust to either layout.
    let build = dst.join("build");
    for dir in [
        dst.join("lib"),
        build.join("src"),
        build.join("src/Release"),
        build.join("ggml/src"),
        build.join("ggml/src/Release"),
        build.join("ggml/src/ggml-vulkan"),
        build.join("ggml/src/ggml-vulkan/Release"),
    ] {
        println!("cargo:rustc-link-search=native={}", dir.display());
    }

    for lib in ["whisper", "ggml", "ggml-cpu", "ggml-vulkan", "ggml-base"] {
        println!("cargo:rustc-link-lib=static={lib}");
    }

    // The Vulkan loader import library, from the Vulkan SDK.
    if let Ok(sdk) = env::var("VULKAN_SDK") {
        println!("cargo:rustc-link-search=native={}\\Lib", sdk);
    }
    println!("cargo:rustc-link-lib=vulkan-1");
}

/// Generates the Rust FFI bindings for whisper.cpp's C API. Platform-independent — it only parses the
/// public headers, so the same bindings serve every backend.
fn generate_bindings(src: &Path) {
    let whisper_h = src.join("include/whisper.h");
    let bindings = bindgen::Builder::default()
        .header(whisper_h.to_string_lossy())
        .clang_arg(format!("-I{}", src.join("ggml/include").display()))
        .allowlist_function("whisper_.*")
        .allowlist_type("whisper_.*")
        .allowlist_var("WHISPER_.*")
        .default_enum_style(bindgen::EnumVariation::Rust {
            non_exhaustive: false,
        })
        .generate()
        .expect("generate whisper.cpp bindings");
    bindings
        .write_to_file(PathBuf::from(env::var("OUT_DIR").unwrap()).join("bindings.rs"))
        .expect("write whisper.cpp bindings");

    println!("cargo:rerun-if-changed={}", whisper_h.display());
}
