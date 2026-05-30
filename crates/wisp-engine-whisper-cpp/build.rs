//! Builds the vendored whisper.cpp (with the Metal backend) and generates Rust bindings for its
//! C API. macOS-only for now; on other targets this is a no-op so the crate is an empty shell.

use std::env;
use std::path::PathBuf;

fn main() {
    if env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("macos") {
        return;
    }

    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let src = manifest.join("vendor/whisper.cpp");
    assert!(
        src.join("include/whisper.h").exists(),
        "whisper.cpp submodule missing — run `git submodule update --init --recursive`"
    );

    // Build whisper.cpp + ggml as static libs with the Metal backend, embedding the Metal shader
    // library into the binary so nothing extra has to ship at runtime.
    let dst = cmake::Config::new(&src)
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
