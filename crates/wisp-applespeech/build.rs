//! Compiles the Swift shim ([`swift/WispSpeech.swift`]) into a static library and links it plus the
//! Apple frameworks it needs. macOS only — every other target builds an empty stub crate.

fn main() {
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("macos") {
        return;
    }

    println!("cargo:rerun-if-changed=swift/WispSpeech.swift");

    let out_dir = std::env::var("OUT_DIR").expect("OUT_DIR");
    let lib_path = format!("{out_dir}/libWispSpeech.a");

    // Deployment target 11.0 (matches the app) so the binary still runs on older macOS; the macOS-26
    // SpeechAnalyzer symbols are weak-linked and `#available`-guarded in the Swift.
    let status = std::process::Command::new("swiftc")
        .args([
            "-emit-library",
            "-static",
            "-O",
            "-target",
            "arm64-apple-macosx11.0",
            "-module-name",
            "WispSpeech",
            "-o",
            &lib_path,
            "swift/WispSpeech.swift",
        ])
        .status()
        .expect("failed to run swiftc — is the Xcode command-line toolchain installed?");
    assert!(
        status.success(),
        "swiftc failed to build the Apple Speech shim"
    );

    println!("cargo:rustc-link-search=native={out_dir}");
    println!("cargo:rustc-link-lib=static=WispSpeech");

    // The static Swift lib force-loads the Swift runtime + back-deploy compatibility archives
    // (libswiftCompatibility56.a, libswiftCompatibilityConcurrency.a, …). They live in the toolchain's
    // Swift lib dir, which isn't on the default linker search path — add it.
    if let Some(swift_lib) = toolchain_swift_lib_dir() {
        println!("cargo:rustc-link-search=native={swift_lib}");
    }
    // The dynamic Swift runtime resolves at load time from the OS copy.
    println!("cargo:rustc-link-arg=-Wl,-rpath,/usr/lib/swift");

    for framework in [
        "Speech",
        "AVFAudio",
        "AVFoundation",
        "CoreMedia",
        "Foundation",
    ] {
        println!("cargo:rustc-link-lib=framework={framework}");
    }
}

/// `<toolchain>/usr/lib/swift/macosx`, derived from `xcrun --find swiftc`, or `None` if `xcrun` fails.
fn toolchain_swift_lib_dir() -> Option<String> {
    let output = std::process::Command::new("xcrun")
        .args(["--find", "swiftc"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }

    let swiftc = String::from_utf8(output.stdout).ok()?;
    let usr = std::path::Path::new(swiftc.trim()).parent()?.parent()?; // .../usr/bin/swiftc -> .../usr
    Some(usr.join("lib/swift/macosx").to_string_lossy().into_owned())
}
