//! Compiles the Swift shim ([`swift/WispSpeech.swift`]) into a static library and links it plus the
//! Apple frameworks it needs. macOS only — every other target builds an empty stub crate.

fn main() {
    // `apple_speech_real` is set only when the genuine macOS-26 Swift shim compiles (below). Declaring
    // it on every target keeps `cfg(apple_speech_real)` from tripping the unexpected-cfg lint.
    println!("cargo:rustc-check-cfg=cfg(apple_speech_real)");

    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("macos") {
        return;
    }

    println!("cargo:rerun-if-changed=swift/WispSpeech.swift");

    let out_dir = std::env::var("OUT_DIR").expect("OUT_DIR");
    let lib_path = format!("{out_dir}/libWispSpeech.a");

    // The real shim calls `SpeechAnalyzer`/`SpeechTranscriber`, which only exist in the macOS-26 SDK.
    // On an older SDK (e.g. CI runners) compile a tiny stub exporting the same C ABI instead: the crate
    // still links and runs but reports itself unavailable. `apple_speech_real` marks the genuine build
    // so the live-availability test only runs where the API exists.
    let source = if macos_sdk_has_speech_analyzer() {
        println!("cargo:rustc-cfg=apple_speech_real");
        "swift/WispSpeech.swift".to_owned()
    } else {
        let stub_path = format!("{out_dir}/WispSpeechStub.swift");
        std::fs::write(&stub_path, STUB_SWIFT).expect("write Apple Speech stub shim");
        stub_path
    };

    // Deployment target 11.0 (matches the app) so the binary still runs on older macOS.
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
            &source,
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

/// Whether the active macOS SDK ships `SpeechAnalyzer` (macOS 26+). `xcrun --show-sdk-version` prints the
/// SDK version; anything we can't parse as a major ≥ 26 is treated as too old, so we fall back to the stub.
fn macos_sdk_has_speech_analyzer() -> bool {
    let Ok(output) = std::process::Command::new("xcrun")
        .args(["--show-sdk-version"])
        .output()
    else {
        return false;
    };
    if !output.status.success() {
        return false;
    }

    let version = String::from_utf8_lossy(&output.stdout);
    let major: u32 = version
        .trim()
        .split('.')
        .next()
        .and_then(|major| major.parse().ok())
        .unwrap_or(0);

    major >= 26
}

/// A no-API stub exporting the shim's C ABI so the crate links on pre-macOS-26 SDKs. Every entry point
/// reports "unavailable": `wisp_applespeech_available` is `false` and `start` returns null, so the Rust
/// `AppleSpeechEngine::new` fails cleanly with its macOS-26 requirement message before ever calling in.
const STUB_SWIFT: &str = r#"import Foundation

@_cdecl("wisp_applespeech_available")
public func wisp_applespeech_available() -> Bool { false }

@_cdecl("wisp_applespeech_start")
public func wisp_applespeech_start(
    _ locale: UnsafePointer<CChar>?,
    _ ctx: UnsafeMutableRawPointer?,
    _ callback: @convention(c) (UnsafeMutableRawPointer?, UnsafePointer<CChar>?, Bool) -> Void
) -> UnsafeMutableRawPointer? { nil }

@_cdecl("wisp_applespeech_feed")
public func wisp_applespeech_feed(
    _ handle: UnsafeMutableRawPointer?,
    _ samples: UnsafePointer<Float>?,
    _ count: Int,
    _ rate: Double
) {}

@_cdecl("wisp_applespeech_stop")
public func wisp_applespeech_stop(_ handle: UnsafeMutableRawPointer?) {}
"#;
