fn main() {
    // screencapturekit pulls in Apple's Swift runtime via a Swift bridge; the binary references
    // `@rpath/libswift_Concurrency.dylib`. macOS ships the Swift runtime under /usr/lib/swift
    // (resolved from the dyld shared cache), so add it to the binary's rpath.
    #[cfg(target_os = "macos")]
    println!("cargo:rustc-link-arg=-Wl,-rpath,/usr/lib/swift");

    tauri_build::build()
}
