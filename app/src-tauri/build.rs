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

    tauri_build::build()
}
