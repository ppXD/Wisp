use std::io;
use std::path::{Path, PathBuf};

pub fn sync_runtime_dlls(
    source: &Path,
    staged: &Path,
    required: &[&str],
) -> io::Result<Vec<PathBuf>> {
    let runtime_dlls = discover_runtime_dlls(source)?;
    validate_required_runtime(&runtime_dlls, required)?;
    std::fs::create_dir_all(staged)?;
    remove_stale_runtime_dlls(staged, &runtime_dlls)?;

    for source in &runtime_dlls {
        let file_name = source.file_name().expect("discovered DLL has a file name");
        copy_if_changed(source, &staged.join(file_name))?;
    }

    Ok(runtime_dlls)
}

pub fn newest_crt_dir(redist_root: &Path) -> Option<PathBuf> {
    let mut versions: Vec<_> = std::fs::read_dir(redist_root)
        .ok()?
        .filter_map(Result::ok)
        .filter(|entry| entry.file_type().is_ok_and(|kind| kind.is_dir()))
        .map(|entry| entry.path())
        .collect();
    versions.sort_by(|a, b| b.file_name().cmp(&a.file_name()));
    versions.into_iter().find_map(|root| find_crt_dir(&root))
}

pub fn find_crt_dir(redist_root: &Path) -> Option<PathBuf> {
    let x64 = redist_root.join("x64");
    std::fs::read_dir(x64)
        .ok()?
        .filter_map(Result::ok)
        .find(|entry| {
            entry.file_type().is_ok_and(|kind| kind.is_dir())
                && entry
                    .file_name()
                    .to_string_lossy()
                    .starts_with("Microsoft.VC")
                && entry.file_name().to_string_lossy().ends_with(".CRT")
                && entry.path().join("msvcp140.dll").is_file()
        })
        .map(|entry| entry.path())
}

fn discover_runtime_dlls(source: &Path) -> io::Result<Vec<PathBuf>> {
    let mut dlls = Vec::new();
    for entry in std::fs::read_dir(source)? {
        let entry = entry?;
        if entry.file_type()?.is_file() && has_dll_extension(&entry.path()) {
            dlls.push(entry.path());
        }
    }
    dlls.sort_by_key(|path| file_name_lowercase(path));
    Ok(dlls)
}

fn validate_required_runtime(runtime_dlls: &[PathBuf], required: &[&str]) -> io::Result<()> {
    let present: std::collections::HashSet<_> = runtime_dlls
        .iter()
        .map(|path| file_name_lowercase(path))
        .collect();
    let missing: Vec<_> = required
        .iter()
        .filter(|name| !present.contains(&name.to_ascii_lowercase()))
        .collect();
    if missing.is_empty() {
        return Ok(());
    }

    Err(io::Error::new(
        io::ErrorKind::NotFound,
        format!(
            "MSVC runtime directory {} is missing required files: {}",
            runtime_dlls
                .first()
                .and_then(|path| path.parent())
                .unwrap_or_else(|| Path::new("<empty>"))
                .display(),
            missing.into_iter().copied().collect::<Vec<_>>().join(", ")
        ),
    ))
}

fn remove_stale_runtime_dlls(staged: &Path, current: &[PathBuf]) -> io::Result<()> {
    let current_names: std::collections::HashSet<_> = current
        .iter()
        .map(|path| file_name_lowercase(path))
        .collect();
    for entry in std::fs::read_dir(staged)? {
        let entry = entry?;
        if entry.file_type()?.is_file()
            && has_dll_extension(&entry.path())
            && !current_names.contains(&file_name_lowercase(&entry.path()))
        {
            std::fs::remove_file(entry.path())?;
        }
    }
    Ok(())
}

fn has_dll_extension(path: &Path) -> bool {
    path.extension()
        .is_some_and(|extension| extension.eq_ignore_ascii_case("dll"))
}

fn file_name_lowercase(path: &Path) -> String {
    path.file_name()
        .expect("runtime path has a file name")
        .to_string_lossy()
        .to_ascii_lowercase()
}

pub fn copy_if_changed(source: &Path, destination: &Path) -> io::Result<()> {
    let source_bytes = std::fs::read(source)?;
    if std::fs::read(destination).ok().as_deref() != Some(source_bytes.as_slice()) {
        std::fs::write(destination, source_bytes)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    static NEXT_DIR: AtomicUsize = AtomicUsize::new(0);
    const REQUIRED: &[&str] = &[
        "msvcp140.dll",
        "msvcp140_1.dll",
        "vcruntime140.dll",
        "vcruntime140_1.dll",
    ];

    struct TestDir(PathBuf);

    impl TestDir {
        fn new(name: &str) -> Self {
            let unique = NEXT_DIR.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                "wisp-runtime-{name}-{}-{unique}",
                std::process::id()
            ));
            std::fs::create_dir(&path).expect("create unique test dir");
            Self(path)
        }

        fn join(&self, name: &str) -> PathBuf {
            self.0.join(name)
        }
    }

    impl Drop for TestDir {
        fn drop(&mut self) {
            std::fs::remove_dir_all(&self.0).expect("clean unique test dir");
        }
    }

    fn write_runtime_set(dir: &TestDir, extra: &str) {
        write_runtime_set_at(&dir.0, extra);
    }

    fn write_runtime_set_at(dir: &Path, extra: &str) {
        std::fs::create_dir_all(dir).unwrap();
        for name in REQUIRED {
            std::fs::write(dir.join(name), format!("fixture {name}")).unwrap();
        }
        std::fs::write(dir.join(extra), b"toolset-specific").unwrap();
        std::fs::write(dir.join("README.txt"), b"not a runtime").unwrap();
    }

    #[test]
    fn syncs_the_dlls_present_in_each_toolset_and_removes_stale_files() {
        for (toolset, extra) in [
            ("vc143", "vcamp140.dll"),
            ("vc145", "msvcp140_atomic_wait.dll"),
        ] {
            let source = TestDir::new(toolset);
            let staged = TestDir::new("staged");
            write_runtime_set(&source, extra);
            std::fs::write(staged.join("removed-in-new-toolset.dll"), b"stale").unwrap();

            let copied = sync_runtime_dlls(&source.0, &staged.0, REQUIRED).unwrap();
            let mut names: Vec<_> = copied
                .iter()
                .map(|path| path.file_name().unwrap().to_string_lossy().into_owned())
                .collect();
            names.sort();

            let mut expected: Vec<_> = REQUIRED.iter().map(|name| (*name).to_owned()).collect();
            expected.push(extra.to_owned());
            expected.sort();
            assert_eq!(names, expected);
            assert!(!staged.join("removed-in-new-toolset.dll").exists());
            assert!(!staged.join("README.txt").exists());
            assert_eq!(
                std::fs::read(staged.join(extra)).unwrap(),
                b"toolset-specific"
            );
        }
    }

    #[test]
    fn rejects_a_toolset_missing_a_required_runtime() {
        let source = TestDir::new("incomplete");
        let staged = TestDir::new("staged");
        std::fs::write(source.join("msvcp140.dll"), b"only one").unwrap();

        let error = sync_runtime_dlls(&source.0, &staged.0, REQUIRED).unwrap_err();

        assert!(error.to_string().contains("vcruntime140.dll"));
    }

    #[test]
    fn resolves_the_newest_crt_without_assuming_a_toolset_generation() {
        let redist = TestDir::new("redist");
        let vc143 = redist.join("14.44.35211/x64/Microsoft.VC143.CRT");
        let vc145 = redist.join("14.51.36231/x64/Microsoft.VC145.CRT");
        write_runtime_set_at(&vc143, "vcamp140.dll");
        write_runtime_set_at(&vc145, "msvcp140_atomic_wait.dll");

        assert_eq!(newest_crt_dir(&redist.0), Some(vc145.clone()));
        assert_eq!(find_crt_dir(&redist.join("14.44.35211")), Some(vc143));
    }
}
