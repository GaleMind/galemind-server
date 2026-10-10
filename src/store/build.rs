use std::path::Path;

// `librocksdb-sys` emits the C++ runtime link flag from inside the code path
// that compiles RocksDB from source. When ROCKSDB_LIB_DIR points it at a
// prebuilt archive instead -- see scripts/stage-rocksdb-lib.sh -- that path is
// skipped and only the `rocksdb` and search-path flags are emitted, so the
// runtime has to be named here or the link fails on every C++ symbol RocksDB
// uses. Without ROCKSDB_LIB_DIR the sources are compiled as usual and
// librocksdb-sys still emits the flag itself, so this stays quiet.
fn main() {
    println!("cargo:rerun-if-env-changed=ROCKSDB_LIB_DIR");
    println!("cargo:rerun-if-env-changed=ROCKSDB_STATIC");
    println!("cargo:rerun-if-env-changed=ROCKSDB_COMPILE");

    if !cfg!(feature = "rocksdb-store") {
        return;
    }

    let Some(lib_dir) = std::env::var_os("ROCKSDB_LIB_DIR") else {
        return;
    };

    // librocksdb-sys checks ROCKSDB_COMPILE before ROCKSDB_LIB_DIR, so this
    // means the sources are built and the flag comes from there. It is read
    // the same way librocksdb-sys reads it, and read before the archive is
    // looked for: the staging script sets it to build the very archive that is
    // still missing on a fresh clone.
    if let Ok(compile) = std::env::var("ROCKSDB_COMPILE") {
        if compile.to_lowercase() == "true" || compile == "1" {
            return;
        }
    }

    // .cargo/config.toml sets ROCKSDB_LIB_DIR for everyone, but the archive it
    // names is a build artifact and is not in the repository. Catch a missing
    // one here: librocksdb-sys trusts the variable and skips the source build,
    // so the alternative is a bare `cannot find -lrocksdb` at link time.
    if std::env::var_os("ROCKSDB_STATIC").is_some() {
        let archive = Path::new(&lib_dir).join("librocksdb.a");
        if !archive.exists() {
            panic!(
                "no prebuilt RocksDB at {}; run scripts/stage-rocksdb-lib.sh to build and \
                 stage one, or build with ROCKSDB_COMPILE=1 to compile the vendored sources",
                archive.display()
            );
        }
    }

    // Mirrors the stdlib librocksdb-sys picks for the same target.
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    match target_os.as_str() {
        "linux" => println!("cargo:rustc-link-lib=stdc++"),
        "windows" => {}
        _ => println!("cargo:rustc-link-lib=c++"),
    }
}
