#!/usr/bin/env bash
# Stage a prebuilt librocksdb.a so that `librocksdb-sys` links it instead of
# recompiling RocksDB's ~330 C++ translation units from source.
#
# Compiling the vendored sources takes minutes, and the `cc` crate keeps no
# object cache, so any change that invalidates the build script's fingerprint
# -- a different feature unification between `cargo build` and `cargo test`, a
# moved LIBCLANG_PATH, or a plain Ctrl-C -- pays the full cost again. Linking a
# prebuilt archive skips `build_rocksdb()` entirely and leaves only bindgen,
# which runs in seconds.
#
# `.cargo/config.toml` points ROCKSDB_LIB_DIR at the directory written here.
#
# Re-run this after bumping the `rocksdb` crate. bindgen always generates
# against the headers vendored in the new librocksdb-sys, so a stale archive
# would be linked against bindings for a different RocksDB -- the version stamp
# below exists to make that mismatch loud instead of silent.
set -euo pipefail

cd "$(dirname "$0")/.."

lib_dir=".cargo/rocksdb"
stamp="$lib_dir/version"

# The version whose vendored headers bindgen will generate against.
want=$(awk '
    /^name = "librocksdb-sys"$/ { found = 1; next }
    found && /^version = / { gsub(/[",]/, "", $3); print $3; exit }
' Cargo.lock)

if [ -z "$want" ]; then
    echo "stage-rocksdb-lib: no librocksdb-sys entry in Cargo.lock" >&2
    exit 1
fi

if [ "${1:-}" != "--force" ] &&
   [ -f "$lib_dir/librocksdb.a" ] &&
   [ -f "$stamp" ] &&
   [ "$(cat "$stamp")" = "$want" ]; then
    echo "stage-rocksdb-lib: librocksdb-sys $want already staged; --force to rebuild"
    exit 0
fi

# bindgen needs libclang, and .cargo/config.toml pins LIBCLANG_PATH at a
# symlink under .cargo/clanglink so the path is identical in every shell: a
# LIBCLANG_PATH that comes and goes reruns bindgen's build script, and with it
# the RocksDB one. Neither the symlink nor its target is in the repository, so
# create it when it is missing.
clang_dir=".cargo/clanglink"
if [ ! -e "$clang_dir/libclang.so" ]; then
    libclang=$(ls -1 /usr/lib/llvm-*/lib/libclang.so.1 /usr/lib/*/libclang.so.1 \
        2>/dev/null | sort -V | tail -1)
    if [ -z "$libclang" ]; then
        echo "stage-rocksdb-lib: no libclang found; install libclang-dev" >&2
        exit 1
    fi
    mkdir -p "$clang_dir"
    ln -sfn "$libclang" "$clang_dir/libclang.so"
    echo "stage-rocksdb-lib: linked libclang from $libclang"
fi

# ROCKSDB_COMPILE=1 is what forces a source build: the build script checks it
# before ROCKSDB_LIB_DIR, so this still works once the config below is in
# place and the archive would otherwise just be linked.
#
# Built under --release so the archive is optimised and small. Debug builds
# link it just as happily, and an -O0 RocksDB is both enormous and slow.
echo "stage-rocksdb-lib: compiling librocksdb-sys $want from source (several minutes)"
ROCKSDB_COMPILE=1 cargo build --release -p galemind-store

src=$(ls -t target/release/build/librocksdb-sys-*/out/librocksdb.a 2>/dev/null | head -1)
if [ -z "$src" ]; then
    echo "stage-rocksdb-lib: build produced no librocksdb.a" >&2
    exit 1
fi

mkdir -p "$lib_dir"
cp "$src" "$lib_dir/librocksdb.a"
printf '%s\n' "$want" > "$stamp"

echo "stage-rocksdb-lib: staged $(du -h "$lib_dir/librocksdb.a" | cut -f1) archive for librocksdb-sys $want"
