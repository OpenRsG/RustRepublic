#!/bin/sh
# Builds the WebGPU browser version into target/www (index.html + wasm + JS glue).
# Needs the wasm32-unknown-unknown std (Arch: rust-wasm) and wasm-bindgen-cli matching
# the wasm-bindgen version in Cargo.lock. wasm-opt (binaryen, on PATH or unpacked under
# .tools/binaryen/) is used when present.
set -eu
cd "$(dirname "$0")/.."
for d in .tools/binaryen/*/bin; do if [ -d "$d" ]; then PATH="$PWD/$d:$PATH"; fi; done
out=target/www
cargo build --locked --profile web --target wasm32-unknown-unknown --bin rider-rep-rust
wasm-bindgen --target web --no-typescript --out-dir "$out" \
  target/wasm32-unknown-unknown/web/rider-rep-rust.wasm
wasm="$out/rider-rep-rust_bg.wasm"
if command -v wasm-opt >/dev/null; then
  wasm-opt -O3 --enable-bulk-memory --enable-nontrapping-float-to-int --enable-sign-ext \
    --enable-mutable-globals --enable-reference-types --enable-multivalue -o "$wasm" "$wasm"
fi
# The page fetches this copy and unpacks it in the browser (DecompressionStream).
gzip -9 -k -f "$wasm"
cp web/index.html "$out/"
echo "Built $out"
