#!/bin/sh
# Builds the WebGPU browser version into target/web (index.html + wasm + JS glue).
# Needs the wasm32-unknown-unknown std (Arch: rust-wasm) and wasm-bindgen-cli matching
# the wasm-bindgen version in Cargo.lock.
set -eu
cd "$(dirname "$0")/.."
out=target/www
cargo build --locked --profile web --target wasm32-unknown-unknown --bin rider-rep-rust
wasm-bindgen --target web --no-typescript --out-dir "$out" \
  target/wasm32-unknown-unknown/web/rider-rep-rust.wasm
cp web/index.html "$out/"
echo "Built $out"
