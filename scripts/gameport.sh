#!/usr/bin/env bash
set -euo pipefail
project="$(dirname "$(dirname "$(realpath "${BASH_SOURCE[0]}")")")"
toolkit="${GAMEPORT2RUST_ROOT:-$project/.tools/GamePort2Rust}"
if [[ "${1:-help}" == install && ! -f "$toolkit/scripts/query.sh" ]]; then
    toolkit="${GAMEPORT2RUST_ROOT:-$project/../GamePort2Rust}"
fi
if [[ ! -f "$toolkit/scripts/query.sh" ]]; then
    printf 'GamePort2Rust not found: %s\nRun bash scripts/gameport.sh install or set GAMEPORT2RUST_ROOT.\n' "$toolkit" >&2
    exit 1
fi
case "${1:-help}" in
    install)
        if [[ ! -x "$toolkit/tools/node_modules/.bin/rea" || ! -f "$toolkit/.local/ghidra-dir" || ! -f "$toolkit/.local/java-home" ]]; then
            bash "$toolkit/scripts/setup-tools.sh"
        fi
        mkdir -p "$project/.tools"
        ln -sfnT "$(realpath "$toolkit")" "$project/.tools/GamePort2Rust"
        printf 'Installed local GamePort2Rust link: %s\n' "$project/.tools/GamePort2Rust"
        ;;
    query)
        shift
        exec bash "$toolkit/scripts/query.sh" "$@"
        ;;
    rea)
        shift
        exec bash "$toolkit/scripts/rea.sh" "$@"
        ;;
    help|--help|-h)
        printf 'Usage: bash scripts/gameport.sh install | query BINARY PROCEDURE [--fresh] | rea COMMAND...\n'
        ;;
    *)
        printf 'Unknown command: %s\n' "$1" >&2
        exit 2
        ;;
esac
