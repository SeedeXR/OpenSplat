#!/usr/bin/env bash
# build.sh — configure + build OpenSplat.
# Wraps the cmake commands documented in README.md so a build is one command.
#
# Usage:
#   scripts/build.sh --libtorch /path/to/libtorch [--backend CPU|CUDA|HIP|MPS] [--build-dir build] [extra cmake args...]
#
# Examples:
#   scripts/build.sh --libtorch ~/libtorch                         # CPU
#   scripts/build.sh --libtorch ~/libtorch --backend MPS           # Apple Metal
#   scripts/build.sh --libtorch ~/libtorch --backend CUDA
#
# Requires: cmake, a C++17 toolchain, OpenCV, and LibTorch (see docs/getting_started.md).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LIBTORCH=""
BACKEND=""           # empty => CMake default (CPU unless CUDA toolkit found)
BUILD_DIR="build"
EXTRA=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --libtorch)  LIBTORCH="$2"; shift 2;;
    --backend)   BACKEND="$2"; shift 2;;
    --build-dir) BUILD_DIR="$2"; shift 2;;
    -h|--help)   sed -n '2,12p' "$0"; exit 0;;
    *)           EXTRA+=("$1"); shift;;
  esac
done

# Emit binaries to <repo>/output by default (overridable by passing your own -D in extra args,
# which is appended last and therefore wins).
CMAKE_ARGS=("-DCMAKE_RUNTIME_OUTPUT_DIRECTORY=$REPO_ROOT/output")
[[ -n "$LIBTORCH" ]] && CMAKE_ARGS+=("-DCMAKE_PREFIX_PATH=$LIBTORCH")
[[ -n "$BACKEND"  ]] && CMAKE_ARGS+=("-DGPU_RUNTIME=$BACKEND")
# Safe expansion of a possibly-empty array under `set -u` (no stray "" argument).
CMAKE_ARGS+=(${EXTRA[@]+"${EXTRA[@]}"})

JOBS="$( (command -v nproc >/dev/null && nproc) || sysctl -n hw.logicalcpu 2>/dev/null || echo 4 )"

echo ">> Configuring ($BUILD_DIR) ${BACKEND:+backend=$BACKEND} ..."
cmake -S "$REPO_ROOT" -B "$REPO_ROOT/$BUILD_DIR" "${CMAKE_ARGS[@]}"
echo ">> Building with -j$JOBS ..."
cmake --build "$REPO_ROOT/$BUILD_DIR" -j"$JOBS"
echo ">> Done. Binary: $REPO_ROOT/$BUILD_DIR/opensplat"
