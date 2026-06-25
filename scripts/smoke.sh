#!/usr/bin/env bash
# smoke.sh — minimal end-to-end sanity check: run a short training on a small dataset
# and confirm OpenSplat produces an output splat. See docs/testing.md.
#
# Usage:
#   scripts/smoke.sh /path/to/dataset [iterations] [bin-dir]
#
# bin-dir defaults to "output" (where scripts/build.sh emits the binary). Pass "build" if you
# built with a plain `cmake .. && make` instead.
#
# Example (after scripts/build.sh, with a dataset downloaded):
#   scripts/smoke.sh data/db/drjohnson 50
#
# Exit code is non-zero if the binary is missing, the run fails, or no output is produced.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET="${1:-}"
ITERS="${2:-50}"
BIN_DIR="${3:-output}"
BIN="$REPO_ROOT/$BIN_DIR/opensplat"

if [[ -z "$DATASET" ]]; then
  echo "usage: scripts/smoke.sh /path/to/dataset [iterations] [bin-dir]" >&2
  echo "  Fetch a dataset first (scripts/fetch_test_data.sh) and build (scripts/build.sh)." >&2
  exit 2
fi
[[ -x "$BIN" ]] || { echo "ERROR: $BIN not found. Build first: scripts/build.sh --libtorch <path> (or pass the bin-dir)." >&2; exit 1; }
[[ -d "$DATASET" ]] || { echo "ERROR: dataset dir not found: $DATASET" >&2; exit 1; }

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
echo ">> Smoke training: $ITERS iters on $DATASET"
( cd "$WORK" && "$BIN" "$DATASET" -n "$ITERS" )

if ls "$WORK"/*.ply "$WORK"/*.splat >/dev/null 2>&1; then
  echo ">> PASS: output produced:"; ls -1 "$WORK"/*.ply "$WORK"/*.splat 2>/dev/null
else
  echo ">> FAIL: no .ply/.splat output produced" >&2; exit 1
fi
