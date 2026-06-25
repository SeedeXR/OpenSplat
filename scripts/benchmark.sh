#!/usr/bin/env bash
# benchmark.sh — run opensplat on a project and capture resource metrics to memory/profiles/.
# Captures: runtime, peak RAM, backend used. (CPU%/GPU%/thermal need powermetrics/sudo and are
# left as TODO — see memory/operating/agent_governance.md for the full metric schema.)
#
# Usage:
#   scripts/benchmark.sh <project_dir> [iters] [label]
# Example (after fetch_test_data.sh + make_chunks.py):
#   for d in data/chunks/drjohnson_{2,4,8,16,32,64}; do scripts/benchmark.sh "$d" 2000; done
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT="${1:?usage: benchmark.sh <project_dir> [iters] [label]}"
ITERS="${2:-2000}"
LABEL="${3:-$(basename "$PROJECT")}"
BIN="$REPO_ROOT/output/opensplat"

[[ -x "$BIN" ]] || { echo "ERROR: $BIN not found. Build first (scripts/build.sh)." >&2; exit 1; }
[[ -d "$PROJECT" ]] || { echo "ERROR: project dir not found: $PROJECT" >&2; exit 1; }
mkdir -p "$REPO_ROOT/memory/profiles" "$REPO_ROOT/splat_output"

STAMP="$(date +%Y%m%d-%H%M%S)"
NIMG="$(ls -1 "$PROJECT/images" 2>/dev/null | wc -l | tr -d ' ')"
OUT="$REPO_ROOT/memory/profiles/bench_${LABEL}_n${ITERS}_${STAMP}.yaml"
ERRLOG="$(mktemp)"; OUTLOG="$(mktemp)"
trap 'rm -f "$ERRLOG" "$OUTLOG"' EXIT

echo ">> Benchmarking $LABEL ($NIMG images, $ITERS iters) ..."
# /usr/bin/time captures peak RSS but its flags differ: macOS/BSD use -l, GNU (Linux) uses -v.
# Fall back to running the binary directly (no RSS) if /usr/bin/time is unavailable.
TIME_CMD=()
if [ -x /usr/bin/time ]; then
  case "$(uname -s)" in
    Darwin) TIME_CMD=(/usr/bin/time -l) ;;
    *)      TIME_CMD=(/usr/bin/time -v) ;;
  esac
fi
START=$SECONDS
${TIME_CMD[@]+"${TIME_CMD[@]}"} "$BIN" "$PROJECT" -n "$ITERS" -o "$REPO_ROOT/splat_output/${LABEL}.ply" \
    >"$OUTLOG" 2>"$ERRLOG" || { echo "ERROR: opensplat run failed:" >&2; tail -5 "$ERRLOG" >&2; exit 1; }
WALL=$((SECONDS - START))

# Peak RSS: macOS `time -l` reports bytes; GNU `time -v` reports kbytes.
PEAK_BYTES="$(grep -Eo '[0-9]+ +maximum resident set size' "$ERRLOG" | grep -Eo '^[0-9]+' || true)"
if [[ -n "$PEAK_BYTES" ]]; then PEAK_MB=$((PEAK_BYTES / 1024 / 1024))
else
  PEAK_KB="$(grep -Eo 'Maximum resident set size \(kbytes\): [0-9]+' "$ERRLOG" | grep -Eo '[0-9]+$' || echo 0)"
  PEAK_MB=$((PEAK_KB / 1024))
fi
# opensplat prints "Using <backend>" to stdout; search both streams to be safe.
BACKEND="$(grep -hEo 'Using (CUDA|MPS|CPU)' "$OUTLOG" "$ERRLOG" 2>/dev/null | head -1 | awk '{print $2}')"
[ -n "$BACKEND" ] || BACKEND=unknown

cat > "$OUT" <<YAML
# OpenSplat benchmark capture
label: ${LABEL}
project: ${PROJECT}
images: ${NIMG}
iterations: ${ITERS}
backend: ${BACKEND}
runtime_seconds: ${WALL}
peak_ram_mb: ${PEAK_MB}        # target: <= 8192 (ideal 4096-6144) — see todo.md
# TODO (need powermetrics/sudo on Apple Silicon):
average_ram_mb:
cpu_usage_pct:
gpu_usage_pct:
io_wait:
thermal_state:
timestamp: ${STAMP}
YAML

echo ">> $OUT"
echo "   backend=$BACKEND  runtime=${WALL}s  peak_ram=${PEAK_MB}MB  images=$NIMG"
[[ "$PEAK_MB" -gt 8192 ]] && echo "   WARNING: peak RAM ${PEAK_MB}MB exceeds 8 GB cap" >&2 || true
