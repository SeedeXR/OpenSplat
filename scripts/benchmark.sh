#!/usr/bin/env bash
# benchmark.sh — run opensplat on a project and capture resource metrics to memory/profiles/.
# Captures: runtime, peak RAM, backend used. (CPU%/GPU%/thermal need powermetrics/sudo and are
# left as TODO.)
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
# memory/profiles/ (and splat_output/) are generated here on first run if not already present.
mkdir -p "$REPO_ROOT/memory/profiles" "$REPO_ROOT/splat_output"

STAMP="$(date +%Y%m%d-%H%M%S)"
NIMG="$(ls -1 "$PROJECT/images" 2>/dev/null | wc -l | tr -d ' ')"
OUT="$REPO_ROOT/memory/profiles/bench_${LABEL}_n${ITERS}_${STAMP}.yaml"
ERRLOG="$(mktemp)"; OUTLOG="$(mktemp)"
trap 'rm -f "$ERRLOG" "$OUTLOG"' EXIT

# Optional held-out quality gate: set BENCH_VAL=<image-filename-in-the-scene> to withhold that
# camera and capture a validation loss (a quality number for before/after A/B comparisons).
VAL_ARGS=()
if [ -n "${BENCH_VAL:-}" ]; then VAL_ARGS=(--val --val-image "$BENCH_VAL"); fi

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
    ${VAL_ARGS[@]+"${VAL_ARGS[@]}"} \
    >"$OUTLOG" 2>"$ERRLOG" || { echo "ERROR: opensplat run failed:" >&2; tail -5 "$ERRLOG" >&2; exit 1; }
WALL=$((SECONDS - START))

# Peak RSS: macOS `time -l` reports bytes; GNU `time -v` reports kbytes.
PEAK_BYTES="$(grep -Eo '[0-9]+ +maximum resident set size' "$ERRLOG" | grep -Eo '^[0-9]+' || true)"
if [[ -n "$PEAK_BYTES" ]]; then PEAK_MB=$((PEAK_BYTES / 1024 / 1024))
else
  PEAK_KB="$(grep -Eo 'Maximum resident set size \(kbytes\): [0-9]+' "$ERRLOG" | grep -Eo '[0-9]+$' || echo 0)"
  PEAK_MB=$((PEAK_KB / 1024))
fi
# macOS also reports "peak memory footprint" (phys_footprint) — for the Metal/unified-memory
# backend this is the metric that matters (RSS badly undercounts GPU allocations). 0 on Linux.
FOOT_BYTES="$(grep -Eo '[0-9]+ +peak memory footprint' "$ERRLOG" | grep -Eo '^[0-9]+' || echo 0)"
FOOT_MB=$((FOOT_BYTES / 1024 / 1024))
# opensplat prints "Using <backend>" to stdout; search both streams to be safe.
BACKEND="$(grep -hEo 'Using (CUDA|MPS|CPU)' "$OUTLOG" "$ERRLOG" 2>/dev/null | head -1 | awk '{print $2}')"
[ -n "$BACKEND" ] || BACKEND=unknown
# Real GPU memory from opensplat's own MPS diagnostic ("MPS memory: current ... driver ...").
# This is the TRUE budget metric — phys_footprint over-counts transient Metal/MPSGraph scratch.
MPS_CURR="$(grep -oE 'current allocated [0-9]+ MB' "$OUTLOG" 2>/dev/null | grep -oE '[0-9]+' | head -1 || echo 0)"
MPS_DRIVER="$(grep -oE 'driver allocated [0-9]+ MB' "$OUTLOG" 2>/dev/null | grep -oE '[0-9]+' | head -1 || echo 0)"
# Held-out validation loss (quality), if BENCH_VAL was set. opensplat prints "... validation loss: X".
VAL_LOSS="$(grep -oE 'validation loss: [0-9.]+' "$OUTLOG" 2>/dev/null | grep -oE '[0-9.]+$' | head -1 || true)"

cat > "$OUT" <<YAML
# OpenSplat benchmark capture
label: ${LABEL}
project: ${PROJECT}
images: ${NIMG}
iterations: ${ITERS}
backend: ${BACKEND}
runtime_seconds: ${WALL}
validation_loss: ${VAL_LOSS:-}          # held-out quality (only when BENCH_VAL set); lower is better
# --- memory: prefer the real GPU figures below over the OS metrics ---
mps_driver_allocated_mb: ${MPS_DRIVER}   # REAL GPU memory reserved (the budget metric): <= 8192 (ideal 4096-6144)
mps_current_allocated_mb: ${MPS_CURR}    # live MPS tensors
peak_ram_mb: ${PEAK_MB}                  # process RSS (CPU side)
peak_footprint_mb: ${FOOT_MB}            # phys_footprint — OS peak; OVER-COUNTS transient Metal scratch, not real RAM
# TODO (need powermetrics/sudo on Apple Silicon):
average_ram_mb:
cpu_usage_pct:
gpu_usage_pct:
io_wait:
thermal_state:
timestamp: ${STAMP}
YAML

echo ">> $OUT"
echo "   backend=$BACKEND  runtime=${WALL}s  mps_driver=${MPS_DRIVER}MB  rss=${PEAK_MB}MB  (footprint=${FOOT_MB}MB, over-counts)  images=$NIMG"
# Budget against the REAL GPU metric (mps_driver) when available, else RSS.
BUDGET_MB=$([ "$MPS_DRIVER" -gt 0 ] && echo "$MPS_DRIVER" || echo "$PEAK_MB")
[[ "$BUDGET_MB" -gt 8192 ]] && echo "   WARNING: real memory ${BUDGET_MB}MB exceeds 8 GB cap" >&2 || true
