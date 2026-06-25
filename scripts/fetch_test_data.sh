#!/usr/bin/env bash
# fetch_test_data.sh — download a COLMAP test scene from the project dataset on Hugging Face:
#   https://huggingface.co/datasets/alexmkwizu/gaussian_training_datasets
# Each scene has images/ + sparse/0/{cameras,images,points3D}.bin (what OpenSplat needs).
#
# Usage:
#   scripts/fetch_test_data.sh [scene] [dest]
#
#   scene (default: db/drjohnson). Available:
#     db/drjohnson  db/playroom
#     mipnerf360/bicycle  mipnerf360/bonsai  mipnerf360/counter  mipnerf360/garden
#     mipnerf360/kitchen  mipnerf360/room  mipnerf360/stump
#     tandt/train  tandt/truck
#   dest  (default: <repo>/data)
#
# Then train:  ./output/opensplat data/db/drjohnson -n 2000 -o splat_output/drjohnson.ply
# Requires the `hf` CLI (pip install -U huggingface_hub  /  brew install huggingface-cli).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ID="alexmkwizu/gaussian_training_datasets"
SCENE="${1:-db/drjohnson}"
DEST="${2:-$REPO_ROOT/data}"

command -v hf >/dev/null || { echo "ERROR: 'hf' CLI not found. pip install -U huggingface_hub" >&2; exit 1; }

echo ">> Downloading '$SCENE' from $REPO_ID into $DEST ..."
hf download "$REPO_ID" --type dataset --include "${SCENE}/**" --local-dir "$DEST"

PROJECT="$DEST/$SCENE"
if [[ -d "$PROJECT/images" && -d "$PROJECT/sparse" ]]; then
  N="$(ls -1 "$PROJECT/images" 2>/dev/null | wc -l | tr -d ' ')"
  echo ">> Ready: $PROJECT  (images: $N)"
  echo "   Train:  ./output/opensplat \"$PROJECT\" -n 2000 -o splat_output/$(basename "$SCENE").ply"
  echo "   Chunks: scripts/make_chunks.py \"$PROJECT\"   # valid 2/4/8/16/32/64-image sub-scenes"
else
  echo ">> Downloaded to $PROJECT (verify it contains images/ and sparse/)." >&2
fi
