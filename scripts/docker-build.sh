#!/usr/bin/env bash
# docker-build.sh — build an OpenSplat Docker image, selecting a Dockerfile variant.
# The Dockerfiles live in the repo root (build context = repo root); this wraps the
# `docker build [-f Dockerfile.<variant>] .` commands documented in README.md.
#
# Usage:
#   scripts/docker-build.sh [variant] [-t tag] [-- extra docker build args...]
#
#   variant: cuda (default) | rocm | rocm6 | rocm6.3.3 | rocm6.4.0 | rocm7
#
# Examples:
#   scripts/docker-build.sh                          # CUDA, tag opensplat
#   scripts/docker-build.sh rocm6 -t opensplat:rocm6
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VARIANT="${1:-cuda}"; [[ $# -gt 0 ]] && shift || true
TAG="opensplat"
EXTRA=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    -t) TAG="$2"; shift 2;;
    --) shift; EXTRA+=("$@"); break;;
    *)  EXTRA+=("$1"); shift;;
  esac
done

case "$VARIANT" in
  cuda)       DOCKERFILE="Dockerfile";;
  rocm)       DOCKERFILE="Dockerfile.rocm";;
  rocm6)      DOCKERFILE="Dockerfile.rocm6";;
  rocm6.3.3)  DOCKERFILE="Dockerfile.rocm6.3.3";;
  rocm6.4.0)  DOCKERFILE="Dockerfile.rocm6.4.0";;
  rocm7)      DOCKERFILE="Dockerfile.rocm7";;
  *) echo "ERROR: unknown variant '$VARIANT'" >&2; sed -n '9,12p' "$0"; exit 2;;
esac

[[ -f "$REPO_ROOT/$DOCKERFILE" ]] || { echo "ERROR: $DOCKERFILE not found in repo root" >&2; exit 1; }
echo ">> docker build -t $TAG -f $DOCKERFILE ."
# ${EXTRA[@]+...} avoids injecting a stray empty arg when EXTRA is empty (set -u safe).
docker build -t "$TAG" -f "$REPO_ROOT/$DOCKERFILE" ${EXTRA[@]+"${EXTRA[@]}"} "$REPO_ROOT"
