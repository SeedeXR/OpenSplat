# scripts/

Helper scripts that wrap the build/run/Docker commands documented in the root `../README.md`.
They are conveniences — every step they run is also documented in [`../docs/getting_started.md`](../docs/getting_started.md)
and [`../docs/testing.md`](../docs/testing.md).

| Script | Purpose |
| ------ | ------- |
| `build.sh` | Configure + build OpenSplat for a chosen backend (CPU/CUDA/HIP/MPS). Binaries → `output/`. |
| `fetch_test_data.sh` | Download a COLMAP test scene from the project's Hugging Face dataset into `data/`. |
| `make_chunks.py` | Build **valid** 2/4/8/16/32/64-image COLMAP sub-scenes (pycolmap) for the benchmark ladder. |
| `smoke.sh` | End-to-end sanity check: train a few iterations on a dataset and confirm output. |
| `benchmark.sh` | Run a scene and capture runtime / peak RAM / backend to `memory/profiles/`. |
| `docker-build.sh` | Build a Docker image, selecting a root `Dockerfile` variant (CUDA / ROCm*). |

```bash
# Build (Apple Metal), fetch data, smoke-test, then benchmark the chunk ladder:
scripts/build.sh --libtorch ~/libtorch --backend MPS
scripts/fetch_test_data.sh db/drjohnson                 # -> data/db/drjohnson
scripts/smoke.sh data/db/drjohnson 50
scripts/make_chunks.py data/db/drjohnson                # -> data/chunks/drjohnson_{2..64}
for d in data/chunks/drjohnson_{2,4,8,16,32,64}; do scripts/benchmark.sh "$d" 2000; done
```

Conventions: binaries → `output/`, trained splats → `splat_output/`, datasets → `data/`
(all git-ignored). Pass `-o splat_output/<name>.ply` when running `opensplat` directly.

Notes:
- Dockerfiles intentionally live in the **repo root** (so `docker build .` works and matches
  CI `.github/workflows/docker.yml`); `docker-build.sh` just selects the `-f` variant.
- Scripts make no assumptions about installed backends — pass `--backend`/`--libtorch` explicitly.
