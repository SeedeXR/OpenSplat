# Getting Started

How to build and run OpenSplat. The root [`../README.md`](https://github.com/SeedeXR/OpenSplat/blob/main/README.md) is the canonical,
fullest source (per-OS specifics, dataset links, Docker matrix); this page is the quick path
and the dev workflow. The [`../scripts/`](https://github.com/SeedeXR/OpenSplat/tree/main/scripts) helpers wrap these commands.

## Prerequisites

- **CMake** ≥ 3.21, a **C++17** toolchain.
- **OpenCV** (`sudo apt install libopencv-dev`, or `brew install opencv`).
- **LibTorch** — download from <https://pytorch.org/get-started/locally/> (select *LibTorch*;
  match your CUDA/ROCm version if using a GPU).
- Backend extras: **CUDA** (`nvcc`), **ROCm** at `/opt/rocm`, or **Xcode CLT** for Apple Metal.

## Build & run flow

```mermaid
flowchart LR
    DL["download LibTorch + dataset"] --> CFG["cmake configure<br/>-DCMAKE_PREFIX_PATH, -DGPU_RUNTIME"]
    CFG --> BLD["cmake --build (make -j)"]
    BLD --> RUN["output/opensplat &lt;dataset&gt; -n N<br/>-o splat_output/x.ply"]
    RUN --> OUT["splat_output/x.ply + cameras.json"]
```

> `scripts/build.sh` emits binaries to **`output/`** (via `-DCMAKE_RUNTIME_OUTPUT_DIRECTORY`);
> a plain `cmake … && make` (the root README flow) puts them in `build/`. Trained splats go to
> **`splat_output/`** (run with `-o splat_output/<name>.ply`). `output/`, `splat_output/`, and
> `data/` are git-ignored.

## Build (quick path)

```bash
git clone https://github.com/pierotofy/OpenSplat OpenSplat
cd OpenSplat
mkdir build && cd build
cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch/ .. && make -j$(nproc)   # Linux
```

Backend selection via `-DGPU_RUNTIME=`:

| Platform | Configure flags |
| -------- | --------------- |
| CPU (portable, ~100× slower) | *(default — omit `GPU_RUNTIME`)* |
| NVIDIA CUDA | *(auto-detected when CUDA toolkit present)* |
| AMD ROCm/HIP | `-DGPU_RUNTIME=HIP -DHIP_ROOT_DIR=/opt/rocm` |
| Apple Metal | `-DGPU_RUNTIME=MPS` |

Or use the helper (runs the same cmake commands):

```bash
scripts/build.sh --libtorch /path/to/libtorch --backend MPS   # or CPU / CUDA / HIP
```

Useful options: `-DOPENSPLAT_BUILD_SIMPLE_TRAINER=ON`, `-DOPENSPLAT_BUILD_VISUALIZER=ON`,
`-DOPENSPLAT_USE_FAST_MATH=ON`. macOS/Windows/Docker specifics: see [`../README.md`](https://github.com/SeedeXR/OpenSplat/blob/main/README.md).

On **Linux** the build also generates the `opensplat(1)` man page (`build/opensplat.1.gz`);
`sudo make install` (or `cmake --install build`) places it on the `man` path so `man opensplat`
works. Disable with `-DOPENSPLAT_BUILD_MANPAGE=OFF`.

## Get a dataset

```bash
scripts/fetch_test_data.sh db/drjohnson   # COLMAP scene -> data/db/drjohnson
```

Pulls from the project's [Hugging Face dataset](https://huggingface.co/datasets/alexmkwizu/gaussian_training_datasets)
(or download `banana`/`truck` per [`../README.md`](https://github.com/SeedeXR/OpenSplat/blob/main/README.md#run)). Input must include sparse
points (random init is unsupported).

## Run

```bash
output/opensplat data/db/drjohnson -n 2000 -o splat_output/drjohnson.ply
output/opensplat --help                    # full option list
man opensplat                              # full manual (installed on Linux)
```

Output: a `.ply`/`.splat` + `cameras.json` (written next to the `-o` path), droppable into a
[splat viewer](https://github.com/MrNeRF/awesome-3D-gaussian-splatting#viewers). Resume with
`--resume splat_output/drjohnson.ply`.

### Resource-aware run

OpenSplat sizes itself to the host: at startup it detects RAM/VRAM/CPU and prints a
**process contract** choosing the host image store, a VRAM-bounded gaussian cap, and any
downscale. Override with (precedence high→low) a CLI flag, an `OPENSPLAT_*` env var, a
`--contract file.json`, or a `--profile` preset:

```bash
output/opensplat data/db/drjohnson --profile 4gb --max-splats 1000000   # fit a 4 GB box, bound VRAM
output/opensplat data/db/drjohnson --image-store u8                     # ~4x less host RAM, bit-identical
```

`man opensplat` documents every flag, the JSON contract schema, and the `OPENSPLAT_*`
variables.

## Verify it works

```bash
scripts/smoke.sh data/db/drjohnson 50      # short train + confirms output is produced
```

See [`testing.md`](testing.md) for the full verification approach. On a 16 GB Apple Silicon
machine, prefer CPU or MPS; a single Gaussian is ~2000 bytes (~2 GB GPU memory per million).
