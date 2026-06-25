#!/usr/bin/env python3
"""make_chunks.py — build VALID N-image COLMAP sub-scenes for the benchmark ladder.

Why this exists: OpenSplat's COLMAP reader (src/io/colmap.cpp) loads *every* image listed in
sparse/0/images.bin. You therefore cannot make an "8-image" test by copying 8 image files —
OpenSplat would fail on the 200+ images still referenced by the sparse model. A valid chunk
must contain a sparse model that references exactly the images present. This script does that
with pycolmap (the COLMAP Python API): it keeps the first N registered images, drops the rest
from the model, writes a fresh sparse/0, and copies only those N image files.

Usage:
    scripts/make_chunks.py <scene_dir> [--chunks 2 4 8 16 32 64] [--out data/chunks]

    <scene_dir> is a COLMAP project (e.g. data/db/drjohnson) with images/ and sparse/0/.

Output: <out>/<scene>_<N>/{images/, sparse/0/} for each N — each a valid OpenSplat project.

Requires: pip install pycolmap   (NOTE: pycolmap APIs vary by version; this targets the
modern Reconstruction API. Verify on first run — this has not been executed in CI yet.)
"""
import argparse
import shutil
import sys
from pathlib import Path

try:
    import pycolmap
except ImportError:
    sys.exit("ERROR: pycolmap is required.  pip install pycolmap")


def find_sparse(scene: Path) -> Path:
    for cand in (scene / "sparse" / "0", scene / "sparse", scene):
        if (cand / "images.bin").exists() or (cand / "images.txt").exists():
            return cand
    sys.exit(f"ERROR: no COLMAP model (images.bin) found under {scene}")


def make_chunk(scene: Path, sparse: Path, images_dir: Path, n: int, out_root: Path) -> None:
    rec = pycolmap.Reconstruction(str(sparse))
    # Deterministic order: by image name.
    ordered = sorted(rec.images.items(), key=lambda kv: kv[1].name)
    if n > len(ordered):
        print(f"  skip N={n}: scene has only {len(ordered)} images")
        return
    keep_ids = {iid for iid, _ in ordered[:n]}
    keep_names = [img.name for iid, img in ordered[:n]]

    for iid, _ in ordered:
        if iid not in keep_ids:
            rec.deregister_image(iid)

    dst = out_root / f"{scene.name}_{n}"
    (dst / "sparse" / "0").mkdir(parents=True, exist_ok=True)
    (dst / "images").mkdir(parents=True, exist_ok=True)
    rec.write(str(dst / "sparse" / "0"))
    for name in keep_names:
        src = images_dir / name
        if not src.exists():
            sys.exit(f"ERROR: image referenced by model not found on disk: {src}")
        out_img = dst / "images" / name
        if out_img.parent != dst / "images":   # only for names that contain a subpath
            out_img.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, out_img)
    print(f"  wrote {dst}  ({n} images)")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build valid N-image COLMAP sub-scenes.")
    ap.add_argument("scene_dir", type=Path)
    ap.add_argument("--chunks", type=int, nargs="+", default=[2, 4, 8, 16, 32, 64])
    ap.add_argument("--out", type=Path, default=Path("data/chunks"))
    args = ap.parse_args()

    scene = args.scene_dir.resolve()
    sparse = find_sparse(scene)
    images_dir = scene / "images"
    if not images_dir.is_dir():
        sys.exit(f"ERROR: {images_dir} not found")

    args.out.mkdir(parents=True, exist_ok=True)
    print(f"Scene: {scene}  (model: {sparse})")
    for n in sorted(args.chunks):
        make_chunk(scene, sparse, images_dir, n, args.out)
    print("Done. Each chunk is a valid OpenSplat project (images/ + sparse/0/).")


if __name__ == "__main__":
    main()
