#!/usr/bin/env python3
"""make_chunks.py — build VALID N-image COLMAP sub-scenes for the benchmark ladder.

Why this exists: OpenSplat's COLMAP reader (src/io/colmap.cpp) loads *every* image listed in
sparse/0/images.bin. You therefore cannot make an "8-image" test by copying 8 image files —
OpenSplat would fail on the 200+ images still referenced by the model. A valid chunk needs a
sparse model that references exactly the images present.

This is a DEPENDENCY-FREE implementation: it parses sparse/0/images.bin directly (the documented
COLMAP binary format the OpenSplat reader expects), keeps the first N image records, and copies
those N image files. cameras.bin and points3D.bin are copied verbatim — extra cameras/points are
harmless to OpenSplat (its reader keys cameras by id and reads only xyz+rgb from points3D).

Usage:
    scripts/make_chunks.py <scene_dir> [--chunks 2 4 8 16 32 64] [--out datasets/chunks]

    <scene_dir> is a COLMAP project (e.g. datasets/tandt/truck) with images/ and sparse/0/.

Output: <out>/<scene>_<N>/{images/, sparse/0/} for each N — each a valid OpenSplat project.
"""
import argparse
import shutil
import struct
import sys
from pathlib import Path


def find_sparse(scene: Path) -> Path:
    for cand in (scene / "sparse" / "0", scene / "sparse", scene):
        if (cand / "images.bin").exists():
            return cand
    sys.exit(f"ERROR: no COLMAP images.bin found under {scene}")


def parse_image_records(data: bytes):
    """Yield (name, raw_record_bytes) for each image in an images.bin blob.

    Layout (little-endian): uint64 num_images, then per image:
      uint32 image_id, 4*double qvec, 3*double tvec, uint32 camera_id,
      char* name (NUL-terminated), uint64 num_points2D, num_points2D * (double x, double y, uint64 pt3D_id)
    """
    num_images = struct.unpack_from("<Q", data, 0)[0]
    off = 8
    for _ in range(num_images):
        start = off
        off += 4 + 8 * 4 + 8 * 3 + 4          # image_id, qvec, tvec, camera_id
        name_end = data.index(b"\x00", off)    # NUL-terminated name
        name = data[off:name_end].decode("utf-8", "replace")
        off = name_end + 1
        num_pts = struct.unpack_from("<Q", data, off)[0]
        off += 8 + num_pts * (8 + 8 + 8)       # each point2D: x, y, pt3D_id
        yield name, data[start:off]


def make_chunk(scene: Path, sparse: Path, images_dir: Path, n: int, out_root: Path) -> None:
    records = list(parse_image_records((sparse / "images.bin").read_bytes()))
    if n > len(records):
        print(f"  skip N={n}: scene has only {len(records)} images")
        return
    chosen = sorted(records, key=lambda r: r[0])[:n]   # deterministic: first N by name

    dst = out_root / f"{scene.name}_{n}"
    (dst / "sparse" / "0").mkdir(parents=True, exist_ok=True)
    (dst / "images").mkdir(parents=True, exist_ok=True)

    # New images.bin: header(N) + the N chosen records verbatim.
    blob = struct.pack("<Q", n) + b"".join(rec for _, rec in chosen)
    (dst / "sparse" / "0" / "images.bin").write_bytes(blob)
    # cameras.bin / points3D.bin copied as-is.
    shutil.copy2(sparse / "cameras.bin", dst / "sparse" / "0" / "cameras.bin")
    shutil.copy2(sparse / "points3D.bin", dst / "sparse" / "0" / "points3D.bin")

    for name, _ in chosen:
        src = images_dir / name
        if not src.exists():
            sys.exit(f"ERROR: image referenced by model not found on disk: {src}")
        out_img = dst / "images" / name
        if out_img.parent != dst / "images":     # only for names that contain a subpath
            out_img.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, out_img)
    print(f"  wrote {dst}  ({n} images)")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build valid N-image COLMAP sub-scenes (no deps).")
    ap.add_argument("scene_dir", type=Path)
    ap.add_argument("--chunks", type=int, nargs="+", default=[2, 4, 8, 16, 32, 64])
    ap.add_argument("--out", type=Path, default=Path("datasets/chunks"))
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
