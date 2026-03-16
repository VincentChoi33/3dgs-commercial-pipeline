#!/usr/bin/env python3
"""Experiment: Improved SfM with more keypoints, smart pairs, and undistortion.

Run on train2 with gsplat env:
  conda activate gsplat
  python scripts/experiment_sfm_improved.py
"""
from __future__ import annotations

import os
import shutil
import struct
import sys
import time
from pathlib import Path

# hloc is in gsplat env
sys.path.insert(0, "/data/choihy/GSplat/hloc")

import numpy as np

# ──────────────────── Config ────────────────────

SCENES = {
    "6027": Path("/data/choihy/GSplat/runs/2026-03-12/video_improved/office_4k_6027"),
    "6028": Path("/data/choihy/GSplat/runs/2026-03-12/video_improved/office_4k_6028"),
}
OUTPUT_BASE = Path("/data/choihy/GSplat/runs/2026-03-13/sfm_improved")


# ──────────────────── Sequential Pair Generator ────────────────────

def generate_sequential_pairs(image_list: list[str], output: Path, overlap: int = 10):
    """Generate pairs from sequential frames with window overlap."""
    pairs = set()
    n = len(image_list)
    for i in range(n):
        for j in range(i + 1, min(i + 1 + overlap, n)):
            pairs.add((image_list[i], image_list[j]))
        # Quadratic overlap: also match with frames at 2^k distance
        k = 1
        while True:
            idx = i + 2**k
            if idx >= n:
                break
            pairs.add((image_list[i], image_list[idx]))
            k += 1

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        for a, b in sorted(pairs):
            f.write(f"{a} {b}\n")

    print(f"  Sequential pairs: {len(pairs)}")
    return output


def merge_pair_files(files: list[Path], output: Path) -> Path:
    """Merge multiple pair files, deduplicating."""
    all_pairs = set()
    for p in files:
        with open(p) as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split()
                    if len(parts) == 2:
                        # Normalize order
                        a, b = sorted(parts)
                        all_pairs.add((a, b))
    with open(output, "w") as f:
        for a, b in sorted(all_pairs):
            f.write(f"{a} {b}\n")
    print(f"  Merged pairs: {len(all_pairs)}")
    return output


# ──────────────────── SfM Stats ────────────────────

def read_sfm_stats(sparse_dir: Path) -> dict:
    """Read basic stats from COLMAP binary model."""
    stats = {}
    images_bin = sparse_dir / "images.bin"
    points_bin = sparse_dir / "points3D.bin"
    cameras_bin = sparse_dir / "cameras.bin"

    if images_bin.exists():
        with open(images_bin, "rb") as f:
            stats["n_images"] = struct.unpack("<Q", f.read(8))[0]
    if points_bin.exists():
        with open(points_bin, "rb") as f:
            stats["n_points"] = struct.unpack("<Q", f.read(8))[0]
    if cameras_bin.exists():
        with open(cameras_bin, "rb") as f:
            stats["n_cameras"] = struct.unpack("<Q", f.read(8))[0]
    return stats


# ──────────────────── Main Experiment ────────────────────

def run_improved_sfm(scene_name: str, scene_dir: Path, output_dir: Path):
    """Run improved SfM pipeline."""
    import pycolmap
    from hloc import extract_features, match_features, pairs_from_exhaustive, pairs_from_retrieval, reconstruction

    images_dir = scene_dir / "images"
    output_dir.mkdir(parents=True, exist_ok=True)
    hloc_dir = output_dir / "hloc"
    hloc_dir.mkdir(exist_ok=True)

    image_list = sorted([
        p.name for p in images_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ])
    print(f"\n{'='*60}")
    print(f"Scene: {scene_name} ({len(image_list)} images)")
    print(f"Output: {output_dir}")
    print(f"{'='*60}")

    # ── 1. Feature extraction: superpoint_max (4096 kp, 1600px) ──
    print("\n[1] Feature extraction: superpoint_max (4096 kp, resize_max=1600)")
    feature_conf = extract_features.confs["superpoint_max"]
    t0 = time.time()
    feature_path = extract_features.main(feature_conf, images_dir, hloc_dir)
    t_feat = time.time() - t0
    print(f"  Done in {t_feat:.1f}s → {feature_path}")

    # ── 2. Pair generation: sequential + retrieval ──
    print("\n[2] Pair generation: sequential(overlap=10) + retrieval(NetVLAD top-15)")

    # Sequential pairs
    seq_pairs = hloc_dir / "pairs-sequential.txt"
    generate_sequential_pairs(image_list, seq_pairs, overlap=10)

    # Retrieval pairs (NetVLAD)
    print("  Extracting NetVLAD global descriptors...")
    retrieval_conf = extract_features.confs["netvlad"]
    t0 = time.time()
    retrieval_path = extract_features.main(retrieval_conf, images_dir, hloc_dir)
    t_ret = time.time() - t0
    print(f"  NetVLAD done in {t_ret:.1f}s")

    ret_pairs = hloc_dir / "pairs-retrieval.txt"
    pairs_from_retrieval.main(retrieval_path, ret_pairs, num_matched=15)
    print(f"  Retrieval pairs generated")

    # Merge
    merged_pairs = hloc_dir / "pairs-merged.txt"
    merge_pair_files([seq_pairs, ret_pairs], merged_pairs)

    # Also generate exhaustive pairs for comparison of pair count
    exh_pairs = hloc_dir / "pairs-exhaustive.txt"
    pairs_from_exhaustive.main(exh_pairs, image_list=image_list)
    with open(exh_pairs) as f:
        n_exh = sum(1 for _ in f)
    with open(merged_pairs) as f:
        n_merged = sum(1 for _ in f)
    print(f"  Exhaustive would be: {n_exh} pairs")
    print(f"  Sequential+Retrieval: {n_merged} pairs ({n_merged/n_exh*100:.1f}% of exhaustive)")

    # ── 3. Matching ──
    print("\n[3] Feature matching (LightGlue)")
    matcher_conf = match_features.confs["superpoint+lightglue"]
    t0 = time.time()
    match_path = match_features.main(
        matcher_conf, merged_pairs,
        features=feature_path,
        matches=hloc_dir / "matches-splg.h5",
    )
    t_match = time.time() - t0
    print(f"  Done in {t_match:.1f}s → {match_path}")

    # ── 4. Reconstruction ──
    print("\n[4] COLMAP reconstruction")
    sfm_dir = output_dir / "sparse" / "sfm_temp"
    sfm_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    model = reconstruction.main(
        sfm_dir=sfm_dir,
        image_dir=images_dir,
        pairs=merged_pairs,
        features=feature_path,
        matches=match_path,
        camera_mode=pycolmap.CameraMode.SINGLE,
    )
    t_recon = time.time() - t0
    print(f"  Done in {t_recon:.1f}s")

    # reconstruction.main returns best Reconstruction object
    # Write it directly to target using the pycolmap API
    target = output_dir / "sparse" / "0"
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)
    model.write(str(target))
    # Clean up temp
    if sfm_dir.exists():
        shutil.rmtree(sfm_dir)

    stats_before = read_sfm_stats(target)
    print(f"\n  SfM result (before undistort): {stats_before}")

    # ── 5. Undistort ──
    print("\n[5] Image undistortion (pycolmap)")
    undist_dir = output_dir / "undistorted"
    if undist_dir.exists():
        shutil.rmtree(undist_dir)
    t0 = time.time()
    pycolmap.undistort_images(
        output_path=str(undist_dir),
        input_path=str(target),
        image_path=str(images_dir),
    )
    t_undist = time.time() - t0
    print(f"  Done in {t_undist:.1f}s")

    # Debug: show undistort output structure
    print(f"  Undistort output contents:")
    for p in sorted(undist_dir.rglob("*")):
        if p.is_file():
            sz = p.stat().st_size / 1024
            print(f"    {p.relative_to(undist_dir)} ({sz:.0f} KB)")

    # Replace images and sparse model with undistorted versions
    undist_images = undist_dir / "images"
    # sparse model might be at undistorted/sparse/ directly (files) or in a subdir
    undist_sparse_dir = undist_dir / "sparse"

    if undist_images.exists():
        n_undist = len(list(undist_images.iterdir()))
        print(f"  Undistorted images: {n_undist}")

        # Symlink original images for reference, replace with undistorted
        final_images = output_dir / "images"
        final_images.mkdir(parents=True, exist_ok=True)
        shutil.copytree(undist_images, final_images, dirs_exist_ok=True)

    # Replace sparse model with undistorted version
    if undist_sparse_dir.exists():
        # Check if model files are directly in sparse/ or in sparse/0/
        if (undist_sparse_dir / "cameras.bin").exists():
            # Files directly in sparse/
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(undist_sparse_dir, target)
            print(f"  Replaced sparse model with undistorted (direct)")
        else:
            # Look for subdirectory
            for sub in sorted(undist_sparse_dir.iterdir()):
                if sub.is_dir() and (sub / "cameras.bin").exists():
                    if target.exists():
                        shutil.rmtree(target)
                    shutil.copytree(sub, target)
                    print(f"  Replaced sparse model with undistorted ({sub.name})")
                    break

    stats_after = read_sfm_stats(target)

    # ── Summary ──
    print(f"\n{'='*60}")
    print(f"RESULTS: {scene_name}")
    print(f"{'='*60}")
    print(f"  Feature extraction: {t_feat:.1f}s")
    print(f"  NetVLAD:            {t_ret:.1f}s")
    print(f"  Matching:           {t_match:.1f}s")
    print(f"  Reconstruction:     {t_recon:.1f}s")
    print(f"  Undistortion:       {t_undist:.1f}s")
    print(f"  Total:              {t_feat + t_ret + t_match + t_recon + t_undist:.1f}s")
    print(f"")
    print(f"  Registered images:  {stats_after.get('n_images', '?')}")
    print(f"  3D points:          {stats_after.get('n_points', '?')}")
    print(f"  Camera model:       PINHOLE (undistorted)")
    print(f"{'='*60}\n")

    return stats_after


def main():
    print("=" * 60)
    print("SfM Improvement Experiment")
    print("  1. SuperPoint max_keypoints=4096, resize_max=1600")
    print("  2. Sequential(overlap=10) + Retrieval(NetVLAD top-15) pairs")
    print("  3. Proper image undistortion via pycolmap")
    print("=" * 60)

    # Baseline stats
    print("\n=== BASELINE STATS ===")
    for name, scene_dir in SCENES.items():
        baseline = scene_dir / "sparse" / "0"
        if baseline.exists():
            stats = read_sfm_stats(baseline)
            total = len(list((scene_dir / "images").iterdir()))
            print(f"  {name}: {stats.get('n_images', '?')}/{total} images, "
                  f"{stats.get('n_points', '?')} points")

    # Run improved SfM
    results = {}
    for name, scene_dir in SCENES.items():
        out_dir = OUTPUT_BASE / name
        results[name] = run_improved_sfm(name, scene_dir, out_dir)

    # Final comparison
    print("\n" + "=" * 60)
    print("COMPARISON: Baseline vs Improved")
    print("=" * 60)
    print(f"{'Scene':<8} {'Metric':<16} {'Baseline':<12} {'Improved':<12} {'Change':<10}")
    print("-" * 58)
    for name, scene_dir in SCENES.items():
        baseline = read_sfm_stats(scene_dir / "sparse" / "0")
        improved = results[name]
        for metric in ["n_images", "n_points"]:
            b = baseline.get(metric, 0)
            i = improved.get(metric, 0)
            change = f"+{i-b}" if i >= b else f"{i-b}"
            pct = f"({(i-b)/b*100:+.1f}%)" if b > 0 else ""
            print(f"{name:<8} {metric:<16} {b:<12} {i:<12} {change} {pct}")


if __name__ == "__main__":
    main()
