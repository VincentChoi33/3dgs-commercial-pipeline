#!/usr/bin/env python3
"""SfM experiment with FULL 4K resolution (resize_max=3840) on 6027 only."""
from __future__ import annotations

import copy
import shutil
import struct
import sys
import time
from pathlib import Path

sys.path.insert(0, "/data/choihy/GSplat/hloc")

SCENE_DIR = Path("/data/choihy/GSplat/runs/2026-03-12/video_improved/office_4k_6027")
OUTPUT_DIR = Path("/data/choihy/GSplat/runs/2026-03-13/sfm_fullres/6027")


def read_sfm_stats(sparse_dir):
    stats = {}
    for name, key in [("images.bin", "n_images"), ("points3D.bin", "n_points"), ("cameras.bin", "n_cameras")]:
        p = sparse_dir / name
        if p.exists():
            with open(p, "rb") as f:
                stats[key] = struct.unpack("<Q", f.read(8))[0]
    return stats


def generate_sequential_pairs(image_list, output, overlap=10):
    pairs = set()
    n = len(image_list)
    for i in range(n):
        for j in range(i + 1, min(i + 1 + overlap, n)):
            pairs.add((image_list[i], image_list[j]))
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


def merge_pair_files(files, output):
    all_pairs = set()
    for p in files:
        with open(p) as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split()
                    if len(parts) == 2:
                        a, b = sorted(parts)
                        all_pairs.add((a, b))
    with open(output, "w") as f:
        for a, b in sorted(all_pairs):
            f.write(f"{a} {b}\n")
    print(f"  Merged pairs: {len(all_pairs)}")
    return output


def main():
    import pycolmap
    from hloc import extract_features, match_features, pairs_from_retrieval, reconstruction

    images_dir = SCENE_DIR / "images"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    hloc_dir = OUTPUT_DIR / "hloc"
    hloc_dir.mkdir(exist_ok=True)

    image_list = sorted([
        p.name for p in images_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ])
    print(f"Scene: 6027 ({len(image_list)} images)")
    print(f"Resolution: FULL 4K (resize_max=3840)")

    # 1. Feature extraction at full resolution
    print("\n[1] Feature extraction: superpoint FULL RES (4096 kp, resize_max=3840)")
    feature_conf = copy.deepcopy(extract_features.confs["superpoint_max"])
    feature_conf["preprocessing"]["resize_max"] = 3840
    feature_conf["output"] = "feats-superpoint-n4096-r3840"
    t0 = time.time()
    feature_path = extract_features.main(feature_conf, images_dir, hloc_dir)
    t_feat = time.time() - t0
    print(f"  Done in {t_feat:.1f}s")

    # 2. Pair generation (same as improved: sequential + retrieval)
    print("\n[2] Pair generation: sequential + retrieval")
    seq_pairs = hloc_dir / "pairs-sequential.txt"
    generate_sequential_pairs(image_list, seq_pairs, overlap=10)

    retrieval_conf = extract_features.confs["netvlad"]
    t0 = time.time()
    retrieval_path = extract_features.main(retrieval_conf, images_dir, hloc_dir)
    t_ret = time.time() - t0
    print(f"  NetVLAD done in {t_ret:.1f}s")

    ret_pairs = hloc_dir / "pairs-retrieval.txt"
    pairs_from_retrieval.main(retrieval_path, ret_pairs, num_matched=15)
    merged_pairs = hloc_dir / "pairs-merged.txt"
    merge_pair_files([seq_pairs, ret_pairs], merged_pairs)

    # 3. Matching
    print("\n[3] Feature matching (LightGlue)")
    matcher_conf = match_features.confs["superpoint+lightglue"]
    t0 = time.time()
    match_path = match_features.main(
        matcher_conf, merged_pairs,
        features=feature_path,
        matches=hloc_dir / "matches-splg.h5",
    )
    t_match = time.time() - t0
    print(f"  Done in {t_match:.1f}s")

    # 4. Reconstruction
    print("\n[4] COLMAP reconstruction")
    sfm_dir = OUTPUT_DIR / "sparse" / "sfm_temp"
    sfm_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    model = reconstruction.main(
        sfm_dir=sfm_dir, image_dir=images_dir, pairs=merged_pairs,
        features=feature_path, matches=match_path,
        camera_mode=pycolmap.CameraMode.SINGLE,
    )
    t_recon = time.time() - t0
    print(f"  Done in {t_recon:.1f}s")

    target = OUTPUT_DIR / "sparse" / "0"
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)
    model.write(str(target))
    if sfm_dir.exists():
        shutil.rmtree(sfm_dir)

    stats_before = read_sfm_stats(target)
    print(f"  SfM result: {stats_before}")

    # 5. Undistort
    print("\n[5] Image undistortion")
    undist_dir = OUTPUT_DIR / "undistorted"
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

    undist_images = undist_dir / "images"
    undist_sparse = undist_dir / "sparse"
    if undist_images.exists():
        n = len(list(undist_images.iterdir()))
        print(f"  Undistorted images: {n}")
        final_images = OUTPUT_DIR / "images"
        final_images.mkdir(parents=True, exist_ok=True)
        shutil.copytree(undist_images, final_images, dirs_exist_ok=True)
    if undist_sparse.exists():
        if (undist_sparse / "cameras.bin").exists():
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(undist_sparse, target)
        else:
            for sub in sorted(undist_sparse.iterdir()):
                if sub.is_dir() and (sub / "cameras.bin").exists():
                    if target.exists():
                        shutil.rmtree(target)
                    shutil.copytree(sub, target)
                    break

    stats_after = read_sfm_stats(target)
    total_time = t_feat + t_ret + t_match + t_recon + t_undist

    print(f"\n{'=' * 60}")
    print(f"RESULTS: 6027 FULL RESOLUTION (resize_max=3840)")
    print(f"{'=' * 60}")
    print(f"  Feature extraction: {t_feat:.1f}s")
    print(f"  NetVLAD:            {t_ret:.1f}s")
    print(f"  Matching:           {t_match:.1f}s")
    print(f"  Reconstruction:     {t_recon:.1f}s")
    print(f"  Undistortion:       {t_undist:.1f}s")
    print(f"  Total:              {total_time:.1f}s")
    n_imgs = stats_after.get("n_images", "?")
    n_pts = stats_after.get("n_points", "?")
    print(f"  Registered images:  {n_imgs}")
    print(f"  3D points:          {n_pts}")
    print(f"{'=' * 60}")
    print(f"\nCOMPARISON (6027):")
    print(f"  resize_max=1024 (baseline):  141 imgs, 23,388 pts")
    print(f"  resize_max=1600 (improved):  142 imgs, 36,346 pts")
    print(f"  resize_max=3840 (full res):  {n_imgs} imgs, {n_pts} pts")


if __name__ == "__main__":
    main()
