"""Step 5: Compress Gaussian PLY — prune, SH reduction, f16, downsample."""
import logging
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement

from pipeline.metadata import ensure_metadata_dir, write_json_atomic

log = logging.getLogger("pipeline.compress")


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def load_ply(path: Path):
    ply = PlyData.read(str(path))
    v = ply.elements[0]
    N = v.count

    xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)
    f_dc = np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], axis=1).astype(np.float32)

    rest_names = sorted([p.name for p in v.properties if p.name.startswith("f_rest_")])
    n_rest = len(rest_names)
    if n_rest > 0:
        f_rest = np.stack([v[f"f_rest_{i}"] for i in range(n_rest)], axis=1).astype(np.float32)
    else:
        f_rest = np.zeros((N, 0), dtype=np.float32)

    opacity = np.array(v["opacity"], dtype=np.float32)
    scales = np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], axis=1).astype(np.float32)
    rots = np.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], axis=1).astype(np.float32)

    sh_degree = int(((n_rest // 3) + 1) ** 0.5) - 1 if n_rest > 0 else 0
    return xyz, f_dc, f_rest, opacity, scales, rots, sh_degree


def save_ply(path: Path, xyz, f_dc, f_rest, opacity, scales, rots):
    N = xyz.shape[0]
    n_rest = f_rest.shape[1] if f_rest.ndim == 2 else 0

    dtype = [("x", "f4"), ("y", "f4"), ("z", "f4")]
    dtype += [("f_dc_0", "f4"), ("f_dc_1", "f4"), ("f_dc_2", "f4")]
    for i in range(n_rest):
        dtype.append((f"f_rest_{i}", "f4"))
    dtype.append(("opacity", "f4"))
    dtype += [("scale_0", "f4"), ("scale_1", "f4"), ("scale_2", "f4")]
    dtype += [("rot_0", "f4"), ("rot_1", "f4"), ("rot_2", "f4"), ("rot_3", "f4")]

    arr = np.empty(N, dtype=dtype)
    arr["x"], arr["y"], arr["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    arr["f_dc_0"], arr["f_dc_1"], arr["f_dc_2"] = f_dc[:, 0], f_dc[:, 1], f_dc[:, 2]
    for i in range(n_rest):
        arr[f"f_rest_{i}"] = f_rest[:, i]
    arr["opacity"] = opacity
    arr["scale_0"], arr["scale_1"], arr["scale_2"] = scales[:, 0], scales[:, 1], scales[:, 2]
    arr["rot_0"], arr["rot_1"], arr["rot_2"], arr["rot_3"] = rots[:, 0], rots[:, 1], rots[:, 2], rots[:, 3]

    path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(arr, "vertex")]).write(str(path))


def prune(xyz, f_dc, f_rest, opacity, scales, rots, threshold=0.005):
    mask = sigmoid(opacity) > threshold
    return xyz[mask], f_dc[mask], f_rest[mask], opacity[mask], scales[mask], rots[mask]


def reduce_sh(f_rest, current_degree, target_degree):
    if target_degree >= current_degree or current_degree == 0:
        return f_rest, target_degree
    n_coeffs = (target_degree + 1) ** 2 - 1
    if n_coeffs == 0:
        return np.zeros((f_rest.shape[0], 0), dtype=np.float32), 0
    cur_coeffs = (current_degree + 1) ** 2 - 1
    R = f_rest[:, :cur_coeffs][:, :n_coeffs]
    G = f_rest[:, cur_coeffs:2 * cur_coeffs][:, :n_coeffs]
    B = f_rest[:, 2 * cur_coeffs:3 * cur_coeffs][:, :n_coeffs]
    return np.concatenate([R, G, B], axis=1), target_degree


def quantize_f16(xyz, f_dc, f_rest, opacity, scales, rots):
    to16 = lambda x: x.astype(np.float16).astype(np.float32)
    return (
        to16(xyz), to16(f_dc),
        to16(f_rest) if f_rest.shape[1] > 0 else f_rest,
        to16(opacity), to16(scales), to16(rots),
    )


def downsample(xyz, f_dc, f_rest, opacity, scales, rots, ratio):
    importance = sigmoid(opacity) * np.exp(scales).prod(axis=1)
    k = int(xyz.shape[0] * ratio)
    idx = np.argsort(-importance)[:k]
    return xyz[idx], f_dc[idx], f_rest[idx], opacity[idx], scales[idx], rots[idx]


def compress_ply(input_path: Path, output_path: Path, sh_degree=0, float16=True,
                 prune_threshold=0.005, downsample_ratio=None):
    xyz, f_dc, f_rest, opacity, scales, rots, cur_sh = load_ply(input_path)
    n_orig = xyz.shape[0]

    xyz, f_dc, f_rest, opacity, scales, rots = prune(
        xyz, f_dc, f_rest, opacity, scales, rots, prune_threshold
    )
    log.info(f"  Pruned: {n_orig:,} → {xyz.shape[0]:,}")

    if sh_degree < cur_sh:
        f_rest, cur_sh = reduce_sh(f_rest, cur_sh, sh_degree)
        log.info(f"  SH degree → {cur_sh}")

    if downsample_ratio is not None:
        n_before = xyz.shape[0]
        xyz, f_dc, f_rest, opacity, scales, rots = downsample(
            xyz, f_dc, f_rest, opacity, scales, rots, downsample_ratio
        )
        log.info(f"  Downsampled: {n_before:,} → {xyz.shape[0]:,}")

    if float16:
        xyz, f_dc, f_rest, opacity, scales, rots = quantize_f16(
            xyz, f_dc, f_rest, opacity, scales, rots
        )

    save_ply(output_path, xyz, f_dc, f_rest, opacity, scales, rots)
    output_size_bytes = output_path.stat().st_size
    size_mb = output_size_bytes / 1024 / 1024
    log.info(f"  → {output_path} ({xyz.shape[0]:,} GS, {size_mb:.1f} MB)")
    return {
        "path": str(output_path),
        "gaussian_count": int(xyz.shape[0]),
        "output_size_bytes": int(output_size_bytes),
    }


def run_compress(input_ply: Path, output_dir: Path, cfg: dict):
    sh = cfg.get("sh_degree", 0)
    f16 = cfg.get("float16", True)
    thresh = cfg.get("prune_threshold", 0.005)
    ds = cfg.get("downsample", 0.5)
    metadata_dir = ensure_metadata_dir(output_dir)

    log.info(f"Compressing {input_ply}")

    output_files = [
        compress_ply(
            input_ply,
            output_dir / "compressed.ply",
            sh_degree=sh,
            float16=f16,
            prune_threshold=thresh,
        )
    ]

    if ds is not None:
        pct = int(ds * 100)
        output_files.append(
            compress_ply(
                input_ply,
                output_dir / f"compressed_ds{pct}.ply",
                sh_degree=sh,
                float16=f16,
                prune_threshold=thresh,
                downsample_ratio=ds,
            )
        )

    compression_report = {
        "input_path": str(input_ply),
        "compression_settings": {
            "sh_degree": sh,
            "float16": f16,
            "prune_threshold": thresh,
            "downsample": ds,
        },
        "output_files": output_files,
    }
    write_json_atomic(metadata_dir / "compression_report.json", compression_report)
