"""
Compress 3DGS PLY: original → prune + SH0 + f16 (+ optional downsample).

Usage:
    python compress.py input.ply output.ply
    python compress.py input.ply output.ply --downsample 0.5
    python compress.py input.ply output.ply --downsample 0.5 --sh-degree 1
"""
import argparse
import numpy as np
from plyfile import PlyData, PlyElement


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def load_ply(path):
    ply = PlyData.read(path)
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


def save_ply(path, xyz, f_dc, f_rest, opacity, scales, rots):
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

    PlyData([PlyElement.describe(arr, "vertex")]).write(path)


def prune(xyz, f_dc, f_rest, opacity, scales, rots, threshold=0.005):
    mask = sigmoid(opacity) > threshold
    return xyz[mask], f_dc[mask], f_rest[mask], opacity[mask], scales[mask], rots[mask]


def reduce_sh(f_rest, current_degree, target_degree):
    if target_degree >= current_degree or current_degree == 0:
        return f_rest, target_degree

    n_coeffs_per_channel = (target_degree + 1) ** 2 - 1
    n_rest_new = n_coeffs_per_channel * 3

    if n_rest_new == 0:
        return np.zeros((f_rest.shape[0], 0), dtype=np.float32), 0

    current_coeffs = (current_degree + 1) ** 2 - 1
    R = f_rest[:, :current_coeffs][:, :n_coeffs_per_channel]
    G = f_rest[:, current_coeffs:2 * current_coeffs][:, :n_coeffs_per_channel]
    B = f_rest[:, 2 * current_coeffs:3 * current_coeffs][:, :n_coeffs_per_channel]

    return np.concatenate([R, G, B], axis=1), target_degree


def quantize_f16(xyz, f_dc, f_rest, opacity, scales, rots):
    return (
        xyz.astype(np.float16).astype(np.float32),
        f_dc.astype(np.float16).astype(np.float32),
        f_rest.astype(np.float16).astype(np.float32) if f_rest.shape[1] > 0 else f_rest,
        opacity.astype(np.float16).astype(np.float32),
        scales.astype(np.float16).astype(np.float32),
        rots.astype(np.float16).astype(np.float32),
    )


def downsample(xyz, f_dc, f_rest, opacity, scales, rots, ratio):
    opacity_act = sigmoid(opacity)
    volume = np.exp(scales).prod(axis=1)
    importance = opacity_act * volume

    k = int(xyz.shape[0] * ratio)
    idx = np.argsort(-importance)[:k]

    return xyz[idx], f_dc[idx], f_rest[idx], opacity[idx], scales[idx], rots[idx]


def main():
    parser = argparse.ArgumentParser(description="Compress 3DGS PLY")
    parser.add_argument("input", help="Input PLY path")
    parser.add_argument("output", help="Output PLY path")
    parser.add_argument("--sh-degree", type=int, default=0, help="Target SH degree (default: 0)")
    parser.add_argument("--downsample", type=float, default=None, help="Keep ratio (e.g. 0.5 = keep 50%%)")
    parser.add_argument("--no-prune", action="store_true", help="Skip opacity pruning")
    parser.add_argument("--no-f16", action="store_true", help="Skip float16 quantization")
    parser.add_argument("--prune-threshold", type=float, default=0.005, help="Opacity prune threshold")
    args = parser.parse_args()

    import os
    input_size = os.path.getsize(args.input)

    # Load
    print(f"Loading {args.input} ({input_size / 1e6:.1f} MB)")
    xyz, f_dc, f_rest, opacity, scales, rots, sh_degree = load_ply(args.input)
    N_original = xyz.shape[0]
    print(f"  {N_original:,} Gaussians, SH degree {sh_degree}")

    # Prune
    if not args.no_prune:
        xyz, f_dc, f_rest, opacity, scales, rots = prune(
            xyz, f_dc, f_rest, opacity, scales, rots, args.prune_threshold
        )
        N_pruned = xyz.shape[0]
        print(f"  Pruned: {N_original:,} → {N_pruned:,} ({N_original - N_pruned:,} removed)")

    # SH reduction
    if args.sh_degree < sh_degree:
        f_rest, sh_degree = reduce_sh(f_rest, sh_degree, args.sh_degree)
        print(f"  SH degree → {sh_degree} (f_rest: {f_rest.shape[1]} coeffs)")

    # Downsample
    if args.downsample is not None:
        N_before = xyz.shape[0]
        xyz, f_dc, f_rest, opacity, scales, rots = downsample(
            xyz, f_dc, f_rest, opacity, scales, rots, args.downsample
        )
        print(f"  Downsampled: {N_before:,} → {xyz.shape[0]:,} ({args.downsample:.0%})")

    # Float16
    if not args.no_f16:
        xyz, f_dc, f_rest, opacity, scales, rots = quantize_f16(
            xyz, f_dc, f_rest, opacity, scales, rots
        )
        print(f"  Float16 quantized")

    # Save
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    save_ply(args.output, xyz, f_dc, f_rest, opacity, scales, rots)
    output_size = os.path.getsize(args.output)

    print(f"\nResult: {args.output}")
    print(f"  {xyz.shape[0]:,} Gaussians")
    print(f"  {input_size / 1e6:.1f} MB → {output_size / 1e6:.1f} MB ({output_size / input_size * 100:.1f}%)")


if __name__ == "__main__":
    main()
