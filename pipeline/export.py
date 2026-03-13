"""Step 4: Export full Gaussian PLY from Lightning checkpoint (self-contained)."""
import logging
import re
from pathlib import Path

import numpy as np
import torch
from plyfile import PlyData, PlyElement

log = logging.getLogger("pipeline.export")


def find_best_checkpoint(training_dir: Path, target_step: int | None = None) -> Path:
    """Find checkpoint closest to target step (or latest if None)."""
    ckpt_files = list(training_dir.rglob("*.ckpt"))
    if not ckpt_files:
        raise FileNotFoundError(f"No checkpoints in {training_dir}")

    def extract_step(p: Path) -> int:
        m = re.search(r"step=(\d+)", p.name)
        return int(m.group(1)) if m else 0

    if target_step is None:
        return max(ckpt_files, key=extract_step)

    return min(ckpt_files, key=lambda p: abs(extract_step(p) - target_step))


def ckpt_to_ply(ckpt_path: Path, output_path: Path):
    """Extract Gaussian PLY from Lightning checkpoint."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt["state_dict"]

    prefix = "gaussian_model.gaussians."
    means = state[f"{prefix}means"].numpy()
    shs_dc = state[f"{prefix}shs_dc"].numpy()
    shs_rest = state[f"{prefix}shs_rest"].numpy()
    opacities = state[f"{prefix}opacities"].numpy()
    scales = state[f"{prefix}scales"].numpy()
    rotations = state[f"{prefix}rotations"].numpy()

    N = means.shape[0]
    n_sh_rest = shs_rest.shape[1]
    sh_degree = int((n_sh_rest + 1) ** 0.5) - 1 if n_sh_rest > 0 else 0
    n_rest = n_sh_rest * 3

    shs_dc = shs_dc.reshape(N, 3)
    shs_rest = shs_rest.reshape(N, n_rest)
    opacities = opacities.reshape(N)

    dtype = [("x", "f4"), ("y", "f4"), ("z", "f4")]
    dtype += [("f_dc_0", "f4"), ("f_dc_1", "f4"), ("f_dc_2", "f4")]
    for i in range(n_rest):
        dtype.append((f"f_rest_{i}", "f4"))
    dtype.append(("opacity", "f4"))
    dtype += [("scale_0", "f4"), ("scale_1", "f4"), ("scale_2", "f4")]
    dtype += [("rot_0", "f4"), ("rot_1", "f4"), ("rot_2", "f4"), ("rot_3", "f4")]

    arr = np.zeros(N, dtype=dtype)
    arr["x"], arr["y"], arr["z"] = means[:, 0], means[:, 1], means[:, 2]
    arr["f_dc_0"], arr["f_dc_1"], arr["f_dc_2"] = shs_dc[:, 0], shs_dc[:, 1], shs_dc[:, 2]
    for i in range(n_rest):
        arr[f"f_rest_{i}"] = shs_rest[:, i]
    arr["opacity"] = opacities
    arr["scale_0"], arr["scale_1"], arr["scale_2"] = scales[:, 0], scales[:, 1], scales[:, 2]
    arr["rot_0"], arr["rot_1"], arr["rot_2"], arr["rot_3"] = rotations[:, 0], rotations[:, 1], rotations[:, 2], rotations[:, 3]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(arr, "vertex")]).write(str(output_path))

    size_mb = output_path.stat().st_size / 1024 / 1024
    log.info(f"Exported {N:,} Gaussians (SH degree {sh_degree}) → {output_path} ({size_mb:.1f} MB)")


def run_export(scene_dir: Path, cfg: dict) -> Path:
    """Find checkpoint and export PLY."""
    training_dir = scene_dir / "training"
    target_step = cfg.get("step")
    ckpt = find_best_checkpoint(training_dir, target_step)
    log.info(f"Using checkpoint: {ckpt.name}")

    output_path = scene_dir / "full.ply"
    ckpt_to_ply(ckpt, output_path)
    return output_path
