import pytest
import numpy as np
import torch
from pathlib import Path


def test_ckpt_to_ply(tmp_path):
    from pipeline.export import ckpt_to_ply

    N = 100
    state_dict = {
        "gaussian_model.gaussians.means": torch.randn(N, 3),
        "gaussian_model.gaussians.shs_dc": torch.randn(N, 1, 3),
        "gaussian_model.gaussians.shs_rest": torch.randn(N, 15, 3),
        "gaussian_model.gaussians.opacities": torch.randn(N, 1),
        "gaussian_model.gaussians.scales": torch.randn(N, 3),
        "gaussian_model.gaussians.rotations": torch.randn(N, 4),
    }
    ckpt_path = tmp_path / "test.ckpt"
    torch.save({"state_dict": state_dict}, ckpt_path)

    out_ply = tmp_path / "output.ply"
    ckpt_to_ply(ckpt_path, out_ply)

    assert out_ply.exists()
    assert out_ply.stat().st_size > 0

    from plyfile import PlyData
    ply = PlyData.read(str(out_ply))
    assert ply.elements[0].count == N


def test_find_best_checkpoint(tmp_path):
    from pipeline.export import find_best_checkpoint

    ckpts = tmp_path / "training" / "scene" / "checkpoints"
    ckpts.mkdir(parents=True)
    (ckpts / "epoch=49-step=6999.ckpt").touch()
    (ckpts / "epoch=212-step=29999.ckpt").touch()

    best = find_best_checkpoint(tmp_path / "training", target_step=None)
    assert "29999" in best.name

    best = find_best_checkpoint(tmp_path / "training", target_step=7000)
    assert "6999" in best.name
