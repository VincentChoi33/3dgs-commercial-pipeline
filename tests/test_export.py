import torch


def _make_test_checkpoint(path, gaussian_count=100):
    state_dict = {
        "gaussian_model.gaussians.means": torch.randn(gaussian_count, 3),
        "gaussian_model.gaussians.shs_dc": torch.randn(gaussian_count, 1, 3),
        "gaussian_model.gaussians.shs_rest": torch.randn(gaussian_count, 15, 3),
        "gaussian_model.gaussians.opacities": torch.randn(gaussian_count, 1),
        "gaussian_model.gaussians.scales": torch.randn(gaussian_count, 3),
        "gaussian_model.gaussians.rotations": torch.randn(gaussian_count, 4),
    }
    torch.save({"state_dict": state_dict}, path)
    return path


def test_ckpt_to_ply(tmp_path):
    from pipeline.export import ckpt_to_ply

    ckpt_path = _make_test_checkpoint(tmp_path / "test.ckpt")
    out_ply = tmp_path / "output.ply"
    report = ckpt_to_ply(ckpt_path, out_ply)

    assert out_ply.exists()
    assert out_ply.stat().st_size > 0

    from plyfile import PlyData
    ply = PlyData.read(str(out_ply))
    assert ply.elements[0].count == 100
    assert report["gaussian_count"] == 100
    assert report["output_path"] == str(out_ply)
    assert report["output_size_bytes"] == out_ply.stat().st_size


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


def test_run_export_writes_export_report(tmp_path):
    from pipeline.export import run_export
    from pipeline.metadata import read_optional_json

    ckpt_dir = tmp_path / "training" / "scene" / "checkpoints"
    ckpt_dir.mkdir(parents=True)
    chosen_ckpt = _make_test_checkpoint(ckpt_dir / "epoch=49-step=6999.ckpt", gaussian_count=12)
    _make_test_checkpoint(ckpt_dir / "epoch=212-step=29999.ckpt", gaussian_count=20)

    output = run_export(tmp_path, {"step": 7000})
    report = read_optional_json(tmp_path / "metadata" / "export_report.json")

    assert output == tmp_path / "full.ply"
    assert report == {
        "chosen_checkpoint": str(chosen_ckpt),
        "target_step": 7000,
        "gaussian_count": 12,
        "output_path": str(output),
        "output_size_bytes": output.stat().st_size,
    }
