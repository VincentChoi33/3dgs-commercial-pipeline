import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from pipeline.train import run_train


REPO_ROOT = Path(__file__).resolve().parents[1]


def _make_framework(tmp_path: Path) -> Path:
    framework = tmp_path / "gaussian-splatting-lightning"
    (framework / "configs").mkdir(parents=True)
    (framework / "main.py").write_text("print('train')\n")
    (framework / "configs" / "gsplat_v1.yaml").write_text("trainer:\n  max_steps: 30000\n")
    return framework


def test_run_train_writes_training_profile_and_metrics_for_quality_profile(tmp_path, monkeypatch):
    scene_dir = tmp_path / "scene"
    framework = _make_framework(tmp_path)
    calls = []

    def fake_run(cmd, env, cwd):
        calls.append({"cmd": cmd, "env": env, "cwd": cwd})
        training_dir = scene_dir / "training"
        training_dir.mkdir(parents=True, exist_ok=True)
        (training_dir / "eval_metrics.json").write_text(
            json.dumps({"psnr": 31.5, "ssim": 0.91, "lpips": 0.08, "best_step": 12000})
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("pipeline.train.subprocess.run", fake_run)

    run_train(
        scene_dir,
        {
            "framework_path": str(framework),
            "profile": "quality",
            "max_steps": 12000,
            "lambda_dssim": 0.15,
            "save_iterations": [1000, 12000],
            "phase_overrides": {},
        },
        name="scene",
    )

    metadata_dir = scene_dir / "metadata"
    profile = json.loads((metadata_dir / "training_profile.json").read_text())
    metrics = json.loads((metadata_dir / "training_metrics.json").read_text())

    assert profile["profile"] == "quality"
    assert profile["config_name"] == "quality_first/quality.yaml"
    assert profile["max_steps"] == 12000
    assert profile["synced_profile_path"] == str(framework / "configs" / "quality_first" / "quality.yaml")
    assert profile["command"][0:3] == ["python", str(framework / "main.py"), "fit"]
    assert profile["command_choices"]["uses_repo_profile"] is True
    assert Path(profile["framework_path"]) == framework

    assert metrics["psnr"] == 31.5
    assert metrics["ssim"] == 0.91
    assert metrics["lpips"] == 0.08
    assert metrics["best_step"] == 12000
    assert metrics["source_artifact"] == str(scene_dir / "training" / "eval_metrics.json")

    synced_profile = framework / "configs" / "quality_first" / "quality.yaml"
    assert synced_profile.exists()
    synced_data = yaml.safe_load(synced_profile.read_text())
    assert synced_data["trainer"]["max_steps"] >= 1

    assert calls[0]["cwd"] == str(framework)
    assert calls[0]["env"]["CUDA_VISIBLE_DEVICES"] == "0"



def test_run_train_writes_null_metrics_when_backend_metrics_are_missing(tmp_path, monkeypatch):
    scene_dir = tmp_path / "scene"
    framework = _make_framework(tmp_path)

    monkeypatch.setattr(
        "pipeline.train.subprocess.run",
        lambda cmd, env, cwd: SimpleNamespace(returncode=0),
    )

    run_train(scene_dir, {"framework_path": str(framework), "phase_overrides": {}}, name="scene")

    metrics = json.loads((scene_dir / "metadata" / "training_metrics.json").read_text())

    assert metrics["psnr"] is None
    assert metrics["ssim"] is None
    assert metrics["lpips"] is None
    assert metrics["source_artifact"] is None



def test_run_train_ignores_normalized_metrics_artifact_without_backend_metrics(tmp_path, monkeypatch):
    scene_dir = tmp_path / "scene"
    framework = _make_framework(tmp_path)

    def fake_run(cmd, env, cwd):
        training_dir = scene_dir / "training"
        training_dir.mkdir(parents=True, exist_ok=True)
        (training_dir / "training_metrics.json").write_text(
            json.dumps({"psnr": 99.0, "ssim": 0.99, "lpips": 0.01, "best_step": 1})
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("pipeline.train.subprocess.run", fake_run)

    run_train(scene_dir, {"framework_path": str(framework), "phase_overrides": {}}, name="scene")

    metrics = json.loads((scene_dir / "metadata" / "training_metrics.json").read_text())

    assert metrics["psnr"] is None
    assert metrics["ssim"] is None
    assert metrics["lpips"] is None
    assert metrics["best_step"] is None
    assert metrics["source_artifact"] is None



def test_run_train_raises_clear_error_for_missing_config(tmp_path):
    framework = _make_framework(tmp_path)
    scene_dir = tmp_path / "scene"

    with pytest.raises(FileNotFoundError, match="Training config 'missing.yaml' for profile 'default' was not found"):
        run_train(
            scene_dir,
            {
                "framework_path": str(framework),
                "config": "missing.yaml",
                "phase_overrides": {},
            },
            name="scene",
        )



def test_run_train_command_changes_for_profile_and_phase_overrides(tmp_path, monkeypatch):
    scene_dir = tmp_path / "scene"
    framework = _make_framework(tmp_path)
    commands = []

    def fake_run(cmd, env, cwd):
        commands.append(cmd)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("pipeline.train.subprocess.run", fake_run)

    run_train(
        scene_dir / "default",
        {"framework_path": str(framework), "config": "gsplat_v1.yaml", "phase_overrides": {}},
        name="default-scene",
    )
    run_train(
        scene_dir / "quality",
        {"framework_path": str(framework), "profile": "quality", "phase_overrides": {}},
        name="quality-scene",
    )
    run_train(
        scene_dir / "patched",
        {
            "framework_path": str(framework),
            "config": "gsplat_v1.yaml",
            "phase_overrides": {"geometry": {"max_steps": 5000}},
        },
        name="patched-scene",
    )

    default_config = commands[0][commands[0].index("--config") + 1]
    quality_config = commands[1][commands[1].index("--config") + 1]
    patched_config = commands[2][commands[2].index("--config") + 1]

    assert default_config == str(framework / "configs" / "gsplat_v1.yaml")
    assert quality_config == str(framework / "configs" / "quality_first" / "quality.yaml")
    assert patched_config != default_config
    assert Path(patched_config).exists()

    patched_data = yaml.safe_load(Path(patched_config).read_text())
    assert patched_data["pipeline_overrides"]["geometry"]["max_steps"] == 5000
