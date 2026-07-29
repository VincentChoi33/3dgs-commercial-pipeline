"""Step 3: Train 3DGS via gaussian-splatting-lightning."""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
from pathlib import Path

import yaml

from pipeline.metadata import ensure_metadata_dir, read_optional_json, write_json_atomic

log = logging.getLogger("pipeline.train")

_REPO_ROOT = Path(__file__).resolve().parent.parent
_REPO_PROFILE_DIR = _REPO_ROOT / "configs" / "lightning" / "quality_first"
_REPO_PROFILES = {
    "quality": {
        "config_name": "quality_first/quality.yaml",
        "source": _REPO_PROFILE_DIR / "quality.yaml",
    }
}
_STANDARD_METRIC_KEYS = ("psnr", "ssim", "lpips")


def find_framework(cfg_path: str | None = None) -> Path:
    """Locate gaussian-splatting-lightning installation."""
    candidates = []
    if cfg_path:
        candidates.append(Path(cfg_path))

    env_path = os.environ.get("GS_LIGHTNING_PATH")
    if env_path:
        candidates.append(Path(env_path))

    candidates += [
        Path.home() / "gaussian-splatting-lightning",
        Path("/opt/gaussian-splatting-lightning"),
        Path("/opt/gs-lightning"),
    ]

    for c in candidates:
        if (c / "main.py").exists():
            return c

    raise FileNotFoundError(
        "gaussian-splatting-lightning not found. "
        "Set train.framework_path in config or GS_LIGHTNING_PATH env var."
    )


def _copy_repo_profile(framework: Path, profile: str) -> Path:
    profile_info = _REPO_PROFILES[profile]
    target = framework / "configs" / profile_info["config_name"]
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(profile_info["source"], target)
    return target


def _write_phase_override_config(base_config: Path, training_dir: Path, phase_overrides: dict) -> Path:
    patched_config = training_dir / "patched_train_config.yaml"
    config_data = read_optional_json(base_config)
    if config_data is None:
        with open(base_config, encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}
        config_data = loaded if isinstance(loaded, dict) else {}

    config_data["pipeline_overrides"] = phase_overrides
    with open(patched_config, "w", encoding="utf-8") as handle:
        yaml.safe_dump(config_data, handle, sort_keys=False)
    return patched_config


def _resolve_training_config(framework: Path, cfg: dict, training_dir: Path) -> tuple[Path, dict]:
    profile = cfg.get("profile", "default")
    phase_overrides = cfg.get("phase_overrides") or {}
    config_name = cfg.get("config", "gsplat_v1.yaml")
    uses_repo_profile = profile in _REPO_PROFILES
    synced_profile_path = None

    if uses_repo_profile:
        config_path = _copy_repo_profile(framework, profile)
        config_name = _REPO_PROFILES[profile]["config_name"]
        synced_profile_path = config_path
    else:
        config_path = framework / "configs" / config_name

    if phase_overrides:
        config_path = _write_phase_override_config(config_path, training_dir, phase_overrides)

    command_choices = {
        "profile": profile,
        "uses_repo_profile": uses_repo_profile,
        "has_phase_overrides": bool(phase_overrides),
    }

    return config_path, {
        "profile": profile,
        "config_name": config_name,
        "phase_overrides": phase_overrides,
        "synced_profile_path": str(synced_profile_path) if synced_profile_path else None,
        "command_choices": command_choices,
    }


def _extract_training_metrics(training_dir: Path) -> dict:
    metrics = {
        "psnr": None,
        "ssim": None,
        "lpips": None,
        "best_step": None,
        "source_artifact": None,
    }

    for candidate in (
        training_dir / "eval_metrics.json",
        training_dir / "training_metrics.json",
        training_dir / "metrics.json",
        training_dir / "results.json",
    ):
        payload = read_optional_json(candidate)
        if not isinstance(payload, dict):
            continue

        metrics["source_artifact"] = str(candidate)
        for key in _STANDARD_METRIC_KEYS + ("best_step",):
            if key in payload:
                metrics[key] = payload[key]

        for key, value in payload.items():
            if key not in metrics and isinstance(value, (str, int, float, bool)):
                metrics[key] = value
        break

    return metrics


def run_train(scene_dir: Path, cfg: dict, name: str = "scene"):
    """Run 3DGS training."""
    framework = find_framework(cfg.get("framework_path"))
    max_steps = cfg.get("max_steps", 30000)
    lambda_dssim = cfg.get("lambda_dssim", 0.3)
    gpu = cfg.get("gpu", 0)
    save_iters = cfg.get("save_iterations", [7000, 30000])

    training_dir = scene_dir / "training"
    training_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir = ensure_metadata_dir(scene_dir)

    config_file, training_selection = _resolve_training_config(framework, cfg, training_dir)

    cmd = [
        "python", str(framework / "main.py"), "fit",
        "--config", str(config_file),
        "--data.path", str(scene_dir),
        "--model.metric.init_args.lambda_dssim", str(lambda_dssim),
        "--max_steps", str(max_steps),
        "--output", str(training_dir),
        "-n", name,
        "--trainer.devices", "1",
    ]

    for it in save_iters:
        cmd += [f"--save_iterations+={it}"]

    profile_metadata = {
        "profile": training_selection["profile"],
        "config_name": training_selection["config_name"],
        "framework_path": str(framework),
        "framework_config_path": str(config_file),
        "synced_profile_path": training_selection["synced_profile_path"],
        "max_steps": max_steps,
        "save_iterations": list(save_iters),
        "phase_overrides": training_selection["phase_overrides"],
        "command_choices": training_selection["command_choices"],
        "command": cmd,
        "lambda_dssim": lambda_dssim,
        "gpu": gpu,
        "scene_name": name,
    }
    write_json_atomic(metadata_dir / "training_profile.json", profile_metadata)

    log.info(f"Training: {name}, {max_steps} steps, dssim={lambda_dssim}, GPU={gpu}")
    log.info(f"  Framework: {framework}")
    log.info(f"  Command: CUDA_VISIBLE_DEVICES={gpu} {' '.join(cmd)}")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)

    result = subprocess.run(cmd, env=env, cwd=str(framework))
    if result.returncode != 0:
        raise RuntimeError(f"Training failed with exit code {result.returncode}")

    write_json_atomic(metadata_dir / "training_metrics.json", _extract_training_metrics(training_dir))

    log.info(f"Training complete → {training_dir}")
