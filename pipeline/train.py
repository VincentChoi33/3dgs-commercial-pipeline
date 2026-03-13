"""Step 3: Train 3DGS via gaussian-splatting-lightning."""
import logging
import os
import subprocess
from pathlib import Path

log = logging.getLogger("pipeline.train")


def find_framework(cfg_path: str | None = None) -> Path:
    """Locate gaussian-splatting-lightning installation."""
    candidates = []
    if cfg_path:
        candidates.append(Path(cfg_path))

    # Check environment variable
    env_path = os.environ.get("GS_LIGHTNING_PATH")
    if env_path:
        candidates.append(Path(env_path))

    # Check common locations
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


def run_train(scene_dir: Path, cfg: dict, name: str = "scene"):
    """Run 3DGS training."""
    framework = find_framework(cfg.get("framework_path"))
    config_name = cfg.get("config", "gsplat_v1.yaml")
    config_file = framework / "configs" / config_name
    max_steps = cfg.get("max_steps", 30000)
    lambda_dssim = cfg.get("lambda_dssim", 0.3)
    gpu = cfg.get("gpu", 0)
    save_iters = cfg.get("save_iterations", [7000, 30000])

    training_dir = scene_dir / "training"
    training_dir.mkdir(parents=True, exist_ok=True)

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

    log.info(f"Training: {name}, {max_steps} steps, dssim={lambda_dssim}, GPU={gpu}")
    log.info(f"  Framework: {framework}")
    log.info(f"  Command: CUDA_VISIBLE_DEVICES={gpu} {' '.join(cmd)}")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)

    result = subprocess.run(cmd, env=env, cwd=str(framework))
    if result.returncode != 0:
        raise RuntimeError(f"Training failed with exit code {result.returncode}")

    log.info(f"Training complete → {training_dir}")
