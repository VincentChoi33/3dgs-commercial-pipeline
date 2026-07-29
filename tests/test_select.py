import json
from pathlib import Path

import cv2
import numpy as np


def _write_pattern(path: Path, seed: int, flat: bool = False) -> None:
    if flat:
        image = np.full((40, 40, 3), 127, dtype=np.uint8)
    else:
        rng = np.random.default_rng(seed)
        image = rng.integers(0, 255, (40, 40, 3), dtype=np.uint8)
    cv2.imwrite(str(path), image)


def _build_scene(tmp_path: Path, count: int = 6) -> Path:
    scene_dir = tmp_path / "scene"
    images_dir = scene_dir / "images"
    metadata_dir = scene_dir / "metadata"
    images_dir.mkdir(parents=True)
    metadata_dir.mkdir(parents=True)

    frames = []
    for index in range(count):
        image_name = f"frame_{index:04d}.jpg"
        _write_pattern(images_dir / image_name, seed=index)
        frames.append({"frame_index": index, "image_name": image_name})

    (metadata_dir / "frames.json").write_text(json.dumps(frames))
    return scene_dir


def test_run_select_writes_selection_artifacts(tmp_path):
    from pipeline.select import run_select

    scene_dir = _build_scene(tmp_path)
    selected_dir = run_select(scene_dir, {"keep_ratio": 0.5})

    assert selected_dir == scene_dir / "selected_images"
    assert selected_dir.exists()
    assert (scene_dir / "selected_frames.txt").exists()
    assert (scene_dir / "metadata" / "selection_metrics.json").exists()
    assert (scene_dir / "metadata" / "coverage_summary.json").exists()

    metrics = json.loads((scene_dir / "metadata" / "selection_metrics.json").read_text())
    assert metrics["selected_count"] >= 3
    assert metrics["fallback_used"] is False
    for frame in metrics["selected_frames"]:
        assert "overlap_novelty" in frame
        assert "baseline_diversity" in frame
        assert "coverage_signal" in frame
        assert frame["total_score"] >= (
            frame["blur_score"]
            + (frame["temporal_spacing"] * 25.0)
            + (frame["overlap_novelty"] * 20.0)
            + (frame["baseline_diversity"] * 20.0)
            + (frame["coverage_signal"] * 50.0)
            + (frame["pose_confidence"] * 10.0)
        )

    coverage = json.loads((scene_dir / "metadata" / "coverage_summary.json").read_text())
    assert "novelty_mean" in coverage
    assert "coverage_signal" in coverage


def test_run_select_uses_conservative_fallback_when_filtering_too_aggressive(tmp_path):
    from pipeline.select import run_select

    scene_dir = _build_scene(tmp_path, count=5)
    for image_path in (scene_dir / "images").glob("*.jpg"):
        _write_pattern(image_path, seed=0, flat=True)

    run_select(
        scene_dir,
        {
            "keep_ratio": 0.2,
            "min_blur_score": 1000.0,
            "max_temporal_gap": 1,
        },
    )

    metrics = json.loads((scene_dir / "metadata" / "selection_metrics.json").read_text())
    selected_frames = (scene_dir / "selected_frames.txt").read_text().strip().splitlines()

    assert metrics["fallback_used"] is True
    assert len(selected_frames) >= 2


def test_run_select_writes_revisit_candidates_when_pose_metadata_exists(tmp_path):
    from pipeline.select import run_select

    scene_dir = _build_scene(tmp_path, count=4)
    poses = {
        "frames": [
            {
                "image_name": f"frame_{index:04d}.jpg",
                "translation": [float(index), 0.0, 0.0],
                "confidence": 0.2 if index == 2 else 0.95,
            }
            for index in range(4)
        ]
    }
    (scene_dir / "metadata" / "input_poses.json").write_text(json.dumps(poses))

    run_select(scene_dir, {"keep_ratio": 0.5})

    metrics = json.loads((scene_dir / "metadata" / "selection_metrics.json").read_text())
    assert any(frame["baseline_diversity"] > 0 for frame in metrics["selected_frames"])

    revisit_path = scene_dir / "metadata" / "revisit_candidates.json"
    assert revisit_path.exists()

    revisit = json.loads(revisit_path.read_text())
    assert any(candidate["image_name"] == "frame_0002.jpg" for candidate in revisit["candidates"])


def test_run_select_writes_revisit_candidates_without_pose_metadata(tmp_path):
    from pipeline.select import run_select

    scene_dir = _build_scene(tmp_path, count=6)

    run_select(scene_dir, {"keep_ratio": 0.34})

    revisit_path = scene_dir / "metadata" / "revisit_candidates.json"
    assert revisit_path.exists()

    revisit = json.loads(revisit_path.read_text())
    assert any(candidate["reason"] == "coverage_gap" for candidate in revisit["candidates"])
