import json
from pathlib import Path

import cv2
import numpy as np


def _write_image(path: Path, value: int) -> None:
    image = np.full((24, 24, 3), value, dtype=np.uint8)
    cv2.imwrite(str(path), image)


def test_run_ingest_copies_photos_and_writes_metadata(tmp_path):
    from pipeline.ingest import run_ingest

    photos_dir = tmp_path / "photos"
    photos_dir.mkdir()
    _write_image(photos_dir / "a.jpg", 64)
    _write_image(photos_dir / "b.png", 128)

    scene_dir = tmp_path / "scene"
    result = run_ingest(photos_dir, scene_dir, {"pose_path": None})

    assert result == scene_dir / "images"
    assert (scene_dir / "images" / "a.jpg").exists()
    assert (scene_dir / "images" / "b.png").exists()

    frames = json.loads((scene_dir / "metadata" / "frames.json").read_text())
    assert [frame["image_name"] for frame in frames] == ["a.jpg", "b.png"]
    assert frames[0]["frame_index"] == 0

    capture_info = json.loads((scene_dir / "metadata" / "capture_info.json").read_text())
    assert capture_info["source_type"] == "photos"
    assert capture_info["image_count"] == 2
    assert capture_info["has_pose_input"] is False


def test_run_ingest_imports_pose_metadata(tmp_path):
    from pipeline.ingest import run_ingest

    photos_dir = tmp_path / "photos"
    photos_dir.mkdir()
    _write_image(photos_dir / "frame_0000.jpg", 64)

    pose_path = tmp_path / "poses.json"
    poses = {
        "frames": [
            {
                "image_name": "frame_0000.jpg",
                "translation": [0.0, 0.0, 0.0],
                "confidence": 0.9,
            }
        ]
    }
    pose_path.write_text(json.dumps(poses))

    scene_dir = tmp_path / "scene"
    run_ingest(photos_dir, scene_dir, {"pose_path": str(pose_path)})

    imported = json.loads((scene_dir / "metadata" / "input_poses.json").read_text())
    assert imported["frames"][0]["image_name"] == "frame_0000.jpg"
    assert imported["frames"][0]["confidence"] == 0.9

    capture_info = json.loads((scene_dir / "metadata" / "capture_info.json").read_text())
    assert capture_info["has_pose_input"] is True


def test_run_ingest_warns_and_continues_on_malformed_pose_input(tmp_path, caplog):
    from pipeline.ingest import run_ingest

    photos_dir = tmp_path / "photos"
    photos_dir.mkdir()
    _write_image(photos_dir / "frame_0000.jpg", 64)

    pose_path = tmp_path / "poses.json"
    pose_path.write_text("{not-json}")

    scene_dir = tmp_path / "scene"
    result = run_ingest(photos_dir, scene_dir, {"pose_path": str(pose_path)})

    assert result == scene_dir / "images"
    assert (scene_dir / "images" / "frame_0000.jpg").exists()
    assert not (scene_dir / "metadata" / "input_poses.json").exists()
    assert "Failed to load pose metadata" in caplog.text
