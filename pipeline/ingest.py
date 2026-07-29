from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

from pipeline.metadata import ensure_metadata_dir, read_optional_json, write_json_atomic

log = logging.getLogger("pipeline.ingest")

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
IMAGE_EXTS = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")


def is_video(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in VIDEO_EXTS


def extract_frames(video_path: Path, output_dir: Path, fps: int = 2) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-i",
        str(video_path),
        "-vf",
        f"fps={fps}",
        "-q:v",
        "1",
        str(output_dir / "frame_%04d.jpg"),
        "-y",
    ]
    log.info("Extracting frames at %s fps: %s", fps, video_path)
    subprocess.run(cmd, check=True, capture_output=True)
    return output_dir


def _copy_images(input_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for pattern in IMAGE_EXTS:
        for image_path in sorted(input_dir.glob(pattern)):
            shutil.copy2(image_path, output_dir / image_path.name)


def _list_images(images_dir: Path) -> list[Path]:
    image_paths: list[Path] = []
    for pattern in IMAGE_EXTS:
        image_paths.extend(sorted(images_dir.glob(pattern)))
    return sorted({path.name: path for path in image_paths}.values(), key=lambda path: path.name)


def _write_frame_metadata(images_dir: Path, metadata_dir: Path) -> list[dict]:
    frames = [
        {
            "frame_index": index,
            "image_name": image_path.name,
            "image_path": str(image_path),
        }
        for index, image_path in enumerate(_list_images(images_dir))
    ]
    write_json_atomic(metadata_dir / "frames.json", frames)
    return frames


def _maybe_import_poses(pose_path_value: str | Path | None, metadata_dir: Path) -> dict | None:
    if not pose_path_value:
        return None

    pose_path = Path(pose_path_value)
    pose_data = read_optional_json(pose_path)
    if pose_data is None:
        if pose_path.exists():
            log.warning("Failed to load pose metadata from %s; continuing without poses", pose_path)
        return None

    write_json_atomic(metadata_dir / "input_poses.json", pose_data)
    return pose_data


def run_ingest(input_path: Path, scene_dir: Path, cfg: dict) -> Path:
    images_dir = scene_dir / "images"
    metadata_dir = ensure_metadata_dir(scene_dir)

    if is_video(input_path):
        extract_frames(input_path, images_dir, fps=cfg.get("fps", 2))
        source_type = "video"
    else:
        _copy_images(input_path, images_dir)
        source_type = "photos"

    frames = _write_frame_metadata(images_dir, metadata_dir)
    pose_data = _maybe_import_poses(cfg.get("pose_path"), metadata_dir)

    capture_info = {
        "source_type": source_type,
        "input_path": str(input_path),
        "image_count": len(frames),
        "has_pose_input": pose_data is not None,
    }
    write_json_atomic(metadata_dir / "capture_info.json", capture_info)

    log.info("Ingested %s images into %s", len(frames), images_dir)
    return images_dir
