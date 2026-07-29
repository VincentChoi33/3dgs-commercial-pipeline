from __future__ import annotations

import shutil
from pathlib import Path

import cv2
import numpy as np

from pipeline.metadata import ensure_metadata_dir, read_optional_json, write_json_atomic
from pipeline.preprocess import compute_blur_score


def _load_frames(scene_dir: Path) -> list[dict]:
    metadata_dir = ensure_metadata_dir(scene_dir)
    frames = read_optional_json(metadata_dir / "frames.json")
    if not isinstance(frames, list) or not frames:
        raise FileNotFoundError(f"Missing frame metadata in {metadata_dir / 'frames.json'}")
    return frames


def _load_pose_map(scene_dir: Path) -> dict[str, dict]:
    metadata_dir = ensure_metadata_dir(scene_dir)
    pose_data = read_optional_json(metadata_dir / "input_poses.json")
    if not isinstance(pose_data, dict):
        return {}
    frames = pose_data.get("frames")
    if not isinstance(frames, list):
        return {}
    return {
        frame["image_name"]: frame
        for frame in frames
        if isinstance(frame, dict) and isinstance(frame.get("image_name"), str)
    }


def _coverage_signal(frame_index: int, frame_count: int, pose_entry: dict | None) -> float:
    if frame_count <= 1:
        base_signal = 1.0
    else:
        midpoint = (frame_count - 1) / 2
        base_signal = abs(frame_index - midpoint) / max(midpoint, 1.0)

    translation_signal = 0.0
    if pose_entry is not None:
        translation = pose_entry.get("translation")
        if isinstance(translation, list) and len(translation) >= 3:
            translation_signal = float(np.linalg.norm(np.asarray(translation[:3], dtype=float)))

    return base_signal + (0.1 * translation_signal)


def _score_frame(image_path: Path, frame_index: int, frame_count: int, pose_entry: dict | None) -> dict:
    image = cv2.imread(str(image_path))
    blur_score = compute_blur_score(image)
    temporal_spacing = min(frame_index, max(frame_count - frame_index - 1, 0))
    coverage_signal = _coverage_signal(frame_index, frame_count, pose_entry)
    pose_confidence = 1.0
    if pose_entry is not None and isinstance(pose_entry.get("confidence"), (int, float)):
        pose_confidence = float(pose_entry["confidence"])

    total_score = blur_score + (temporal_spacing * 25.0) + (coverage_signal * 50.0) + (pose_confidence * 10.0)
    return {
        "image_name": image_path.name,
        "frame_index": frame_index,
        "blur_score": blur_score,
        "temporal_spacing": temporal_spacing,
        "coverage_signal": coverage_signal,
        "pose_confidence": pose_confidence,
        "total_score": total_score,
    }


def _copy_selected_images(scene_dir: Path, selected_frames: list[dict]) -> Path:
    selected_dir = scene_dir / "selected_images"
    if selected_dir.exists():
        shutil.rmtree(selected_dir)
    selected_dir.mkdir(parents=True, exist_ok=True)

    images_dir = scene_dir / "images"
    for frame in selected_frames:
        image_name = frame["image_name"]
        shutil.copy2(images_dir / image_name, selected_dir / image_name)
    return selected_dir


def _fallback_frames(scored_frames: list[dict], keep_count: int) -> list[dict]:
    keep_count = max(2, keep_count)
    ranked = sorted(scored_frames, key=lambda frame: (frame["blur_score"], -frame["frame_index"]), reverse=True)
    return sorted(ranked[:keep_count], key=lambda frame: frame["frame_index"])


def run_select(scene_dir: Path, cfg: dict) -> Path:
    frames = _load_frames(scene_dir)
    pose_map = _load_pose_map(scene_dir)
    frame_count = len(frames)
    keep_ratio = float(cfg.get("keep_ratio", 1.0))
    target_keep_count = max(1, int(np.ceil(frame_count * keep_ratio)))
    min_blur_score = cfg.get("min_blur_score")
    max_temporal_gap = cfg.get("max_temporal_gap")

    scored_frames = []
    for frame in frames:
        image_name = frame["image_name"]
        image_path = scene_dir / "images" / image_name
        scored_frames.append(
            _score_frame(
                image_path,
                int(frame.get("frame_index", len(scored_frames))),
                frame_count,
                pose_map.get(image_name),
            )
        )

    ranked = sorted(scored_frames, key=lambda frame: frame["total_score"], reverse=True)
    selected = ranked[:target_keep_count]

    if min_blur_score is not None:
        selected = [frame for frame in selected if frame["blur_score"] >= float(min_blur_score)]

    if max_temporal_gap is not None and selected:
        filtered = [selected[0]]
        for frame in sorted(selected[1:], key=lambda item: item["frame_index"]):
            if frame["frame_index"] - filtered[-1]["frame_index"] >= int(max_temporal_gap):
                filtered.append(frame)
        selected = filtered

    minimum_safe_count = max(2, min(frame_count, target_keep_count))
    fallback_used = len(selected) < minimum_safe_count
    if fallback_used:
        selected = _fallback_frames(scored_frames, minimum_safe_count)
    else:
        selected = sorted(selected, key=lambda frame: frame["frame_index"])

    selected_dir = _copy_selected_images(scene_dir, selected)
    selected_names = [frame["image_name"] for frame in selected]
    (scene_dir / "selected_frames.txt").write_text("\n".join(selected_names) + "\n")

    metadata_dir = ensure_metadata_dir(scene_dir)
    selection_metrics = {
        "input_count": frame_count,
        "selected_count": len(selected),
        "fallback_used": fallback_used,
        "keep_ratio": keep_ratio,
        "selected_frames": selected,
    }
    write_json_atomic(metadata_dir / "selection_metrics.json", selection_metrics)

    coverage_summary = {
        "coverage_signal": float(np.mean([frame["coverage_signal"] for frame in selected])) if selected else 0.0,
        "novelty_mean": float(np.mean([frame["temporal_spacing"] for frame in selected])) if selected else 0.0,
        "pose_frames_available": len(pose_map),
    }
    write_json_atomic(metadata_dir / "coverage_summary.json", coverage_summary)

    if pose_map:
        revisit_candidates = {
            "candidates": [
                {
                    "image_name": frame["image_name"],
                    "reason": "low_pose_confidence",
                    "pose_confidence": frame["pose_confidence"],
                }
                for frame in scored_frames
                if frame["pose_confidence"] < 0.5
            ]
        }
        write_json_atomic(metadata_dir / "revisit_candidates.json", revisit_candidates)

    return selected_dir
