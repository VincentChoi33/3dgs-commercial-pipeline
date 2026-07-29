"""Step 2: Structure-from-Motion with LightGlue + COLMAP."""
from __future__ import annotations

import logging
import shutil
import struct
from copy import deepcopy
from pathlib import Path
from typing import Any

from pipeline.metadata import ensure_metadata_dir, read_optional_json, write_json_atomic

log = logging.getLogger("pipeline.sfm")

_RECONSTRUCT_DEFAULTS = {
    "feature": "superpoint_max",
    "matcher": "superpoint+lightglue",
    "camera_model": "SIMPLE_RADIAL",
    "resize_max": 1600,
    "max_keypoints": 4096,
    "convert_to_pinhole": True,
    "undistort": True,
    "pairing": {
        "strategy": "sequential+retrieval",
        "window": 10,
        "retrieval_num": 15,
    },
}


def find_largest_model(sparse_dir: Path) -> Path:
    """Find COLMAP sub-model with most registered images."""
    models = sorted(sparse_dir.iterdir())
    if not models:
        raise FileNotFoundError(f"No models in {sparse_dir}")

    best, best_count = None, 0
    for model_dir in models:
        images_bin = model_dir / "images.bin"
        if not images_bin.exists():
            continue
        with open(images_bin, "rb") as handle:
            count = struct.unpack("<Q", handle.read(8))[0]
        log.info("  Model %s: %s images", model_dir.name, count)
        if count > best_count:
            best, best_count = model_dir, count

    if best is None:
        raise FileNotFoundError(f"No valid models in {sparse_dir}")

    log.info("Selected model %s (%s images)", best.name, best_count)
    return best


def convert_cameras_to_pinhole(sparse_dir: Path):
    """Convert SIMPLE_RADIAL/SIMPLE_PINHOLE cameras to PINHOLE in cameras.bin."""
    cameras_bin = sparse_dir / "cameras.bin"
    if not cameras_bin.exists():
        raise FileNotFoundError(f"No cameras.bin in {sparse_dir}")

    param_counts = {0: 3, 1: 4, 2: 4, 3: 5}

    with open(cameras_bin, "rb") as handle:
        num_cameras = struct.unpack("<Q", handle.read(8))[0]
        cameras = []
        for _ in range(num_cameras):
            cam_id = struct.unpack("<i", handle.read(4))[0]
            model_id = struct.unpack("<i", handle.read(4))[0]
            width = struct.unpack("<Q", handle.read(8))[0]
            height = struct.unpack("<Q", handle.read(8))[0]
            n_params = param_counts.get(model_id)
            if n_params is None:
                raise ValueError(f"Unsupported camera model {model_id}")
            params = struct.unpack(f"<{n_params}d", handle.read(n_params * 8))

            if model_id == 0:  # SIMPLE_PINHOLE
                new_params = (params[0], params[0], params[1], params[2])
                log.info("  Camera %s: SIMPLE_PINHOLE → PINHOLE", cam_id)
            elif model_id == 2:  # SIMPLE_RADIAL
                new_params = (params[0], params[0], params[1], params[2])
                log.info("  Camera %s: SIMPLE_RADIAL (k=%.6f) → PINHOLE", cam_id, params[3])
            elif model_id == 3:  # RADIAL
                new_params = (params[0], params[0], params[1], params[2])
                log.info("  Camera %s: RADIAL → PINHOLE", cam_id)
            elif model_id == 1:  # Already PINHOLE
                new_params = params
                log.info("  Camera %s: already PINHOLE", cam_id)
            cameras.append((cam_id, 1, width, height, new_params))

    shutil.copy2(cameras_bin, str(cameras_bin) + ".bak")
    with open(cameras_bin, "wb") as handle:
        handle.write(struct.pack("<Q", num_cameras))
        for cam_id, model_id, width, height, params in cameras:
            handle.write(struct.pack("<i", cam_id))
            handle.write(struct.pack("<i", model_id))
            handle.write(struct.pack("<Q", width))
            handle.write(struct.pack("<Q", height))
            handle.write(struct.pack(f"<{len(params)}d", *params))

    log.info("Converted %s cameras to PINHOLE", num_cameras)


def _merge_config(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_config(merged[key], value)
        else:
            merged[key] = value
    return merged


def _read_image_count(images_dir: Path) -> int:
    return len([path for path in images_dir.iterdir() if path.is_file()])


def _normalize_pair(image_a: str, image_b: str) -> tuple[str, str] | None:
    if image_a == image_b:
        return None
    return tuple(sorted((image_a, image_b)))


def _generate_sequential_pairs(image_names: list[str], output_path: Path, window: int) -> Path:
    pairs: set[tuple[str, str]] = set()
    for index, image_name in enumerate(image_names):
        for neighbor in range(index + 1, min(index + 1 + max(window, 1), len(image_names))):
            pair = _normalize_pair(image_name, image_names[neighbor])
            if pair is not None:
                pairs.add(pair)
        step = 1
        while True:
            neighbor = index + 2**step
            if neighbor >= len(image_names):
                break
            pair = _normalize_pair(image_name, image_names[neighbor])
            if pair is not None:
                pairs.add(pair)
            step += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        for image_a, image_b in sorted(pairs):
            handle.write(f"{image_a} {image_b}\n")
    return output_path


def _read_pairs(path: Path) -> set[tuple[str, str]]:
    if not path.exists():
        return set()
    pairs = set()
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) != 2:
                continue
            pair = _normalize_pair(parts[0], parts[1])
            if pair is not None:
                pairs.add(pair)
    return pairs


def _write_pairs(path: Path, pairs: set[tuple[str, str]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for image_a, image_b in sorted(pairs):
            handle.write(f"{image_a} {image_b}\n")
    return path


def _copy_undistorted_outputs(undist_dir: Path, output_dir: Path, target_model: Path) -> dict[str, str | None]:
    undist_images_dir = undist_dir / "images"
    copied_images_dir = None
    if undist_images_dir.exists():
        final_images = output_dir / "images_undistorted"
        if final_images.exists():
            shutil.rmtree(final_images)
        shutil.copytree(undist_images_dir, final_images)
        copied_images_dir = str(final_images)

    undist_sparse_dir = undist_dir / "sparse"
    if undist_sparse_dir.exists():
        candidate_dirs = []
        if (undist_sparse_dir / "cameras.bin").exists():
            candidate_dirs.append(undist_sparse_dir)
        candidate_dirs.extend(
            subdir for subdir in sorted(undist_sparse_dir.iterdir()) if subdir.is_dir() and (subdir / "cameras.bin").exists()
        )
        if candidate_dirs:
            if target_model.exists():
                shutil.rmtree(target_model)
            shutil.copytree(candidate_dirs[0], target_model)

    return {"undistorted_images_dir": copied_images_dir}


def _load_pose_priors(metadata_dir: Path, image_names: list[str]) -> dict[str, Any]:
    pose_path = metadata_dir / "input_poses.json"
    base_report = {
        "path": str(pose_path),
        "used": False,
        "status": "ignored",
        "reason": "missing",
        "pair_strategy": None,
        "matched_frame_count": 0,
        "pose_pairs_added": [],
    }
    if not pose_path.exists():
        return {"entries": {}, "report": base_report, "fallback_used": False}

    payload = read_optional_json(pose_path)
    if not isinstance(payload, dict):
        log.warning("Failed to load pose priors from %s; continuing without them", pose_path)
        base_report["reason"] = "malformed_pose_priors"
        return {"entries": {}, "report": base_report, "fallback_used": True}

    frames = payload.get("frames")
    if not isinstance(frames, list):
        log.warning("Failed to load pose priors from %s; continuing without them", pose_path)
        base_report["reason"] = "malformed_pose_priors"
        return {"entries": {}, "report": base_report, "fallback_used": True}

    valid_entries = {}
    for frame in frames:
        if not isinstance(frame, dict):
            continue
        image_name = frame.get("image_name")
        translation = frame.get("translation")
        if not isinstance(image_name, str) or image_name not in image_names:
            continue
        if not (isinstance(translation, list) and len(translation) >= 3):
            continue
        try:
            coords = [float(translation[index]) for index in range(3)]
        except (TypeError, ValueError):
            continue
        confidence = frame.get("confidence", 1.0)
        try:
            confidence_value = float(confidence)
        except (TypeError, ValueError):
            confidence_value = 0.0
        valid_entries[image_name] = {"translation": coords, "confidence": confidence_value}

    if len(valid_entries) < 2:
        log.warning("Pose priors in %s were unusable; continuing with standard reconstruction", pose_path)
        base_report["reason"] = "insufficient_pose_priors"
        base_report["matched_frame_count"] = len(valid_entries)
        return {"entries": {}, "report": base_report, "fallback_used": True}

    base_report["reason"] = "available"
    base_report["matched_frame_count"] = len(valid_entries)
    return {"entries": valid_entries, "report": base_report, "fallback_used": False}


def _distance(a: list[float], b: list[float]) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2) ** 0.5


def _build_pose_assisted_pairs(image_names: list[str], pose_entries: dict[str, dict[str, Any]], window: int) -> list[list[str]]:
    added_pairs = []
    used_pairs: set[tuple[str, str]] = set()
    indexed_names = {name: index for index, name in enumerate(image_names)}
    ordered = [name for name in image_names if name in pose_entries]

    for image_name in ordered:
        current = pose_entries[image_name]
        if current["confidence"] < 0.5:
            continue
        best_name = None
        best_distance = None
        for candidate in ordered:
            if candidate == image_name:
                continue
            if pose_entries[candidate]["confidence"] < 0.5:
                continue
            if abs(indexed_names[candidate] - indexed_names[image_name]) <= max(window, 1):
                continue
            candidate_distance = _distance(current["translation"], pose_entries[candidate]["translation"])
            if candidate_distance > 0.5:
                continue
            if best_distance is None or candidate_distance < best_distance:
                best_name = candidate
                best_distance = candidate_distance
        if best_name is None:
            continue
        pair = _normalize_pair(image_name, best_name)
        if pair is None or pair in used_pairs:
            continue
        used_pairs.add(pair)
        added_pairs.append([pair[0], pair[1]])

    return sorted(added_pairs)


def _resolve_reconstruct_config(cfg: dict[str, Any]) -> dict[str, Any]:
    resolved = _merge_config(_RECONSTRUCT_DEFAULTS, cfg or {})
    pairing = resolved.setdefault("pairing", {})
    pairing.setdefault("strategy", _RECONSTRUCT_DEFAULTS["pairing"]["strategy"])
    pairing.setdefault("window", _RECONSTRUCT_DEFAULTS["pairing"]["window"])
    pairing.setdefault("retrieval_num", _RECONSTRUCT_DEFAULTS["pairing"]["retrieval_num"])
    return resolved


def run_reconstruct(images_dir: Path, output_dir: Path, cfg: dict):
    """Run reconstruction pipeline with improved defaults and observable metadata."""
    import pycolmap
    from hloc import extract_features, match_features, pairs_from_retrieval, reconstruction

    resolved_cfg = _resolve_reconstruct_config(cfg)
    metadata_dir = ensure_metadata_dir(output_dir)
    sparse_dir = output_dir / "sparse"
    sparse_dir.mkdir(parents=True, exist_ok=True)
    hloc_dir = output_dir / "hloc"
    hloc_dir.mkdir(parents=True, exist_ok=True)

    image_names = sorted(path.name for path in images_dir.iterdir() if path.is_file())
    image_count = len(image_names)
    fallback_used = False

    feature_conf = deepcopy(extract_features.confs[resolved_cfg["feature"]])
    feature_conf.setdefault("model", {})["max_keypoints"] = resolved_cfg["max_keypoints"]
    feature_conf.setdefault("preprocessing", {})["resize_max"] = resolved_cfg["resize_max"]

    matcher_name = resolved_cfg["matcher"]
    matcher_conf = deepcopy(match_features.confs[matcher_name])

    pairing_cfg = resolved_cfg["pairing"]
    pair_strategy = pairing_cfg["strategy"]
    sequential_pairs = _generate_sequential_pairs(image_names, hloc_dir / "pairs-sequential.txt", int(pairing_cfg["window"]))
    merged_pairs = _read_pairs(sequential_pairs)

    pose_state = _load_pose_priors(metadata_dir, image_names)
    pose_report = pose_state["report"]
    fallback_used = fallback_used or pose_state["fallback_used"]

    retrieval_pairs_path = None
    if "retrieval" in pair_strategy:
        retrieval_conf = deepcopy(extract_features.confs["netvlad"])
        retrieval_path = extract_features.main(retrieval_conf, images_dir, hloc_dir)
        retrieval_pairs_path = hloc_dir / "pairs-retrieval.txt"
        pairs_from_retrieval.main(retrieval_path, retrieval_pairs_path, num_matched=int(pairing_cfg["retrieval_num"]))
        merged_pairs.update(_read_pairs(retrieval_pairs_path))

    pose_pairs_added = []
    if pose_state["entries"]:
        pose_pairs_added = _build_pose_assisted_pairs(image_names, pose_state["entries"], int(pairing_cfg["window"]))
        if pose_pairs_added:
            merged_pairs.update((pair[0], pair[1]) for pair in pose_pairs_added)
            pose_report["used"] = True
            pose_report["status"] = "used"
            pose_report["reason"] = "pose_pairs_added"
            pose_report["pair_strategy"] = f"pose-assisted-{pair_strategy}"
            pose_report["pose_pairs_added"] = pose_pairs_added
        else:
            pose_report["status"] = "rejected"
            pose_report["reason"] = "no_pose_pairs_added"
            pose_report["pair_strategy"] = pair_strategy
            fallback_used = True
    else:
        pose_report["pair_strategy"] = pair_strategy

    pairs_path = _write_pairs(hloc_dir / "pairs.txt", merged_pairs)

    feature_path = extract_features.main(feature_conf, images_dir, hloc_dir)
    match_output = hloc_dir / f"{matcher_conf.get('output', 'matches')}.h5"
    match_path = match_features.main(
        matcher_conf,
        pairs_path,
        features=feature_path,
        matches=match_output,
    )

    sfm_temp = sparse_dir / "sfm_temp"
    sfm_temp.mkdir(parents=True, exist_ok=True)
    model = reconstruction.main(
        sfm_dir=sfm_temp,
        image_dir=images_dir,
        pairs=pairs_path,
        features=feature_path,
        matches=match_path,
        camera_mode=pycolmap.CameraMode.SINGLE,
    )

    target = sparse_dir / "0"
    if target.exists():
        shutil.rmtree(target)

    if hasattr(model, "write"):
        target.mkdir(parents=True, exist_ok=True)
        model.write(str(target))
    else:
        best_model = find_largest_model(sfm_temp)
        shutil.copytree(best_model, target)

    if sfm_temp.exists():
        shutil.rmtree(sfm_temp)

    outputs = {"sparse_model_dir": str(target), "undistorted_images_dir": None}
    if resolved_cfg.get("undistort", True):
        undist_dir = output_dir / "undistorted"
        if undist_dir.exists():
            shutil.rmtree(undist_dir)
        pycolmap.undistort_images(
            output_path=str(undist_dir),
            input_path=str(target),
            image_path=str(images_dir),
        )
        outputs.update(_copy_undistorted_outputs(undist_dir, output_dir, target))

    if resolved_cfg.get("convert_to_pinhole", True):
        convert_cameras_to_pinhole(target)

    selected_options = {
        "feature": resolved_cfg["feature"],
        "matcher": matcher_name,
        "resize_max": resolved_cfg["resize_max"],
        "max_keypoints": resolved_cfg["max_keypoints"],
        "undistort": bool(resolved_cfg.get("undistort", True)),
        "convert_to_pinhole": bool(resolved_cfg.get("convert_to_pinhole", True)),
        "pairing": {
            "strategy": pair_strategy,
            "window": int(pairing_cfg["window"]),
            "retrieval_num": int(pairing_cfg["retrieval_num"]),
        },
    }
    if pose_report["used"]:
        selected_options["pairing"]["strategy"] = pose_report["pair_strategy"]

    metrics = {
        "image_count": image_count,
        "selected_options": selected_options,
        "pose_priors_affected_reconstruction": bool(pose_report["used"]),
        "fallback_used": bool(fallback_used),
        "outputs": outputs,
    }
    write_json_atomic(metadata_dir / "reconstruction_metrics.json", metrics)

    if (metadata_dir / "input_poses.json").exists():
        write_json_atomic(metadata_dir / "pose_alignment_report.json", pose_report)

    log.info("Reconstruction complete → %s", target)
    return target


def run_sfm(images_dir: Path, output_dir: Path, cfg: dict):
    """Compatibility wrapper for reconstruct stage."""
    return run_reconstruct(images_dir, output_dir, cfg)
