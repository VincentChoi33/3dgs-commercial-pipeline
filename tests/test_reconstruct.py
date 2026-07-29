import json
import sys
import types
from pathlib import Path

import pytest


class _DummyModel:
    def __init__(self, source: Path):
        self.source = source

    def write(self, target: str) -> None:
        target_path = Path(target)
        target_path.mkdir(parents=True, exist_ok=True)
        for name in ("images.bin", "cameras.bin", "points3D.bin"):
            (target_path / name).write_bytes((self.source / name).read_bytes())


class _Recorder:
    def __init__(self, sparse_root: Path):
        self.sparse_root = sparse_root
        self.extract_calls = []
        self.match_calls = []
        self.reconstruction_calls = []
        self.undistort_calls = []

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        recorder = self

        def extract_main(conf, image_dir, output_dir):
            recorder.extract_calls.append(
                {
                    "conf": dict(conf),
                    "image_dir": Path(image_dir),
                    "output_dir": Path(output_dir),
                }
            )
            name = conf.get("output", conf.get("name", "features"))
            path = Path(output_dir) / f"{name}.h5"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("features")
            return path

        def match_main(conf, pairs, features=None, matches=None, output_dir=None):
            recorder.match_calls.append(
                {
                    "conf": dict(conf),
                    "pairs": Path(pairs),
                    "features": Path(features) if features is not None else None,
                    "matches": Path(matches) if matches is not None else None,
                    "output_dir": Path(output_dir) if output_dir is not None else None,
                }
            )
            path = Path(matches) if matches is not None else Path(output_dir) / "matches.h5"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("matches")
            return path

        def reconstruction_main(**kwargs):
            recorder.reconstruction_calls.append({k: Path(v) if isinstance(v, Path) else v for k, v in kwargs.items()})
            temp_model = kwargs["sfm_dir"] / "0"
            temp_model.mkdir(parents=True, exist_ok=True)
            for name in ("images.bin", "cameras.bin", "points3D.bin"):
                (temp_model / name).write_bytes((recorder.sparse_root / name).read_bytes())
            return _DummyModel(temp_model)

        def retrieval_main(retrieval_path, output_path, num_matched=15):
            Path(output_path).write_text("frame_0000.jpg frame_0002.jpg\n")
            return Path(output_path)

        def exhaustive_main(output_path, image_list=None):
            names = list(image_list or [])
            with open(output_path, "w", encoding="utf-8") as handle:
                for index, first in enumerate(names):
                    for second in names[index + 1 :]:
                        handle.write(f"{first} {second}\n")
            return Path(output_path)

        def undistort_images(*, output_path, input_path, image_path):
            recorder.undistort_calls.append(
                {
                    "output_path": Path(output_path),
                    "input_path": Path(input_path),
                    "image_path": Path(image_path),
                }
            )
            output = Path(output_path)
            (output / "images").mkdir(parents=True, exist_ok=True)
            (output / "sparse").mkdir(parents=True, exist_ok=True)
            for source in Path(image_path).glob("*.jpg"):
                (output / "images" / source.name).write_text("undistorted")
            for name in ("images.bin", "cameras.bin", "points3D.bin"):
                (output / "sparse" / name).write_bytes((Path(input_path) / name).read_bytes())

        extract_features = types.SimpleNamespace(
            confs={
                "superpoint": {"name": "superpoint", "output": "features", "model": {"max_keypoints": 2048}, "preprocessing": {"resize_max": 1024}},
                "superpoint_max": {"name": "superpoint_max", "output": "features-max", "model": {"max_keypoints": 4096}, "preprocessing": {"resize_max": 1024}},
                "netvlad": {"name": "netvlad", "output": "global-feats"},
            },
            main=extract_main,
        )
        match_features = types.SimpleNamespace(
            confs={
                "lightglue": {"output": "matches-lightglue"},
                "superpoint+lightglue": {"output": "matches-splg"},
                "superglue": {"output": "matches-superglue"},
            },
            main=match_main,
        )
        reconstruction = types.SimpleNamespace(main=reconstruction_main)
        pairs_from_retrieval = types.SimpleNamespace(main=retrieval_main)
        pairs_from_exhaustive = types.SimpleNamespace(main=exhaustive_main)
        hloc = types.SimpleNamespace(
            extract_features=extract_features,
            match_features=match_features,
            reconstruction=reconstruction,
            pairs_from_retrieval=pairs_from_retrieval,
            pairs_from_exhaustive=pairs_from_exhaustive,
        )
        pycolmap = types.SimpleNamespace(
            CameraMode=types.SimpleNamespace(SINGLE="single"),
            undistort_images=undistort_images,
        )

        monkeypatch.setitem(sys.modules, "hloc", hloc)
        monkeypatch.setitem(sys.modules, "hloc.extract_features", extract_features)
        monkeypatch.setitem(sys.modules, "hloc.match_features", match_features)
        monkeypatch.setitem(sys.modules, "hloc.reconstruction", reconstruction)
        monkeypatch.setitem(sys.modules, "hloc.pairs_from_retrieval", pairs_from_retrieval)
        monkeypatch.setitem(sys.modules, "hloc.pairs_from_exhaustive", pairs_from_exhaustive)
        monkeypatch.setitem(sys.modules, "pycolmap", pycolmap)


def _write_sparse_binary(path: Path, count: int) -> None:
    path.write_bytes(int(count).to_bytes(8, byteorder="little", signed=False))


def _build_scene(tmp_path: Path, image_count: int = 3) -> tuple[Path, Path]:
    scene_dir = tmp_path / "scene"
    images_dir = scene_dir / "images"
    metadata_dir = scene_dir / "metadata"
    sparse_root = tmp_path / "sparse-template"
    images_dir.mkdir(parents=True)
    metadata_dir.mkdir(parents=True)
    sparse_root.mkdir(parents=True)

    for index in range(image_count):
        (images_dir / f"frame_{index:04d}.jpg").write_text("image")

    _write_sparse_binary(sparse_root / "images.bin", image_count)
    _write_sparse_binary(sparse_root / "cameras.bin", 1)
    _write_sparse_binary(sparse_root / "points3D.bin", 12)
    return scene_dir, sparse_root


def test_run_reconstruct_writes_metrics_and_pose_report_with_pose_aware_pairs(tmp_path, monkeypatch):
    from pipeline.sfm import run_reconstruct

    scene_dir, sparse_root = _build_scene(tmp_path, image_count=3)
    poses = {
        "frames": [
            {"image_name": "frame_0000.jpg", "translation": [0.0, 0.0, 0.0], "confidence": 0.95},
            {"image_name": "frame_0001.jpg", "translation": [10.0, 0.0, 0.0], "confidence": 0.95},
            {"image_name": "frame_0002.jpg", "translation": [0.05, 0.0, 0.0], "confidence": 0.95},
        ]
    }
    (scene_dir / "metadata" / "input_poses.json").write_text(json.dumps(poses))

    recorder = _Recorder(sparse_root)
    recorder.install(monkeypatch)

    monkeypatch.setattr("pipeline.sfm.convert_cameras_to_pinhole", lambda _sparse_dir: None)

    result = run_reconstruct(scene_dir / "images", scene_dir, {"pairing": {"window": 1}})

    assert result == scene_dir / "sparse" / "0"
    metrics = json.loads((scene_dir / "metadata" / "reconstruction_metrics.json").read_text())
    assert metrics["image_count"] == 3
    assert metrics["registered_images"] == 3
    assert metrics["camera_count"] == 1
    assert metrics["point3d_count"] == 12
    assert metrics["selected_options"]["feature"] == "superpoint_max"
    assert metrics["selected_options"]["pairing"]["strategy"] == "pose-assisted-sequential+retrieval"
    assert metrics["pose_priors_affected_reconstruction"] is True
    assert metrics["fallback_used"] is False

    report = json.loads((scene_dir / "metadata" / "pose_alignment_report.json").read_text())
    assert report["status"] == "used"
    assert report["used"] is True
    assert report["pair_strategy"] == "pose-assisted-sequential+retrieval"
    assert any(pair == ["frame_0000.jpg", "frame_0002.jpg"] for pair in report["pose_pairs_added"])

    pairs_path = recorder.match_calls[0]["pairs"]
    pairs_lines = pairs_path.read_text().strip().splitlines()
    assert "frame_0000.jpg frame_0002.jpg" in pairs_lines
    assert len(recorder.undistort_calls) == 1


def test_run_reconstruct_warns_and_falls_back_on_malformed_pose_priors(tmp_path, monkeypatch, caplog):
    from pipeline.sfm import run_reconstruct

    scene_dir, sparse_root = _build_scene(tmp_path, image_count=2)
    (scene_dir / "metadata" / "input_poses.json").write_text("{not-json}")

    recorder = _Recorder(sparse_root)
    recorder.install(monkeypatch)
    monkeypatch.setattr("pipeline.sfm.convert_cameras_to_pinhole", lambda _sparse_dir: None)

    run_reconstruct(scene_dir / "images", scene_dir, {})

    metrics = json.loads((scene_dir / "metadata" / "reconstruction_metrics.json").read_text())
    assert metrics["pose_priors_affected_reconstruction"] is False
    assert metrics["fallback_used"] is True

    report = json.loads((scene_dir / "metadata" / "pose_alignment_report.json").read_text())
    assert report["status"] == "ignored"
    assert report["reason"] == "malformed_pose_priors"
    assert report["used"] is False
    assert "Failed to load pose priors" in caplog.text


def test_run_reconstruct_warns_and_falls_back_on_low_confidence_pose_priors(tmp_path, monkeypatch, caplog):
    from pipeline.sfm import run_reconstruct

    scene_dir, sparse_root = _build_scene(tmp_path, image_count=2)
    poses = {
        "frames": [
            {"image_name": "frame_0000.jpg", "translation": [0.0, 0.0, 0.0], "confidence": 0.2},
            {"image_name": "frame_0001.jpg", "translation": [0.1, 0.0, 0.0], "confidence": 0.3},
        ]
    }
    (scene_dir / "metadata" / "input_poses.json").write_text(json.dumps(poses))

    recorder = _Recorder(sparse_root)
    recorder.install(monkeypatch)
    monkeypatch.setattr("pipeline.sfm.convert_cameras_to_pinhole", lambda _sparse_dir: None)

    run_reconstruct(scene_dir / "images", scene_dir, {})

    metrics = json.loads((scene_dir / "metadata" / "reconstruction_metrics.json").read_text())
    assert metrics["pose_priors_affected_reconstruction"] is False
    assert metrics["fallback_used"] is True
    assert metrics["registered_images"] == 2

    report = json.loads((scene_dir / "metadata" / "pose_alignment_report.json").read_text())
    assert report["status"] == "rejected"
    assert report["reason"] == "low_confidence_pose_priors"
    assert report["used"] is False
    assert report["usable_frame_count"] == 0
    assert report["rejected_frame_count"] == 2
    assert report["rejected_images"] == [
        {"image_name": "frame_0000.jpg", "reason": "low_confidence", "confidence": 0.2},
        {"image_name": "frame_0001.jpg", "reason": "low_confidence", "confidence": 0.3},
    ]
    assert "all low-confidence" in caplog.text


def test_run_reconstruct_keeps_undistort_independent_from_pinhole_conversion(tmp_path, monkeypatch):
    from pipeline import sfm

    scene_dir, sparse_root = _build_scene(tmp_path, image_count=2)
    recorder = _Recorder(sparse_root)
    recorder.install(monkeypatch)

    convert_calls = []
    monkeypatch.setattr(sfm, "convert_cameras_to_pinhole", lambda sparse_dir: convert_calls.append(Path(sparse_dir)))

    run_result = sfm.run_reconstruct(
        scene_dir / "images",
        scene_dir,
        {"undistort": False, "convert_to_pinhole": True, "pairing": {"strategy": "sequential"}},
    )

    assert run_result == scene_dir / "sparse" / "0"
    assert len(recorder.undistort_calls) == 0
    assert convert_calls == [scene_dir / "sparse" / "0"]

    metrics = json.loads((scene_dir / "metadata" / "reconstruction_metrics.json").read_text())
    assert metrics["selected_options"]["undistort"] is False
    assert metrics["selected_options"]["convert_to_pinhole"] is True
    assert metrics["outputs"]["undistorted_images_dir"] is None
