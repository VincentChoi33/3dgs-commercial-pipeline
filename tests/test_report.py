import json

from pipeline.metadata import read_optional_json, write_json_atomic
from pipeline.report import run_report


def test_run_report_writes_summary_and_metrics(tmp_path):
    metadata_dir = tmp_path / "metadata"
    write_json_atomic(
        metadata_dir / "capture_info.json",
        {
            "source_type": "photos",
            "image_count": 4,
            "input_path": "/tmp/input",
            "has_pose_input": True,
        },
    )
    write_json_atomic(
        metadata_dir / "selection_metrics.json",
        {
            "input_count": 4,
            "selected_count": 3,
            "fallback_used": False,
        },
    )

    report_dir = run_report(tmp_path, {"write_summary": True})

    summary_path = report_dir / "summary.md"
    metrics_path = report_dir / "metrics.json"
    assert summary_path.exists()
    assert metrics_path.exists()
    assert "# Pipeline Report" in summary_path.read_text()


def test_run_report_aggregates_known_metadata_files(tmp_path):
    metadata_dir = tmp_path / "metadata"
    write_json_atomic(metadata_dir / "capture_info.json", {"source_type": "video", "image_count": 12})
    write_json_atomic(metadata_dir / "selection_metrics.json", {"selected_count": 8, "fallback_used": True})
    write_json_atomic(
        metadata_dir / "reconstruction_metrics.json",
        {
            "registered_images": 7,
            "point3d_count": 123,
            "pose_priors_affected_reconstruction": False,
            "fallback_used": True,
        },
    )
    write_json_atomic(
        metadata_dir / "training_metrics.json",
        {"psnr": 28.5, "ssim": 0.93, "lpips": 0.07},
    )
    write_json_atomic(
        metadata_dir / "export_report.json",
        {"gaussian_count": 45000, "output_path": str(tmp_path / "full.ply")},
    )
    write_json_atomic(
        metadata_dir / "compression_report.json",
        {
            "output_files": [
                {"path": str(tmp_path / "compressed.ply"), "gaussian_count": 23000, "output_size_bytes": 1024}
            ]
        },
    )

    run_report(tmp_path, {"write_summary": True})
    metrics = read_optional_json(tmp_path / "report" / "metrics.json")

    assert metrics == {
        "capture": {"source_type": "video", "image_count": 12},
        "selection": {"selected_count": 8, "fallback_used": True},
        "reconstruction": {
            "registered_images": 7,
            "point3d_count": 123,
            "pose_priors_affected_reconstruction": False,
            "fallback_used": True,
        },
        "training": {"psnr": 28.5, "ssim": 0.93, "lpips": 0.07},
        "export": {"gaussian_count": 45000, "output_path": str(tmp_path / "full.ply")},
        "compression": {
            "output_files": [
                {"path": str(tmp_path / "compressed.ply"), "gaussian_count": 23000, "output_size_bytes": 1024}
            ]
        },
    }


def test_run_report_writes_decision_trace_for_found_artifacts_and_fallbacks(tmp_path):
    metadata_dir = tmp_path / "metadata"
    write_json_atomic(metadata_dir / "capture_info.json", {"has_pose_input": True})
    write_json_atomic(
        metadata_dir / "reconstruction_metrics.json",
        {
            "pose_priors_affected_reconstruction": False,
            "fallback_used": True,
            "outputs": {"sparse_model_dir": str(tmp_path / "sparse" / "0")},
        },
    )
    write_json_atomic(
        metadata_dir / "pose_alignment_report.json",
        {
            "status": "rejected",
            "reason": "low_confidence_pose_priors",
            "used": False,
            "rejected_frame_count": 2,
        },
    )

    (tmp_path / "full.ply").write_text("ply")
    (tmp_path / "compressed.ply").write_text("ply")

    run_report(tmp_path, {"write_summary": True})
    trace = json.loads((tmp_path / "report" / "decision_trace.json").read_text())

    assert trace["metadata_files"]["capture_info"] is True
    assert trace["metadata_files"]["reconstruction_metrics"] is True
    assert trace["metadata_files"]["pose_alignment_report"] is True
    assert trace["artifacts"]["full_ply"] is True
    assert trace["artifacts"]["compressed_ply"] is True
    assert trace["decisions"]["pose_input_present"] is True
    assert trace["decisions"]["pose_priors_affected_reconstruction"] is False
    assert trace["decisions"]["malformed_pose_handling_present"] is False
    assert trace["decisions"]["pose_prior_rejection_reason"] == "low_confidence_pose_priors"
    assert trace["decisions"]["reconstruction_fallback_reason"] == "low_confidence_pose_priors"
