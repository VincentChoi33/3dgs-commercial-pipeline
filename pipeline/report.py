from __future__ import annotations

from pathlib import Path
from typing import Any

from pipeline.metadata import read_optional_json, write_json_atomic


_METRICS_SOURCES: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("capture", "capture_info.json", ("source_type", "image_count")),
    ("selection", "selection_metrics.json", ("selected_count", "fallback_used")),
    ("coverage", "coverage_summary.json", ("coverage_signal", "novelty_mean", "pose_frames_available")),
    (
        "reconstruction",
        "reconstruction_metrics.json",
        (
            "registered_images",
            "camera_count",
            "point3d_count",
            "pose_priors_affected_reconstruction",
            "fallback_used",
        ),
    ),
    ("training", "training_metrics.json", ("psnr", "ssim", "lpips", "best_step")),
    ("export", "export_report.json", ("gaussian_count", "output_path", "output_size_bytes")),
    ("compression", "compression_report.json", ("output_files",)),
)

_DECISION_TRACE_FILES: tuple[str, ...] = (
    "capture_info",
    "selection_metrics",
    "coverage_summary",
    "reconstruction_metrics",
    "pose_alignment_report",
    "training_metrics",
    "export_report",
    "compression_report",
    "revisit_candidates",
)


def _read_metadata(scene_dir: Path, stem: str) -> Any | None:
    return read_optional_json(scene_dir / "metadata" / f"{stem}.json")


def _select_fields(payload: Any, fields: tuple[str, ...]) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    selected = {field: payload[field] for field in fields if field in payload}
    return selected or None


def _collect_metrics(scene_dir: Path) -> tuple[dict[str, Any], dict[str, bool], dict[str, Any]]:
    metrics: dict[str, Any] = {}
    payloads: dict[str, Any] = {}
    found_files: dict[str, bool] = {}

    for stage, filename, fields in _METRICS_SOURCES:
        stem = Path(filename).stem
        payload = _read_metadata(scene_dir, stem)
        payloads[stem] = payload
        found_files[stem] = payload is not None
        selected = _select_fields(payload, fields)
        if selected is not None:
            metrics[stage] = selected

    for stem in _DECISION_TRACE_FILES:
        found_files.setdefault(stem, _read_metadata(scene_dir, stem) is not None)
        payloads.setdefault(stem, _read_metadata(scene_dir, stem))

    if not any(found_files.values()):
        raise FileNotFoundError(f"No known metadata files found in {scene_dir / 'metadata'}")

    return metrics, found_files, payloads


def _artifact_flags(scene_dir: Path, payloads: dict[str, Any]) -> dict[str, bool]:
    compression_report = payloads.get("compression_report") if isinstance(payloads.get("compression_report"), dict) else {}
    output_files = compression_report.get("output_files") if isinstance(compression_report, dict) else None
    compressed_paths = []
    if isinstance(output_files, list):
        compressed_paths = [Path(entry["path"]) for entry in output_files if isinstance(entry, dict) and isinstance(entry.get("path"), str)]

    reconstruction_metrics = payloads.get("reconstruction_metrics") if isinstance(payloads.get("reconstruction_metrics"), dict) else {}
    outputs = reconstruction_metrics.get("outputs") if isinstance(reconstruction_metrics, dict) else None
    sparse_model_dir = None
    if isinstance(outputs, dict) and isinstance(outputs.get("sparse_model_dir"), str):
        sparse_model_dir = Path(outputs["sparse_model_dir"])

    return {
        "full_ply": (scene_dir / "full.ply").exists(),
        "compressed_ply": any(path.name == "compressed.ply" and path.exists() for path in compressed_paths)
        or (scene_dir / "compressed.ply").exists(),
        "downsampled_ply": any(path.exists() and path.name.startswith("compressed_ds") for path in compressed_paths),
        "sparse_model": sparse_model_dir.exists() if sparse_model_dir is not None else (scene_dir / "sparse" / "0").exists(),
    }


def _reconstruction_fallback_reason(reconstruction_metrics: Any, pose_report: Any) -> str | None:
    if not isinstance(reconstruction_metrics, dict) or not reconstruction_metrics.get("fallback_used"):
        return None
    if isinstance(pose_report, dict) and isinstance(pose_report.get("reason"), str):
        return pose_report["reason"]
    return "reported_in_reconstruction_metrics"


def _decision_trace(scene_dir: Path, found_files: dict[str, bool], payloads: dict[str, Any]) -> dict[str, Any]:
    capture_info = payloads.get("capture_info") if isinstance(payloads.get("capture_info"), dict) else {}
    reconstruction_metrics = payloads.get("reconstruction_metrics") if isinstance(payloads.get("reconstruction_metrics"), dict) else {}
    pose_report = payloads.get("pose_alignment_report") if isinstance(payloads.get("pose_alignment_report"), dict) else {}

    pose_reason = pose_report.get("reason") if isinstance(pose_report.get("reason"), str) else None

    return {
        "metadata_files": found_files,
        "artifacts": _artifact_flags(scene_dir, payloads),
        "decisions": {
            "pose_input_present": bool(capture_info.get("has_pose_input") or found_files.get("pose_alignment_report")),
            "pose_priors_affected_reconstruction": reconstruction_metrics.get("pose_priors_affected_reconstruction"),
            "malformed_pose_handling_present": pose_reason == "malformed_pose_priors",
            "pose_prior_rejection_reason": pose_reason if pose_report.get("used") is not True else None,
            "reconstruction_fallback_reason": _reconstruction_fallback_reason(reconstruction_metrics, pose_report),
        },
    }


def _summary_lines(metrics: dict[str, Any], trace: dict[str, Any]) -> list[str]:
    lines = ["# Pipeline Report", ""]

    lines.append("## Available stages")
    for stage in ("capture", "selection", "coverage", "reconstruction", "training", "export", "compression"):
        stage_metrics = metrics.get(stage)
        if stage_metrics is None:
            continue
        lines.append(f"- {stage}: {stage_metrics}")

    artifacts = trace["artifacts"]
    available_outputs = [name for name, present in artifacts.items() if present]
    if available_outputs:
        lines.extend(["", "## Outputs", f"- {', '.join(available_outputs)}"])

    decisions = trace["decisions"]
    notable = []
    if decisions.get("pose_input_present"):
        notable.append("pose input metadata present")
    if decisions.get("pose_priors_affected_reconstruction") is True:
        notable.append("pose priors influenced reconstruction")
    if decisions.get("pose_prior_rejection_reason"):
        notable.append(f"pose prior outcome: {decisions['pose_prior_rejection_reason']}")
    if decisions.get("reconstruction_fallback_reason"):
        notable.append(f"reconstruction fallback: {decisions['reconstruction_fallback_reason']}")

    if notable:
        lines.extend(["", "## Notable decisions", *[f"- {entry}" for entry in notable]])

    lines.append("")
    return lines


def run_report(scene_dir: Path, cfg: dict) -> Path:
    metrics, found_files, payloads = _collect_metrics(scene_dir)
    trace = _decision_trace(scene_dir, found_files, payloads)

    report_dir = scene_dir / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(report_dir / "metrics.json", metrics)
    write_json_atomic(report_dir / "decision_trace.json", trace)

    if cfg.get("write_summary", True):
        summary = "\n".join(_summary_lines(metrics, trace))
        (report_dir / "summary.md").write_text(summary, encoding="utf-8")

    return report_dir
