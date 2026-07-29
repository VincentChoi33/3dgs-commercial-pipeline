#!/usr/bin/env python3
"""Gaussian Splatting Pipeline — end-to-end video/photos to compressed PLY."""
import argparse
import importlib
import logging
from pathlib import Path

from pipeline.config import load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("pipeline")


def _load_ingest_entrypoint():
    from pipeline.ingest import run_ingest

    return run_ingest


def _load_select_entrypoint():
    from pipeline.select import run_select

    return run_select


def _load_preprocess_entrypoint():
    from pipeline.preprocess import run_preprocess

    return run_preprocess


def _load_reconstruct_entrypoint():
    sfm_module = importlib.import_module("pipeline.sfm")
    return getattr(sfm_module, "run_reconstruct", sfm_module.run_sfm)


def _load_train_entrypoint():
    from pipeline.train import run_train

    return run_train


def _load_export_entrypoint():
    from pipeline.export import run_export

    return run_export


def _load_compress_entrypoint():
    from pipeline.compress import run_compress

    return run_compress


def _compat_run_report(scene_dir: Path, cfg: dict):
    report_dir = scene_dir / "report"
    report_dir.mkdir(parents=True, exist_ok=True)

    if cfg.get("write_summary"):
        summary_path = report_dir / "summary.txt"
        if not summary_path.exists():
            summary_path.write_text("Report stage placeholder. Detailed report implementation pending.\n")

    log.info(f"Report stage placeholder complete → {report_dir}")
    return report_dir


def _load_report_entrypoint():
    try:
        report_module = importlib.import_module("pipeline.report")
    except ModuleNotFoundError:
        return _compat_run_report
    return getattr(report_module, "run_report", _compat_run_report)


def cmd_run(args):
    """Run full pipeline: ingest → select → reconstruct → train → export → compress → report."""
    cfg = load_config(args.config)
    inp = Path(args.input)
    out = Path(args.output) / args.name
    out.mkdir(parents=True, exist_ok=True)

    log.info(f"Pipeline start: {inp} → {out}")

    images_dir = _load_ingest_entrypoint()(inp, out, cfg["ingest"])
    selected_images_dir = _load_select_entrypoint()(out, cfg["select"])
    _load_reconstruct_entrypoint()(selected_images_dir, out, cfg["reconstruct"])
    _load_train_entrypoint()(out, cfg["train"], name=args.name)
    full_ply = _load_export_entrypoint()(out, cfg["export"])
    _load_compress_entrypoint()(full_ply, out, cfg["compress"])
    _load_report_entrypoint()(out, cfg["report"])

    log.info(f"Pipeline complete: {out}")
    return images_dir


def cmd_ingest(args):
    cfg = load_config(args.config)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    return _load_ingest_entrypoint()(Path(args.input), out, cfg["ingest"])


def cmd_select(args):
    cfg = load_config(args.config)
    return _load_select_entrypoint()(Path(args.output), cfg["select"])


def cmd_preprocess(args):
    cfg = load_config(args.config)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    return _load_preprocess_entrypoint()(Path(args.input), out, cfg["preprocess"])


def cmd_reconstruct(args):
    cfg = load_config(args.config)
    out = Path(args.output)
    return _load_reconstruct_entrypoint()(Path(args.input), out, cfg["reconstruct"])


def cmd_sfm(args):
    return cmd_reconstruct(args)


def cmd_train(args):
    cfg = load_config(args.config)
    out = Path(args.output)
    return _load_train_entrypoint()(out, cfg["train"], name=args.name or out.name)


def cmd_export(args):
    cfg = load_config(args.config)
    out = Path(args.output)
    return _load_export_entrypoint()(out, cfg["export"])


def cmd_compress(args):
    cfg = load_config(args.config)
    inp = Path(args.input)
    out = Path(args.output)
    return _load_compress_entrypoint()(inp, out, cfg["compress"])


def cmd_report(args):
    cfg = load_config(args.config)
    out = Path(args.output)
    return _load_report_entrypoint()(out, cfg["report"])


def build_parser():
    parser = argparse.ArgumentParser(description="Gaussian Splatting Pipeline")
    parser.add_argument("--config", type=Path, default=None, help="Custom config YAML")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="Run full pipeline")
    p_run.add_argument("--input", "-i", required=True, help="Video file or image directory")
    p_run.add_argument("--output", "-o", required=True, help="Output base directory")
    p_run.add_argument("--name", "-n", required=True, help="Scene name")
    p_run.set_defaults(func=cmd_run)

    p_ingest = sub.add_parser("ingest", help="Normalize video or photos into scene images")
    p_ingest.add_argument("--input", "-i", required=True, help="Video file or image directory")
    p_ingest.add_argument("--output", "-o", required=True, help="Scene directory")
    p_ingest.set_defaults(func=cmd_ingest)

    p_select = sub.add_parser("select", help="Select useful views into selected_images")
    p_select.add_argument("--output", "-o", required=True, help="Scene directory")
    p_select.set_defaults(func=cmd_select)

    p_pre = sub.add_parser("preprocess", help="Compatibility alias for ingest + select")
    p_pre.add_argument("--input", "-i", required=True)
    p_pre.add_argument("--output", "-o", required=True)
    p_pre.set_defaults(func=cmd_preprocess)

    p_reconstruct = sub.add_parser("reconstruct", help="Run reconstruction (LightGlue + COLMAP)")
    p_reconstruct.add_argument("--input", "-i", required=True, help="selected_images directory")
    p_reconstruct.add_argument("--output", "-o", required=True, help="Scene directory")
    p_reconstruct.set_defaults(func=cmd_reconstruct)

    p_sfm = sub.add_parser("sfm", help="Compatibility alias for reconstruct")
    p_sfm.add_argument("--input", "-i", required=True, help="selected_images directory")
    p_sfm.add_argument("--output", "-o", required=True)
    p_sfm.set_defaults(func=cmd_sfm)

    p_train = sub.add_parser("train", help="Train 3DGS")
    p_train.add_argument("--output", "-o", required=True, help="Scene directory (with sparse/0)")
    p_train.add_argument("--name", "-n", default=None)
    p_train.set_defaults(func=cmd_train)

    p_export = sub.add_parser("export", help="Export ckpt → PLY")
    p_export.add_argument("--output", "-o", required=True, help="Scene directory")
    p_export.set_defaults(func=cmd_export)

    p_compress = sub.add_parser("compress", help="Compress PLY")
    p_compress.add_argument("--input", "-i", required=True, help="Full PLY path")
    p_compress.add_argument("--output", "-o", required=True, help="Output directory")
    p_compress.set_defaults(func=cmd_compress)

    p_report = sub.add_parser("report", help="Write pipeline report artifacts")
    p_report.add_argument("--output", "-o", required=True, help="Scene directory")
    p_report.set_defaults(func=cmd_report)

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    main()
