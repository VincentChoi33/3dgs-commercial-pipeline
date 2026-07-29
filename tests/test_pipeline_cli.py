from argparse import Namespace
import importlib.util
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_PIPELINE_SPEC = importlib.util.spec_from_file_location("pipeline_cli", _REPO_ROOT / "pipeline.py")
pipeline_cli = importlib.util.module_from_spec(_PIPELINE_SPEC)
assert _PIPELINE_SPEC.loader is not None
_PIPELINE_SPEC.loader.exec_module(pipeline_cli)


def _config():
    return {
        "ingest": {"stage": "ingest"},
        "select": {"stage": "select"},
        "reconstruct": {"stage": "reconstruct"},
        "train": {"stage": "train"},
        "export": {"stage": "export"},
        "compress": {"stage": "compress"},
        "report": {"write_summary": True},
        "preprocess": {"stage": "preprocess"},
        "sfm": {"stage": "reconstruct"},
    }


def test_cmd_run_calls_new_stage_order(monkeypatch, tmp_path):
    calls = []
    scene_dir = tmp_path / "output" / "scene"
    input_path = tmp_path / "input"

    monkeypatch.setattr(pipeline_cli, "load_config", lambda _path: _config())
    monkeypatch.setattr(
        pipeline_cli,
        "_load_ingest_entrypoint",
        lambda: lambda inp, out, cfg: calls.append(("ingest", inp, out, cfg)) or out / "images",
    )
    monkeypatch.setattr(
        pipeline_cli,
        "_load_select_entrypoint",
        lambda: lambda out, cfg: calls.append(("select", out, cfg)) or out / "selected_images",
    )
    monkeypatch.setattr(
        pipeline_cli,
        "_load_reconstruct_entrypoint",
        lambda: lambda images, out, cfg: calls.append(("reconstruct", images, out, cfg)),
    )
    monkeypatch.setattr(
        pipeline_cli,
        "_load_train_entrypoint",
        lambda: lambda out, cfg, name: calls.append(("train", out, cfg, name)),
    )
    monkeypatch.setattr(
        pipeline_cli,
        "_load_export_entrypoint",
        lambda: lambda out, cfg: calls.append(("export", out, cfg)) or out / "full.ply",
    )
    monkeypatch.setattr(
        pipeline_cli,
        "_load_compress_entrypoint",
        lambda: lambda full_ply, out, cfg: calls.append(("compress", full_ply, out, cfg)),
    )
    monkeypatch.setattr(
        pipeline_cli,
        "_load_report_entrypoint",
        lambda: lambda out, cfg: calls.append(("report", out, cfg)),
    )

    args = Namespace(config=None, input=str(input_path), output=str(tmp_path / "output"), name="scene")
    pipeline_cli.cmd_run(args)

    assert [call[0] for call in calls] == [
        "ingest",
        "select",
        "reconstruct",
        "train",
        "export",
        "compress",
        "report",
    ]
    assert calls[0][1] == input_path
    assert calls[0][2] == scene_dir
    assert calls[1][1] == scene_dir
    assert calls[2][1] == scene_dir / "selected_images"
    assert calls[2][2] == scene_dir
    assert calls[4][1] == scene_dir
    assert calls[5][1] == scene_dir / "full.ply"
    assert calls[6][1] == scene_dir


def test_cmd_report_calls_report_entrypoint(monkeypatch, tmp_path):
    called = {}
    scene_dir = tmp_path / "scene"

    monkeypatch.setattr(pipeline_cli, "load_config", lambda _path: _config())
    monkeypatch.setattr(
        pipeline_cli,
        "_load_report_entrypoint",
        lambda: lambda out, cfg: called.update({"out": out, "cfg": cfg}) or out / "report",
    )

    result = pipeline_cli.cmd_report(Namespace(config=None, output=str(scene_dir)))

    assert result == scene_dir / "report"
    assert called == {"out": scene_dir, "cfg": _config()["report"]}


def test_preprocess_command_remains_compatibility_surface(monkeypatch, tmp_path):
    called = {}

    monkeypatch.setattr(pipeline_cli, "load_config", lambda _path: _config())
    monkeypatch.setattr(
        pipeline_cli,
        "_load_preprocess_entrypoint",
        lambda: lambda inp, out, cfg: called.update({"inp": inp, "out": out, "cfg": cfg}) or out / "selected_images",
    )

    args = Namespace(config=None, input=str(tmp_path / "input"), output=str(tmp_path / "scene"))
    result = pipeline_cli.cmd_preprocess(args)

    assert result == Path(args.output) / "selected_images"
    assert called == {
        "inp": Path(args.input),
        "out": Path(args.output),
        "cfg": _config()["preprocess"],
    }


def test_sfm_command_remains_compatibility_alias(monkeypatch, tmp_path):
    called = {}

    monkeypatch.setattr(pipeline_cli, "load_config", lambda _path: _config())
    monkeypatch.setattr(
        pipeline_cli,
        "_load_reconstruct_entrypoint",
        lambda: lambda inp, out, cfg: called.update({"inp": inp, "out": out, "cfg": cfg}) or out / "sparse" / "0",
    )

    args = Namespace(config=None, input=str(tmp_path / "selected_images"), output=str(tmp_path / "scene"))
    result = pipeline_cli.cmd_sfm(args)

    assert result == Path(args.output) / "sparse" / "0"
    assert called == {
        "inp": Path(args.input),
        "out": Path(args.output),
        "cfg": _config()["reconstruct"],
    }


def test_parser_exposes_new_and_compatibility_commands():
    parser = pipeline_cli.build_parser()
    subparsers_action = next(
        action for action in parser._actions if getattr(action, "dest", None) == "command"
    )

    commands = set(subparsers_action.choices)
    assert {"run", "ingest", "select", "reconstruct", "report"}.issubset(commands)
    assert {"preprocess", "sfm"}.issubset(commands)


def test_report_command_uses_placeholder_when_report_module_missing(tmp_path):
    report_dir = pipeline_cli._compat_run_report(tmp_path, {"write_summary": True})

    assert report_dir == tmp_path / "report"
    assert (report_dir / "summary.txt").exists()


def test_load_report_entrypoint_reraises_nested_module_not_found(monkeypatch):
    def _raise_nested_missing(_module_name):
        raise ModuleNotFoundError("missing nested dependency", name="markdown")

    monkeypatch.setattr(pipeline_cli.importlib, "import_module", _raise_nested_missing)

    try:
        pipeline_cli._load_report_entrypoint()
    except ModuleNotFoundError as exc:
        assert exc.name == "markdown"
    else:
        raise AssertionError("Expected nested ModuleNotFoundError to be re-raised")
