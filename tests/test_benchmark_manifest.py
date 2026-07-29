import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "validate_benchmarks.py"
SPEC = importlib.util.spec_from_file_location("validate_benchmarks", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_reported_benchmark_manifest_is_consistent() -> None:
    path = ROOT / "benchmarks" / "results" / "reported_rtx4090.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    assert MODULE.validate_record(record, path) == []


def test_benchmark_validator_rejects_inconsistent_percentages(tmp_path: Path) -> None:
    record = {
        "schema_version": 1,
        "evidence_status": "reproduced",
        "scenes": [
            {
                "name": "tiny",
                "artifacts": [
                    {"variant": "original", "size_mb": 100, "size_percent": 100},
                    {"variant": "compressed", "size_mb": 25, "size_percent": 80}
                ]
            }
        ]
    }
    errors = MODULE.validate_record(record, tmp_path / "bad.json")
    assert any("sizes imply 25.0%" in error for error in errors)
