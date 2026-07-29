#!/usr/bin/env python3
"""Validate committed Gaussian Splatting benchmark evidence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "benchmarks" / "results"


def validate_record(record: dict[str, Any], source: Path) -> list[str]:
    errors: list[str] = []
    if record.get("schema_version") != 1:
        errors.append(f"{source}: schema_version must be 1")
    if record.get("evidence_status") not in {"historical_reported", "reproduced"}:
        errors.append(f"{source}: unsupported evidence_status")

    scenes = record.get("scenes")
    if not isinstance(scenes, list) or not scenes:
        return errors + [f"{source}: scenes must be a non-empty list"]

    for scene in scenes:
        name = scene.get("name", "<unnamed>")
        artifacts = scene.get("artifacts")
        if not isinstance(artifacts, list) or not artifacts:
            errors.append(f"{source}: {name} has no artifacts")
            continue
        originals = [item for item in artifacts if item.get("variant") == "original"]
        if len(originals) != 1:
            errors.append(f"{source}: {name} must have one original artifact")
            continue
        original_size = originals[0].get("size_mb")
        if not isinstance(original_size, (int, float)) or original_size <= 0:
            errors.append(f"{source}: {name} has an invalid original size")
            continue
        for artifact in artifacts:
            size = artifact.get("size_mb")
            percent = artifact.get("size_percent")
            if not isinstance(size, (int, float)) or size <= 0:
                errors.append(f"{source}: {name}/{artifact.get('variant')} has an invalid size")
                continue
            expected = round((size / original_size) * 100, 1)
            if not isinstance(percent, (int, float)) or abs(percent - expected) > 0.2:
                errors.append(
                    f"{source}: {name}/{artifact.get('variant')} "
                    f"reports {percent}% but sizes imply {expected}%"
                )
    return errors


def main() -> int:
    files = sorted(RESULTS_DIR.glob("*.json"))
    if not files:
        print("No benchmark result files found.")
        return 1

    errors: list[str] = []
    for path in files:
        record = json.loads(path.read_text(encoding="utf-8"))
        errors.extend(validate_record(record, path.relative_to(ROOT)))

    if errors:
        print("\n".join(errors))
        return 1
    print(f"Validated {len(files)} benchmark result file(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
