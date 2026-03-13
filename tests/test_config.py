import pytest
from pathlib import Path
from pipeline.config import load_config, merge_configs

def test_load_default_config():
    cfg = load_config()
    assert cfg["train"]["max_steps"] == 30000
    assert cfg["compress"]["sh_degree"] == 0

def test_merge_overrides():
    base = {"train": {"max_steps": 30000, "gpu": 0}}
    override = {"train": {"max_steps": 75000}}
    merged = merge_configs(base, override)
    assert merged["train"]["max_steps"] == 75000
    assert merged["train"]["gpu"] == 0

def test_load_custom_config(tmp_path):
    custom = tmp_path / "custom.yaml"
    custom.write_text("train:\n  max_steps: 50000\n")
    cfg = load_config(custom)
    assert cfg["train"]["max_steps"] == 50000
    assert cfg["preprocess"]["fps"] == 2
