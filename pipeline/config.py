"""YAML config loading with defaults merging."""
import yaml
from pathlib import Path

_DEFAULT_CONFIG = Path(__file__).parent.parent / "configs" / "default.yaml"


def merge_configs(base: dict, override: dict) -> dict:
    """Deep-merge override into base."""
    merged = base.copy()
    for k, v in override.items():
        if k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
            merged[k] = merge_configs(merged[k], v)
        else:
            merged[k] = v
    return merged


def load_config(config_path: Path | None = None) -> dict:
    """Load config: default.yaml merged with optional custom config."""
    with open(_DEFAULT_CONFIG) as f:
        cfg = yaml.safe_load(f)

    if config_path is not None:
        with open(config_path) as f:
            custom = yaml.safe_load(f) or {}
        cfg = merge_configs(cfg, custom)

    return cfg
