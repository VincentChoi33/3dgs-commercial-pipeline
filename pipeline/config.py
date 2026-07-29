"""YAML config loading with defaults merging."""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import yaml

_DEFAULT_CONFIG = Path(__file__).parent.parent / "configs" / "default.yaml"
_LEGACY_SECTION_ALIASES = {
    "preprocess": "ingest",
    "sfm": "reconstruct",
}
_REQUIRED_MAPPING_SECTIONS = (
    "ingest",
    "select",
    "reconstruct",
    "train",
    "report",
)


def merge_configs(base: dict, override: dict) -> dict:
    """Deep-merge override into base."""
    merged = base.copy()
    for k, v in override.items():
        if k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
            merged[k] = merge_configs(merged[k], v)
        else:
            merged[k] = v
    return merged


def _expand_legacy_sections(cfg: dict) -> dict:
    expanded = cfg.copy()
    for legacy_key, canonical_key in _LEGACY_SECTION_ALIASES.items():
        if legacy_key in expanded:
            expanded[canonical_key] = merge_configs(
                expanded.get(canonical_key, {}), expanded[legacy_key]
            )
    return expanded


def _add_compat_aliases(cfg: dict) -> dict:
    compatible = cfg.copy()
    for legacy_key, canonical_key in _LEGACY_SECTION_ALIASES.items():
        compatible[legacy_key] = compatible.get(canonical_key, {})
    return compatible


def _require_mapping(value: object, path: str) -> None:
    if not isinstance(value, Mapping):
        raise ValueError(f"Config section '{path}' must be a mapping")


def _validate_config_shape(cfg: dict) -> dict:
    for section in _REQUIRED_MAPPING_SECTIONS:
        _require_mapping(cfg.get(section), section)

    _require_mapping(cfg["reconstruct"].get("pairing"), "reconstruct.pairing")
    _require_mapping(cfg["train"].get("phase_overrides"), "train.phase_overrides")
    return cfg


def load_config(config_path: Path | None = None) -> dict:
    """Load config: default.yaml merged with optional custom config."""
    with open(_DEFAULT_CONFIG) as f:
        cfg = yaml.safe_load(f) or {}

    cfg = _expand_legacy_sections(cfg)

    if config_path is not None:
        with open(config_path) as f:
            custom = yaml.safe_load(f) or {}
        cfg = merge_configs(cfg, _expand_legacy_sections(custom))

    return _add_compat_aliases(_validate_config_shape(cfg))
