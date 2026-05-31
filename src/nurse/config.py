"""Loads and exposes typed config from YAML files."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

# Project root is two levels up from this file (src/nurse/config.py → project root)
ROOT = Path(__file__).resolve().parents[2]


def _load(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


# Runtime overrides set before startup (e.g. by a CLI flag). Merged last, above
# default.yaml and local.yaml, so a launch-time choice wins.
_OVERRIDES: dict[str, Any] = {}


def set_overrides(overrides: dict[str, Any]) -> None:
    """Apply in-process config overrides (deep-merged on top of files). Must be called
    before get_config() is first read; clears the cache so the change takes effect."""
    global _OVERRIDES
    _OVERRIDES = _deep_merge(_OVERRIDES, overrides)
    get_config.cache_clear()


@lru_cache(maxsize=1)
def get_config() -> dict[str, Any]:
    cfg = _load(ROOT / "config" / "default.yaml")
    local = ROOT / "config" / "local.yaml"
    if local.exists():
        cfg = _deep_merge(cfg, _load(local))
    if _OVERRIDES:
        cfg = _deep_merge(cfg, _OVERRIDES)
    return cfg


@lru_cache(maxsize=1)
def get_persona() -> dict[str, Any]:
    return _load(ROOT / "config" / "persona.yaml")


def resolve(relative_path: str) -> Path:
    """Resolve a path relative to project root."""
    return ROOT / relative_path
