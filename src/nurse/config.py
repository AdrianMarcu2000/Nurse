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


@lru_cache(maxsize=1)
def get_config() -> dict[str, Any]:
    cfg = _load(ROOT / "config" / "default.yaml")
    local = ROOT / "config" / "local.yaml"
    if local.exists():
        import collections.abc

        def deep_merge(base: dict, override: dict) -> dict:
            out = dict(base)
            for k, v in override.items():
                if isinstance(v, dict) and isinstance(out.get(k), dict):
                    out[k] = deep_merge(out[k], v)
                else:
                    out[k] = v
            return out

        cfg = deep_merge(cfg, _load(local))
    return cfg


@lru_cache(maxsize=1)
def get_persona() -> dict[str, Any]:
    return _load(ROOT / "config" / "persona.yaml")


def resolve(relative_path: str) -> Path:
    """Resolve a path relative to project root."""
    return ROOT / relative_path
