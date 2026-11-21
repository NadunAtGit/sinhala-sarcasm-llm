"""YAML configuration: configs/base.yaml + configs/conditions/<id>.yaml."""
from __future__ import annotations

from pathlib import Path

import yaml


def load_yaml(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_base(root: str | Path) -> dict:
    return load_yaml(Path(root) / "configs" / "base.yaml")


def load_condition(root: str | Path, cond_id: str) -> dict:
    cfg = load_yaml(Path(root) / "configs" / "conditions" / f"{cond_id.lower()}.yaml")
    if cfg.get("id", "").upper() != cond_id.upper():
        raise ValueError(f"{cond_id}.yaml declares id {cfg.get('id')!r}")
    return cfg