"""Configuration loading (YAML) with defaults."""
from __future__ import annotations
import copy
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parent.parent


def _merge(base: dict, new: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (new or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> dict:
    """Load config.yaml (default one next to the project) and apply overrides."""
    default_path = ROOT / "config.yaml"
    with open(default_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if path and Path(path).resolve() != default_path.resolve():
        with open(path, "r", encoding="utf-8") as f:
            cfg = _merge(cfg, yaml.safe_load(f))
    if overrides:
        cfg = _merge(cfg, overrides)
    return cfg
