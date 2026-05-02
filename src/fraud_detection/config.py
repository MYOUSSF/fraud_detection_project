"""Config loader — reads configs/config.yaml and exposes typed helpers."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


_ROOT = Path(__file__).resolve().parents[2]  # repo root


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load YAML config.  Defaults to <repo_root>/configs/config.yaml."""
    if path is None:
        path = _ROOT / "configs" / "config.yaml"
    with open(path) as fh:
        cfg = yaml.safe_load(fh)
    _resolve_paths(cfg, _ROOT)
    return cfg


def _resolve_paths(cfg: dict, root: Path) -> None:
    """Make all path values absolute relative to repo root."""
    for key in ("raw_dir", "processed_dir", "models_dir", "outputs_dir", "mlruns_dir"):
        if key in cfg.get("paths", {}):
            cfg["paths"][key] = str(root / cfg["paths"][key])


def get_path(cfg: dict, key: str) -> Path:
    return Path(cfg["paths"][key])


def flat_params(cfg: dict) -> dict[str, Any]:
    """Return a flat key=value dict for MLflow param logging."""
    out: dict[str, Any] = {}
    for section, values in cfg.items():
        if isinstance(values, dict):
            for k, v in values.items():
                if not isinstance(v, (dict, list)):
                    out[f"{section}.{k}"] = v
    return out
