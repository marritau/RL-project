"""Configuration and serialization helpers for scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import pickle

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load YAML file into a dictionary."""
    path_obj = Path(path)
    with path_obj.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping at top-level YAML: {path_obj}")
    return data


def ensure_directory(path: str | Path) -> Path:
    """Create directory if missing and return a Path instance."""
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def save_json(payload: dict[str, Any], path: str | Path) -> None:
    """Serialize dictionary to JSON with stable formatting."""
    path_obj = Path(path)
    with path_obj.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def save_pickle(payload: Any, path: str | Path) -> None:
    """Serialize arbitrary Python object as pickle."""
    path_obj = Path(path)
    with path_obj.open("wb") as handle:
        pickle.dump(payload, handle)


def load_pickle(path: str | Path) -> Any:
    """Load pickled object from disk."""
    path_obj = Path(path)
    with path_obj.open("rb") as handle:
        return pickle.load(handle)
