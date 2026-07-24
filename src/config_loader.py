"""Configuration loading utilities for ATCC."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load a YAML camera/pipeline config file.

    Args:
        config_path: Path to ``camera_config.yaml`` (or equivalent).

    Returns:
        Parsed configuration dictionary.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: If the file is empty or not a mapping.
    """
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Config not found: {path.resolve()}")

    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)

    if not isinstance(data, dict):
        raise ValueError(f"Config must be a YAML mapping: {path}")

    return data


def project_root() -> Path:
    """Return the repository root (parent of ``src/``)."""
    return Path(__file__).resolve().parent.parent


def resolve_path(path_str: str, base: Path | None = None) -> Path:
    """Resolve a path relative to the project root unless already absolute.

    Args:
        path_str: Path string from config.
        base: Optional base directory (defaults to project root).

    Returns:
        Absolute ``Path``.
    """
    path = Path(path_str)
    if path.is_absolute():
        return path
    root = base or project_root()
    return (root / path).resolve()
