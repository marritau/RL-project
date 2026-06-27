"""Pytest bootstrap for src-layout package imports."""

from __future__ import annotations

from pathlib import Path
import sys


def pytest_sessionstart(session) -> None:
    """Insert project src path into sys.path before tests execute."""
    del session
    project_root = Path(__file__).resolve().parents[1]
    src_path = project_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
