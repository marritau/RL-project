"""Tests for baseline evaluation protocol and metrics."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys


def _load_eval_module():
    """Import scripts/eval.py as a module for pure-function tests."""
    project_root = Path(__file__).resolve().parents[1]
    module_path = project_root / "scripts" / "eval.py"
    spec = importlib.util.spec_from_file_location("eval_script", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load scripts/eval.py for testing.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_resolve_eval_schedule_defaults_for_baseline() -> None:
    """Use baseline defaults (episodes=200, base_seed=42) when no overrides exist."""
    eval_module = _load_eval_module()
    episodes, base_seed = eval_module.resolve_eval_schedule(
        policy_name="baseline",
        args_episodes=None,
        args_seed=None,
        eval_cfg={},
        env_seed=999,
    )
    assert episodes == 200
    assert base_seed == 42


def test_build_episode_seeds_for_baseline_protocol() -> None:
    """Build deterministic seed schedule as base_seed + i."""
    eval_module = _load_eval_module()
    seeds = eval_module.build_episode_seeds(base_seed=42, episodes=5)
    assert seeds == [42, 43, 44, 45, 46]


def test_eval_cli_baseline_writes_expected_metrics(tmp_path: Path) -> None:
    """Run baseline eval CLI and verify output metrics include new baseline fields."""
    output_dir = tmp_path / "eval_baseline"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/eval.py",
            "--config",
            "configs/default.yaml",
            "--policy",
            "baseline",
            "--episodes",
            "2",
            "--seed",
            "42",
            "--output-dir",
            str(output_dir),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise AssertionError(
            "Baseline eval command failed:\n"
            f"STDOUT:\n{completed.stdout}\n"
            f"STDERR:\n{completed.stderr}"
        )

    metrics_path = output_dir / "eval_metrics.json"
    assert metrics_path.exists()
    with metrics_path.open(encoding="utf-8") as handle:
        metrics = json.load(handle)

    assert metrics["policy"] == "baseline"
    assert metrics["base_seed"] == 42
    assert "win_rate" in metrics
    assert "avg_reward" in metrics
    assert "avg_episode_length" in metrics
    assert len(metrics["returns"]) == 2
