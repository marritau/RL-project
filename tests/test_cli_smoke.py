"""Smoke tests for command-line scripts."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys


def run_command(command: list[str]) -> None:
    """Execute command and fail test on non-zero exit code."""
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        raise AssertionError(
            "Command failed:\n"
            f"CMD: {' '.join(command)}\n"
            f"STDOUT:\n{completed.stdout}\n"
            f"STDERR:\n{completed.stderr}"
        )


def test_train_eval_and_play_smoke(tmp_path: Path) -> None:
    """Run train/eval/play scripts with tiny settings to verify wiring."""
    train_dir = tmp_path / "train"
    eval_dir = tmp_path / "eval"

    run_command(
        [
            sys.executable,
            "scripts/train.py",
            "--config",
            "configs/default.yaml",
            "--episodes",
            "2",
            "--output-dir",
            str(train_dir),
        ]
    )

    assert (train_dir / "model.pkl").exists()
    assert (train_dir / "train_metrics.json").exists()
    assert (train_dir / "learning_curve.png").exists()

    run_command(
        [
            sys.executable,
            "scripts/eval.py",
            "--config",
            "configs/default.yaml",
            "--episodes",
            "2",
            "--output-dir",
            str(eval_dir),
            "--model",
            str(train_dir / "model.pkl"),
        ]
    )

    assert (eval_dir / "eval_metrics.json").exists()

    run_command(
        [
            sys.executable,
            "scripts/play.py",
            "--config",
            "configs/default.yaml",
            "--render-mode",
            "ansi",
            "--episodes",
            "1",
            "--no-gif",
        ]
    )

    run_command(
        [
            sys.executable,
            "scripts/play.py",
            "--config",
            "configs/default.yaml",
            "--render-mode",
            "ansi",
            "--episodes",
            "1",
            "--policy",
            "baseline",
            "--no-gif",
        ]
    )
