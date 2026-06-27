"""Run TVS for supported policies and create one comparison table/plots.

This script expects trained artifacts for q_learning, sarsa, vi and pi to exist.
The important methodological detail is that, by default, it uses one shared sigma
for all policies.  The TVS paper calibrates sigma once per environment; using a
separate sigma per policy would make cross-policy TVS values less comparable.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import matplotlib.pyplot as plt
import numpy as np

# Import from sibling script after adding project root to path.
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from evaluate_tvs import POLICY_DEFAULTS, run_tvs_evaluation  # noqa: E402


POLICY_ORDER = ["baseline", "random", "q_learning", "sarsa", "vi", "pi"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate TVS for every policy and summarize results.")
    parser.add_argument("--episodes", type=int, default=128, help="Rollouts per policy.")
    parser.add_argument("--seed", type=int, default=42, help="Base seed for all TVS rollouts.")
    parser.add_argument("--output-dir", default="results/tvs_all", help="Summary output directory.")
    parser.add_argument(
        "--quality-filter",
        default="wins_or_all",
        choices=["all", "wins", "wins_or_all", "non_loss", "top_return"],
        help="Trajectory filter passed to each TVS run.",
    )
    parser.add_argument("--top-fraction", type=float, default=0.5)
    parser.add_argument("--max-points-per-trajectory", type=int, default=160)
    parser.add_argument("--band-ratio", type=float, default=0.2, help="Paper default is 0.2.")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--save-trajectories", action="store_true")
    parser.add_argument(
        "--policies",
        nargs="+",
        default=POLICY_ORDER,
        choices=POLICY_ORDER,
        help="Subset of policies to evaluate.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if a trained model artifact is missing instead of skipping it.",
    )
    parser.add_argument(
        "--fallback-on-zero-q",
        action="store_true",
        help="For Q/SARSA only: use heuristic fallback when all legal Q-values are zero.",
    )
    parser.add_argument(
        "--per-policy-sigma",
        action="store_true",
        help=(
            "Calibrate sigma separately for every policy. This is useful for diagnostics, "
            "but shared sigma is preferred for cross-policy comparisons."
        ),
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=None,
        help="Explicit sigma to use for all policies. Overrides automatic shared calibration.",
    )
    parser.add_argument(
        "--sigma-source-policy",
        choices=POLICY_ORDER,
        default="vi",
        help=(
            "Policy used to calibrate the shared environment sigma when --sigma is not set. "
            "The paper calibrates sigma once per environment; VI is the default because it is "
            "usually the strongest solver in this Pacman project."
        ),
    )
    parser.add_argument(
        "--sigma-episodes",
        type=int,
        default=None,
        help="Rollouts for shared-sigma calibration; defaults to min(episodes, 128).",
    )
    return parser.parse_args()


def ensure_dir(path: str | Path) -> Path:
    result = Path(path)
    result.mkdir(parents=True, exist_ok=True)
    return result


def model_exists(policy: str) -> bool:
    model = POLICY_DEFAULTS[policy].get("model") or ""
    return not model or Path(model).exists()


def compact_row(metrics: dict[str, Any], status: str = "ok", message: str = "") -> dict[str, Any]:
    collected = metrics.get("collected_rollouts", {})
    scored = metrics.get("scored_rollouts", {})
    tvs = metrics.get("tvs", {})
    return {
        "policy": metrics.get("policy"),
        "status": status,
        "message": message,
        "episodes_collected": collected.get("episodes"),
        "episodes_scored": scored.get("episodes"),
        "win_rate": collected.get("win_rate"),
        "mean_return": collected.get("mean_return"),
        "mean_score": collected.get("mean_score"),
        "mean_length": collected.get("mean_length"),
        "state_coverage": collected.get("state_coverage"),
        "occupancy_entropy": collected.get("occupancy_entropy"),
        "action_entropy": collected.get("action_entropy"),
        "scored_win_rate": scored.get("win_rate"),
        "scored_mean_return": scored.get("mean_return"),
        "scored_state_coverage": scored.get("state_coverage"),
        "scored_occupancy_entropy": scored.get("occupancy_entropy"),
        "scored_action_entropy": scored.get("action_entropy"),
        "tvs": tvs.get("tvs"),
        "sigma": tvs.get("sigma"),
        "tvs_compute_seconds": metrics.get("tvs_compute_seconds"),
        "quality_filter": metrics.get("quality_filter"),
        "config_path": metrics.get("config_path"),
        "model_path": metrics.get("model_path"),
    }


def save_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fieldnames = list(rows[0].keys()) if rows else ["policy", "status"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _ok_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if row.get("status") == "ok" and row.get("tvs") is not None]


def save_summary_plot(rows: list[dict[str, Any]], path: Path) -> None:
    ok_rows = _ok_rows(rows)
    if not ok_rows:
        return
    labels = [str(row["policy"]) for row in ok_rows]
    tvs_values = [float(row["tvs"]) for row in ok_rows]
    win_rates = [float(row["win_rate"] or 0.0) for row in ok_rows]

    fig, ax1 = plt.subplots(figsize=(10, 4))
    x = list(range(len(labels)))
    ax1.bar([v - 0.18 for v in x], tvs_values, width=0.36, label="TVS")
    ax1.set_ylabel("TVS")
    ax1.set_xticks(x, labels, rotation=25, ha="right")
    ax1.grid(True, axis="y", alpha=0.25)

    ax2 = ax1.twinx()
    ax2.bar([v + 0.18 for v in x], win_rates, width=0.36, label="Win rate")
    ax2.set_ylabel("Win rate")
    ax2.set_ylim(0, max(1.0, max(win_rates) * 1.2))

    fig.suptitle("TVS vs. task performance")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_quality_diversity_scatter(rows: list[dict[str, Any]], path: Path) -> None:
    ok_rows = _ok_rows(rows)
    if not ok_rows:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    for row in ok_rows:
        x = float(row["tvs"])
        y = float(row.get("win_rate") or 0.0)
        ax.scatter([x], [y], s=80)
        ax.annotate(str(row["policy"]), (x, y), xytext=(5, 5), textcoords="offset points")
    ax.set_xlabel("TVS (trajectory diversity)")
    ax.set_ylabel("Win rate (task quality)")
    ax.set_title("Quality–diversity view")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_reward_diversity_scatter(rows: list[dict[str, Any]], path: Path) -> None:
    ok_rows = _ok_rows(rows)
    if not ok_rows:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    for row in ok_rows:
        x = float(row["tvs"])
        y = float(row.get("mean_return") or 0.0)
        ax.scatter([x], [y], s=80)
        ax.annotate(str(row["policy"]), (x, y), xytext=(5, 5), textcoords="offset points")
    ax.set_xlabel("TVS (trajectory diversity)")
    ax.set_ylabel("Mean return")
    ax.set_title("Return–diversity view")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_baseline_metrics_plot(rows: list[dict[str, Any]], path: Path) -> None:
    ok_rows = _ok_rows(rows)
    if not ok_rows:
        return
    labels = [str(row["policy"]) for row in ok_rows]
    metrics = [
        ("tvs", "TVS"),
        ("state_coverage", "Coverage"),
        ("occupancy_entropy", "Occupancy entropy"),
        ("action_entropy", "Action entropy"),
    ]
    x = np.arange(len(labels))
    width = 0.18
    fig, ax = plt.subplots(figsize=(11, 4.8))
    for idx, (key, label) in enumerate(metrics):
        values = np.array([float(row.get(key) or 0.0) for row in ok_rows], dtype=float)
        denom = float(np.max(values)) if np.max(values) > 0 else 1.0
        ax.bar(x + (idx - 1.5) * width, values / denom, width=width, label=label)
    ax.set_xticks(x, labels, rotation=25, ha="right")
    ax.set_ylabel("Value normalized by max across policies")
    ax.set_title("TVS compared with coverage/entropy baselines")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def skipped_row(policy: str, message: str, quality_filter: str) -> dict[str, Any]:
    model = POLICY_DEFAULTS[policy].get("model") or ""
    return {
        "policy": policy,
        "status": "skipped",
        "message": message,
        "episodes_collected": None,
        "episodes_scored": None,
        "win_rate": None,
        "mean_return": None,
        "mean_score": None,
        "mean_length": None,
        "state_coverage": None,
        "occupancy_entropy": None,
        "action_entropy": None,
        "scored_win_rate": None,
        "scored_mean_return": None,
        "scored_state_coverage": None,
        "scored_occupancy_entropy": None,
        "scored_action_entropy": None,
        "tvs": None,
        "sigma": None,
        "tvs_compute_seconds": None,
        "quality_filter": quality_filter,
        "config_path": POLICY_DEFAULTS[policy].get("config"),
        "model_path": model,
    }


def calibrate_shared_sigma(args: argparse.Namespace, output_dir: Path) -> tuple[float | None, dict[str, Any] | None]:
    if args.per_policy_sigma:
        return None, None
    if args.sigma is not None:
        return float(args.sigma), {"mode": "explicit", "sigma": float(args.sigma)}

    source = str(args.sigma_source_policy)
    if not model_exists(source):
        # Fall back to baseline because it has no model artifact.
        source = "baseline"
    sigma_episodes = int(args.sigma_episodes or min(int(args.episodes), 128))
    sigma_output = output_dir / "_sigma_calibration"
    print(f"Calibrating shared sigma from policy={source}, episodes={sigma_episodes} -> {sigma_output}")
    metrics = run_tvs_evaluation(
        policy_name=source,
        output_dir=str(sigma_output),
        episodes=sigma_episodes,
        seed=int(args.seed),
        quality_filter=args.quality_filter,
        top_fraction=float(args.top_fraction),
        band_ratio=float(args.band_ratio),
        max_points_per_trajectory=int(args.max_points_per_trajectory),
        no_plots=bool(args.no_plots),
        save_trajectories=False,
        fallback_on_zero_q=bool(args.fallback_on_zero_q),
    )
    sigma = float(metrics["tvs"]["sigma"])
    return sigma, {
        "mode": "shared_from_policy",
        "source_policy": source,
        "episodes": sigma_episodes,
        "sigma": sigma,
        "metrics_path": str(sigma_output / "tvs_metrics.json"),
    }


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    rows: list[dict[str, Any]] = []
    full_metrics: dict[str, Any] = {}

    shared_sigma, sigma_info = calibrate_shared_sigma(args, output_dir)

    for policy in args.policies:
        if not model_exists(policy):
            model = POLICY_DEFAULTS[policy].get("model") or ""
            message = f"missing model artifact: {model}"
            if args.strict:
                raise FileNotFoundError(message)
            print(f"Skipping {policy}: {message}")
            rows.append(skipped_row(policy, message, args.quality_filter))
            continue

        policy_output_dir = output_dir / policy
        print(f"Running TVS for {policy} -> {policy_output_dir}")
        metrics = run_tvs_evaluation(
            policy_name=policy,
            output_dir=str(policy_output_dir),
            episodes=int(args.episodes),
            seed=int(args.seed),
            quality_filter=args.quality_filter,
            top_fraction=float(args.top_fraction),
            band_ratio=float(args.band_ratio),
            sigma=shared_sigma,
            max_points_per_trajectory=int(args.max_points_per_trajectory),
            no_plots=bool(args.no_plots),
            save_trajectories=bool(args.save_trajectories),
            fallback_on_zero_q=bool(args.fallback_on_zero_q),
        )
        rows.append(compact_row(metrics))
        full_metrics[policy] = metrics

    save_csv(rows, output_dir / "tvs_summary.csv")
    with (output_dir / "tvs_summary.json").open("w", encoding="utf-8") as handle:
        json.dump({"sigma_calibration": sigma_info, "summary": rows, "full_metrics": full_metrics}, handle, indent=2)
    if sigma_info is not None:
        with (output_dir / "sigma_calibration.json").open("w", encoding="utf-8") as handle:
            json.dump(sigma_info, handle, indent=2)

    if not args.no_plots:
        save_summary_plot(rows, output_dir / "tvs_summary.png")
        save_quality_diversity_scatter(rows, output_dir / "quality_diversity_scatter.png")
        save_reward_diversity_scatter(rows, output_dir / "return_diversity_scatter.png")
        save_baseline_metrics_plot(rows, output_dir / "coverage_entropy_tvs_comparison.png")

    print(f"Summary written to: {output_dir / 'tvs_summary.csv'}")
    if sigma_info is not None:
        print(f"Shared sigma: {sigma_info['sigma']:.6f} ({sigma_info['mode']})")


if __name__ == "__main__":
    main()
