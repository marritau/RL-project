"""Evaluate Pacman trajectory diversity with Temporal Vendi Score (TVS).

Examples:
    python scripts/evaluate_tvs.py --policy baseline --episodes 128 --output-dir results/tvs_baseline
    python scripts/evaluate_tvs.py --policy vi --config configs/bitmask_value_iteration.yaml --episodes 256 --output-dir results/tvs_vi
    python scripts/evaluate_tvs.py --policy q_learning --config configs/q_learning.yaml --episodes 256 --output-dir results/tvs_q_learning

The implementation is a Pacman-grid adaptation of:
"Beyond Reward Maximization: Evaluating the Diversity of Trajectories in
Reinforcement Learning with Temporal Vendi Score".
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import matplotlib.pyplot as plt
import numpy as np

from pacman_rldp.agents import BaselineNearestFoodAvoidGhostPolicy, RandomPolicy
from pacman_rldp.agents.obs_policy import ObsPolicy
from pacman_rldp.agents.policies import Policy
from pacman_rldp.algorithms.food_bitmask_value_iteration import FoodBitmaskValueIterationPolicy
from pacman_rldp.diversity import (
    FoodBitmaskHeuristicPolicy,
    GridTimeToReach,
    NoisyPolicy,
    TabularQGreedyPolicy,
    action_entropy,
    collect_trajectories,
    compute_temporal_vendi_score,
    filter_trajectories,
    occupancy_entropy,
    prefix_temporal_vendi_scores,
    state_coverage,
)
from pacman_rldp.env import PacmanEnv, build_env_config
from pacman_rldp.utils import ensure_directory, load_pickle, load_yaml, save_json

PAPER_URL = "https://openreview.net/forum?id=7qGCADaXjr"
PAPER_PDF_URL = "https://openreview.net/pdf/4b56e283f49a8cc5a6271a62bc723a09aefd088d.pdf"

POLICY_DEFAULTS: dict[str, dict[str, str]] = {
    "baseline": {"config": "configs/tvs_eval.yaml", "model": ""},
    "random": {"config": "configs/tvs_eval.yaml", "model": ""},
    "q_learning": {"config": "configs/q_learning.yaml", "model": "results/train_q_learning/q_table.pkl"},
    "sarsa": {"config": "configs/sarsa.yaml", "model": "results/train_sarsa/q_table.pkl"},
    "vi": {"config": "configs/bitmask_value_iteration.yaml", "model": "results/train_food_bitmask_vi/model.pkl"},
    "pi": {"config": "configs/policy_iteration_obs.yaml", "model": "results/obs_policy_iteration/policy.pkl"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Pacman policy diversity with TVS.")
    parser.add_argument(
        "--config",
        default=None,
        help="Experiment config YAML. Defaults depend on --policy.",
    )
    parser.add_argument(
        "--policy",
        choices=["baseline", "random", "q_learning", "sarsa", "vi", "pi"],
        default="baseline",
        help="Policy to rollout for diversity evaluation.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model artifact for q_learning/sarsa/vi/pi. Defaults depend on --policy.",
    )
    parser.add_argument(
        "--epsilon-random",
        type=float,
        default=0.0,
        help="Wrap the selected policy with epsilon-random actions for controlled diversity diagnostics.",
    )
    parser.add_argument(
        "--fallback-on-zero-q",
        action="store_true",
        help="For Q/SARSA TVS only: use the heuristic fallback when all legal Q-values are exactly zero.",
    )
    parser.add_argument("--episodes", type=int, default=128, help="Number of rollouts to collect.")
    parser.add_argument("--seed", type=int, default=None, help="Base seed; defaults to env.seed.")
    parser.add_argument("--output-dir", default=None, help="Directory for TVS artifacts.")
    parser.add_argument(
        "--quality-filter",
        default="wins_or_all",
        choices=["all", "wins", "wins_or_all", "non_loss", "top_return"],
        help="Which trajectories are scored. TVS is most meaningful on high-quality trajectories.",
    )
    parser.add_argument(
        "--top-fraction",
        type=float,
        default=0.5,
        help="Fraction used when --quality-filter=top_return.",
    )
    parser.add_argument(
        "--band-ratio",
        type=float,
        default=0.2,
        help="Sakoe-Chiba band ratio for GAK. The paper uses 0.2.",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=None,
        help="Optional local-kernel bandwidth. If omitted, calibrated from rollout state distances.",
    )
    parser.add_argument(
        "--max-points-per-trajectory",
        type=int,
        default=160,
        help="Uniformly resample longer trajectories to keep GAK tractable. Set <=0 to disable.",
    )
    parser.add_argument(
        "--compress-repeats",
        action="store_true",
        help=(
            "Compress consecutive duplicate Pacman positions before TVS. "
            "Disabled by default to preserve the temporal sequence as in the paper."
        ),
    )
    parser.add_argument("--no-plots", action="store_true", help="Skip PNG plot generation.")
    parser.add_argument(
        "--save-trajectories",
        action="store_true",
        help="Save collected trajectories as JSON for inspection/reproducibility.",
    )
    return parser.parse_args()


def resolve_config_path(policy_name: str, config_path: str | None) -> str:
    if config_path:
        return config_path
    return POLICY_DEFAULTS[policy_name]["config"]


def resolve_model_path(policy_name: str, model_path: str | None) -> str | None:
    if model_path:
        return model_path
    default_model = POLICY_DEFAULTS[policy_name].get("model") or ""
    return default_model or None


def force_raw_observation(env_cfg_dict: dict[str, Any]) -> dict[str, Any]:
    """Baseline/random need walls and food, so they should use raw observations."""
    copied = dict(env_cfg_dict)
    observation = dict(copied.get("observation", {}) or {})
    observation["name"] = "raw"
    copied["observation"] = observation
    return copied


def walls_from_env_or_observation(env: PacmanEnv, observation: dict[str, Any]) -> np.ndarray:
    if "walls" in observation:
        return np.asarray(observation["walls"], dtype=np.int8)
    # Non-raw observations still come from the same runtime state.  TVS needs the
    # static layout walls for shortest-path time-to-reach distances.
    return np.asarray(env.runtime_state.getWalls().data, dtype=np.int8)


def non_wall_coords_from_env(env: PacmanEnv) -> list[tuple[int, int]]:
    context = getattr(env, "_observation_context", None)
    coords = getattr(context, "non_wall_coords", None)
    if coords is None:
        walls = np.asarray(env.runtime_state.getWalls().data, dtype=np.int8)
        return [
            (int(x_coord), int(y_coord))
            for x_coord in range(walls.shape[0])
            for y_coord in range(walls.shape[1])
            if int(walls[x_coord, y_coord]) == 0
        ]
    return [tuple(map(int, coord)) for coord in coords]


def build_policy(
    policy_name: str,
    *,
    cfg: dict[str, Any],
    model_path: str | None,
    env: PacmanEnv,
    seed: int,
    epsilon_random: float,
    fallback_on_zero_q: bool = False,
) -> Policy:
    if policy_name == "baseline":
        policy: Policy = BaselineNearestFoodAvoidGhostPolicy()
    elif policy_name == "random":
        policy = RandomPolicy(seed=seed)
    elif policy_name in {"q_learning", "sarsa"}:
        if model_path is None:
            raise ValueError(f"--model is required for policy={policy_name}")
        encoding_cfg = dict(cfg.get("obs_encoding", {}))
        drop_keys = list(encoding_cfg.get("drop_keys", ["score", "step_count"]))
        float_round = int(encoding_cfg.get("float_round", 3))
        fallback_policy = FoodBitmaskHeuristicPolicy(
            non_wall_coords=non_wall_coords_from_env(env),
            seed=seed + 101,
            danger_distance=2,
        )
        policy = TabularQGreedyPolicy.from_model_path(
            model_path,
            action_size=env.action_space.n,
            drop_keys=drop_keys,
            float_round=float_round,
            seed=seed,
            fallback_policy=fallback_policy,
            fallback_on_zero=fallback_on_zero_q,
        )
    elif policy_name == "vi":
        if model_path is None:
            raise ValueError("--model is required for policy=vi")
        model_data = load_pickle(Path(model_path))
        if not isinstance(model_data, dict):
            raise ValueError("VI model artifact must be a dictionary.")
        policy = FoodBitmaskValueIterationPolicy(model_data=model_data, fallback_seed=seed)
    elif policy_name == "pi":
        if model_path is None:
            raise ValueError("--model is required for policy=pi")
        encoding_cfg = dict(cfg.get("obs_encoding", {}))
        drop_keys = list(encoding_cfg.get("drop_keys", ["score", "step_count"]))
        float_round = int(encoding_cfg.get("float_round", 3))
        policy_table = load_pickle(Path(model_path))
        if not isinstance(policy_table, dict):
            raise ValueError("PI model artifact must be a dictionary policy table.")
        # ObsPolicy has its own heuristic fallback for unseen food-bitmask states.
        policy = ObsPolicy(
            policy_table,
            seed=seed,
            drop_keys=drop_keys,
            float_round=float_round,
            non_wall_coords=non_wall_coords_from_env(env),
            fallback_danger_distance=2,
        )
    else:
        raise ValueError(f"Unsupported policy: {policy_name}")

    if epsilon_random > 0.0:
        policy = NoisyPolicy(base_policy=policy, epsilon=epsilon_random, seed=seed + 17)
    return policy


def summarize_rollouts(trajectories) -> dict[str, Any]:
    returns = [trajectory.total_return for trajectory in trajectories]
    lengths = [trajectory.length for trajectory in trajectories]
    scores = [trajectory.score for trajectory in trajectories]
    return {
        "episodes": len(trajectories),
        "win_rate": float(np.mean([trajectory.win for trajectory in trajectories])) if trajectories else 0.0,
        "lose_rate": float(np.mean([trajectory.lose for trajectory in trajectories])) if trajectories else 0.0,
        "truncation_rate": float(np.mean([trajectory.truncated for trajectory in trajectories])) if trajectories else 0.0,
        "mean_return": float(np.mean(returns)) if returns else 0.0,
        "std_return": float(np.std(returns)) if returns else 0.0,
        "mean_score": float(np.mean(scores)) if scores else 0.0,
        "mean_length": float(np.mean(lengths)) if lengths else 0.0,
        "median_length": float(np.median(lengths)) if lengths else 0.0,
        "state_coverage": int(state_coverage(trajectories)),
        "occupancy_entropy": float(occupancy_entropy(trajectories)),
        "action_entropy": float(action_entropy(trajectories)),
    }


def save_occupancy_heatmap(trajectories, walls: np.ndarray, output_path: Path) -> None:
    width, height = int(walls.shape[0]), int(walls.shape[1])
    heatmap = np.zeros((height, width), dtype=np.float64)
    for trajectory in trajectories:
        for x_coord, y_coord in trajectory.states:
            if 0 <= x_coord < width and 0 <= y_coord < height:
                heatmap[height - 1 - y_coord, x_coord] += 1.0
    wall_mask = np.asarray(walls).T[::-1]
    heatmap = np.ma.array(heatmap, mask=wall_mask.astype(bool))
    plt.figure(figsize=(8, 3))
    plt.imshow(heatmap)
    plt.title("Pacman occupancy heatmap for TVS rollouts")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.colorbar(label="visits")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def save_trajectory_overlay(
    trajectories,
    walls: np.ndarray,
    output_path: Path,
    *,
    max_trajectories: int = 40,
) -> None:
    """Plot sampled trajectories over the maze, similar in spirit to paper occupancy figures.

    This is not part of the TVS formula. It is a diagnostic visualization that
    helps visually check whether a high/low TVS value corresponds to visibly
    different routes through the maze.
    """
    width, height = int(walls.shape[0]), int(walls.shape[1])
    fig, ax = plt.subplots(figsize=(8, 3))
    wall_img = np.asarray(walls).T[::-1]
    ax.imshow(wall_img, cmap="Greys", alpha=0.35, extent=(-0.5, width - 0.5, -0.5, height - 0.5))

    selected = list(trajectories)[: max(0, int(max_trajectories))]
    for trajectory in selected:
        if not trajectory.states:
            continue
        xs = [float(x) for x, _ in trajectory.states]
        ys = [float(y) for _, y in trajectory.states]
        ax.plot(xs, ys, linewidth=1.0, alpha=0.35)
        ax.scatter(xs[0], ys[0], s=18, marker="o", alpha=0.55)
        ax.scatter(xs[-1], ys[-1], s=18, marker="x", alpha=0.55)

    ax.set_xlim(-0.5, width - 0.5)
    ax.set_ylim(-0.5, height - 0.5)
    ax.set_aspect("equal")
    ax.set_title(f"Scored trajectory overlay (first {len(selected)} rollouts)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, alpha=0.15)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_similarity_plot(similarity: np.ndarray, output_path: Path) -> None:
    plt.figure(figsize=(5, 4))
    plt.imshow(similarity, vmin=0.0, vmax=1.0)
    plt.title("Normalized GAK trajectory similarity")
    plt.xlabel("trajectory")
    plt.ylabel("trajectory")
    plt.colorbar(label="similarity")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def save_convergence_plot(prefix_scores: dict[int, float], output_path: Path) -> None:
    xs = sorted(prefix_scores)
    ys = [prefix_scores[x] for x in xs]
    plt.figure(figsize=(6, 4))
    plt.plot(xs, ys, marker="o")
    plt.title("TVS convergence over rollout count")
    plt.xlabel("sampled trajectories")
    plt.ylabel("TVS")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def run_tvs_evaluation(
    *,
    policy_name: str,
    config_path: str | None = None,
    model_path: str | None = None,
    output_dir: str | None = None,
    episodes: int = 128,
    seed: int | None = None,
    epsilon_random: float = 0.0,
    fallback_on_zero_q: bool = False,
    quality_filter: str = "wins_or_all",
    top_fraction: float = 0.5,
    band_ratio: float = 0.2,
    sigma: float | None = None,
    max_points_per_trajectory: int | None = 160,
    compress_repeats: bool = False,
    no_plots: bool = False,
    save_trajectories: bool = False,
) -> dict[str, Any]:
    resolved_config = resolve_config_path(policy_name, config_path)
    resolved_model = resolve_model_path(policy_name, model_path)
    resolved_output = output_dir or f"results/tvs_{policy_name}"
    output_dir_path = ensure_directory(resolved_output)
    started_at = time.perf_counter()

    cfg = load_yaml(resolved_config)
    env_cfg_dict = dict(cfg.get("env", {}))
    if policy_name in {"baseline", "random"}:
        env_cfg_dict = force_raw_observation(env_cfg_dict)
    if seed is not None:
        env_cfg_dict = {**env_cfg_dict, "seed": int(seed)}
    env_cfg = build_env_config(env_cfg_dict)
    base_seed = int(seed if seed is not None else env_cfg.seed)

    env = PacmanEnv(config=env_cfg, render_mode=None)
    initial_observation, _ = env.reset(seed=base_seed)
    walls = walls_from_env_or_observation(env, initial_observation)
    time_to_reach = GridTimeToReach.from_walls(walls)

    policy = build_policy(
        policy_name,
        cfg=cfg,
        model_path=resolved_model,
        env=env,
        seed=base_seed,
        epsilon_random=float(epsilon_random),
        fallback_on_zero_q=bool(fallback_on_zero_q),
    )
    trajectories = collect_trajectories(
        env=env,
        policy=policy,
        episodes=int(episodes),
        base_seed=base_seed,
    )
    env.close()

    selected = filter_trajectories(
        trajectories,
        mode=quality_filter,
        top_fraction=float(top_fraction),
    )
    if not selected:
        raise RuntimeError(
            "Quality filter removed all trajectories. Try --quality-filter all or collect more episodes."
        )

    max_points = max_points_per_trajectory
    if max_points is not None and max_points <= 0:
        max_points = None

    tvs_compute_started_at = time.perf_counter()
    tvs_result = compute_temporal_vendi_score(
        selected,
        time_to_reach,
        sigma=sigma,
        band_ratio=float(band_ratio),
        max_points_per_trajectory=max_points,
        compress_repeats=bool(compress_repeats),
    )
    tvs_compute_seconds = float(time.perf_counter() - tvs_compute_started_at)
    prefix_scores = prefix_temporal_vendi_scores(tvs_result.similarity_matrix)
    total_seconds = float(time.perf_counter() - started_at)

    metrics = {
        "paper": {
            "title": "Beyond Reward Maximization: Evaluating the Diversity of Trajectories in Reinforcement Learning with Temporal Vendi Score",
            "openreview_url": PAPER_URL,
            "pdf_url": PAPER_PDF_URL,
        },
        "policy": policy_name,
        "epsilon_random": float(epsilon_random),
        "config_path": resolved_config,
        "model_path": resolved_model,
        "base_seed": base_seed,
        "quality_filter": quality_filter,
        "fallback_on_zero_q": bool(fallback_on_zero_q),
        "runtime_seconds": total_seconds,
        "tvs_compute_seconds": tvs_compute_seconds,
        "max_points_per_trajectory": None if max_points is None else int(max_points),
        "compress_repeats": bool(compress_repeats),
        "collected_rollouts": summarize_rollouts(trajectories),
        "scored_rollouts": summarize_rollouts(selected),
        "tvs": tvs_result.metrics_dict(),
        "tvs_convergence": {str(k): float(v) for k, v in prefix_scores.items()},
        "notes": [
            "Pacman adaptation uses Pacman's grid position as the state projection.",
            "Time-to-reach distance is exact shortest-path length through layout walls.",
            "Trajectory similarity uses paper-style banded Global Alignment Kernel with normalized K_ii=1.",
            "Final diversity is q=2 Vendi score, matching the paper's main experiments.",
            "Sigma is stored in metrics; for fair cross-policy comparisons prefer evaluate_all_tvs.py with shared sigma.",
            "For trained tabular policies, TVS uses the policy's native observation representation for actions and Pacman's grid position for trajectory scoring.",
        ],
    }
    save_json(metrics, output_dir_path / "tvs_metrics.json")
    save_json({str(k): float(v) for k, v in prefix_scores.items()}, output_dir_path / "tvs_convergence.json")
    np.save(output_dir_path / "similarity_matrix.npy", tvs_result.similarity_matrix)

    if save_trajectories:
        with (output_dir_path / "trajectories.json").open("w", encoding="utf-8") as handle:
            json.dump([trajectory.to_json_dict() for trajectory in trajectories], handle, indent=2)
        with (output_dir_path / "scored_trajectories.json").open("w", encoding="utf-8") as handle:
            json.dump([trajectory.to_json_dict() for trajectory in selected], handle, indent=2)

    if not no_plots:
        save_occupancy_heatmap(selected, walls, output_dir_path / "occupancy_heatmap.png")
        save_trajectory_overlay(selected, walls, output_dir_path / "trajectory_overlay.png")
        save_similarity_plot(tvs_result.similarity_matrix, output_dir_path / "similarity_matrix.png")
        save_convergence_plot(prefix_scores, output_dir_path / "tvs_convergence.png")

    return metrics


def main() -> None:
    args = parse_args()
    metrics = run_tvs_evaluation(
        policy_name=args.policy,
        config_path=args.config,
        model_path=args.model,
        output_dir=args.output_dir,
        episodes=int(args.episodes),
        seed=args.seed,
        epsilon_random=float(args.epsilon_random),
        fallback_on_zero_q=bool(args.fallback_on_zero_q),
        quality_filter=args.quality_filter,
        top_fraction=float(args.top_fraction),
        band_ratio=float(args.band_ratio),
        sigma=args.sigma,
        max_points_per_trajectory=args.max_points_per_trajectory,
        compress_repeats=bool(args.compress_repeats),
        no_plots=bool(args.no_plots),
        save_trajectories=bool(args.save_trajectories),
    )
    print(
        "TVS evaluation complete. "
        f"policy={metrics['policy']}, "
        f"scored={metrics['scored_rollouts']['episodes']}/{metrics['collected_rollouts']['episodes']}, "
        f"tvs={metrics['tvs']['tvs']:.3f}, "
        f"win_rate={metrics['collected_rollouts']['win_rate']:.3f}"
    )
    print(f"Metrics written to: {Path(args.output_dir or f'results/tvs_{args.policy}') / 'tvs_metrics.json'}")


if __name__ == "__main__":
    main()
