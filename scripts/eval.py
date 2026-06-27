"""Evaluation entrypoint.

Supports newer algorithm subcommands and a backward-compatible policy evaluation
mode used by the original tests:
    python scripts/eval.py --policy baseline --episodes 200
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pacman_rldp.agents import BaselineNearestFoodAvoidGhostPolicy, RandomPolicy
from pacman_rldp.agents.policies import Policy
from pacman_rldp.algorithms.policy_iteration.pi_runner import eval_pi
from pacman_rldp.env import PacmanEnv, build_env_config
from pacman_rldp.pipelines_food_bitmask_vi import eval_food_bitmask_value_iteration
from pacman_rldp.pipelines_tabular_q import eval_tabular_q
from pacman_rldp.utils import ensure_directory, load_yaml, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluation script.")
    parser.add_argument(
        "algo",
        nargs="?",
        choices=["pi", "vi", "q_learning", "sarsa"],
        default=None,
        help="Algorithm name. If omitted, evaluates a policy from --policy/config.",
    )
    parser.add_argument("--config", default=None, help="Path to config file")
    parser.add_argument("--model", default=None, help="Path to model artifact")
    parser.add_argument("--output-dir", default=None, help="Override output directory")
    parser.add_argument("--episodes", type=int, default=None, help="Number of episodes")
    parser.add_argument("--seed", type=int, default=None, help="Base seed")
    parser.add_argument("--policy", choices=["random", "baseline"], default=None, help="Legacy policy eval")
    parser.add_argument("--render-mode", default=None, help="Render mode for VI eval")
    parser.add_argument("--gif-title", default=None, help="GIF title for VI eval")
    parser.add_argument("--no-gif", action="store_true", help="Disable GIF for VI eval")
    parser.add_argument("--fallback-on-zero-q", action="store_true", help="Q/SARSA: use heuristic fallback when all legal Q-values are zero")
    return parser.parse_args()


def resolve_eval_schedule(
    *,
    policy_name: str,
    args_episodes: int | None,
    args_seed: int | None,
    eval_cfg: dict[str, Any],
    env_seed: int,
) -> tuple[int, int]:
    """Resolve deterministic evaluation schedule.

    Baseline numbers in the README use 200 episodes with base seed 42. Random
    policy follows the config by default.
    """
    if args_episodes is not None:
        episodes = int(args_episodes)
    elif policy_name == "baseline":
        episodes = int(eval_cfg.get("baseline_episodes", 200))
    else:
        episodes = int(eval_cfg.get("episodes", 10))

    if args_seed is not None:
        base_seed = int(args_seed)
    elif policy_name == "baseline":
        base_seed = int(eval_cfg.get("baseline_seed", 42))
    else:
        base_seed = int(eval_cfg.get("seed_base", env_seed))
    return episodes, base_seed


def build_episode_seeds(*, base_seed: int, episodes: int) -> list[int]:
    """Build the reproducible seed schedule base_seed + i."""
    return [int(base_seed + idx) for idx in range(int(episodes))]


def _build_legacy_policy(policy_name: str, *, seed: int) -> Policy:
    if policy_name == "baseline":
        return BaselineNearestFoodAvoidGhostPolicy()
    if policy_name == "random":
        return RandomPolicy(seed=seed)
    raise ValueError(f"Unsupported policy: {policy_name}")


def eval_policy(
    *,
    config_path: str = "configs/default.yaml",
    policy_name: str | None = None,
    output_dir: str | None = None,
    episodes: int | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    cfg = load_yaml(config_path)
    env_cfg_dict = cfg.get("env", {})
    eval_cfg = cfg.get("eval", {})
    paths_cfg = cfg.get("paths", {})
    if seed is not None:
        env_cfg_dict = {**env_cfg_dict, "seed": seed}
    env_cfg = build_env_config(env_cfg_dict)
    selected_policy = policy_name or str(eval_cfg.get("policy", "random"))
    episode_count, base_seed = resolve_eval_schedule(
        policy_name=selected_policy,
        args_episodes=episodes,
        args_seed=seed,
        eval_cfg=eval_cfg,
        env_seed=env_cfg.seed,
    )

    env = PacmanEnv(config=env_cfg, render_mode=None)
    policy = _build_legacy_policy(selected_policy, seed=int(eval_cfg.get("policy_seed", base_seed)))

    returns: list[float] = []
    scores: list[float] = []
    lengths: list[int] = []
    wins: list[bool] = []
    losses: list[bool] = []
    seeds = build_episode_seeds(base_seed=base_seed, episodes=episode_count)
    for episode_seed in seeds:
        observation, info = env.reset(seed=episode_seed)
        total_return = 0.0
        length = 0
        while True:
            action = policy.select_action(observation, info)
            observation, reward, terminated, truncated, info = env.step(action)
            total_return += float(reward)
            length += 1
            if terminated or truncated:
                returns.append(total_return)
                scores.append(float(info.get("score", 0.0)))
                lengths.append(length)
                wins.append(bool(info.get("is_win", False)))
                losses.append(bool(info.get("is_lose", False)))
                break
    env.close()

    metrics = {
        "policy": selected_policy,
        "episodes": episode_count,
        "base_seed": base_seed,
        "seeds": seeds,
        "win_rate": sum(1 for win in wins if win) / max(1, len(wins)),
        "loss_rate": sum(1 for loss in losses if loss) / max(1, len(losses)),
        "avg_reward": sum(returns) / max(1, len(returns)),
        "avg_score": sum(scores) / max(1, len(scores)),
        "avg_episode_length": sum(lengths) / max(1, len(lengths)),
        "returns": returns,
        "scores": scores,
        "episode_lengths": lengths,
    }
    output_dir_path = ensure_directory(output_dir or paths_cfg.get("eval_output_dir", "results/eval"))
    save_json(metrics, output_dir_path / "eval_metrics.json")
    return metrics


def main() -> None:
    args = parse_args()

    if args.algo is None:
        metrics = eval_policy(
            config_path=args.config or "configs/default.yaml",
            policy_name=args.policy,
            output_dir=args.output_dir,
            episodes=args.episodes,
            seed=args.seed,
        )
        print(
            "Policy evaluation complete. "
            f"policy={metrics['policy']}, episodes={metrics['episodes']}, "
            f"avg_reward={metrics['avg_reward']:.3f}, win_rate={metrics['win_rate']:.3f}"
        )
        return

    if args.algo == "pi":
        config = args.config or "configs/policy_iteration_obs.yaml"
        metrics = eval_pi(
            config_path=config,
            model_path=args.model,
            output_dir=args.output_dir,
            episodes=args.episodes,
            seed=args.seed,
        )
        print(
            "PI evaluation complete. "
            f"episodes={metrics['episodes']}, mean_return={metrics['mean_return']:.3f}, "
            f"win_rate={metrics['win_rate']:.3f}"
        )
        return

    if args.algo == "vi":
        config = args.config or "configs/bitmask_value_iteration.yaml"
        result = eval_food_bitmask_value_iteration(
            config_path=config,
            model_path=args.model,
            output_dir=args.output_dir,
            episodes=args.episodes,
            seed=args.seed,
            render_mode=args.render_mode,
            gif_title=args.gif_title,
            no_gif=args.no_gif,
        )
        metrics = result["metrics"]
        print(
            "VI evaluation complete. "
            f"episodes={metrics['episodes']}, mean_return={metrics['mean_return']:.3f}, "
            f"win_rate={metrics['win_rate']:.3f}"
        )
        return

    if args.algo in {"q_learning", "sarsa"}:
        config = args.config or ("configs/q_learning.yaml" if args.algo == "q_learning" else "configs/sarsa.yaml")
        default_model = (
            "results/train_q_learning/q_table.pkl" if args.algo == "q_learning" else "results/train_sarsa/q_table.pkl"
        )
        metrics = eval_tabular_q(
            config_path=config,
            model_path=args.model or default_model,
            output_dir=args.output_dir or ("results/eval_q_learning" if args.algo == "q_learning" else "results/eval_sarsa"),
            episodes=args.episodes or 200,
            seed=args.seed,
            algo_name=args.algo,
            fallback_on_zero_q=bool(args.fallback_on_zero_q),
        )
        print(
            f"{args.algo} evaluation complete. "
            f"episodes={metrics['episodes']}, mean_return={metrics['mean_return']:.3f}, "
            f"win_rate={metrics['win_rate']:.3f}"
        )
        return


if __name__ == "__main__":
    main()
