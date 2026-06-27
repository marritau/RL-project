"""Training entrypoint.

Supports the newer subcommands (pi/vi/q_learning/sarsa) and a small legacy smoke
path used by the original tests when no algorithm is provided.
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

import matplotlib.pyplot as plt

from pacman_rldp.agents import RandomPolicy
from pacman_rldp.algorithms.policy_iteration.pi_runner import train_pi
from pacman_rldp.env import PacmanEnv, build_env_config
from pacman_rldp.pipelines_food_bitmask_vi import train_food_bitmask_value_iteration
from pacman_rldp.pipelines_tabular_q import train_q_learning, train_sarsa
from pacman_rldp.utils import ensure_directory, load_yaml, save_json, save_pickle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Training script.")
    parser.add_argument(
        "algo",
        nargs="?",
        choices=["pi", "vi", "q_learning", "sarsa"],
        default=None,
        help="Algorithm name. If omitted, runs the legacy random-policy smoke trainer.",
    )
    parser.add_argument("--config", default=None, help="Path to config file")
    parser.add_argument("--output-dir", default=None, help="Override output directory")
    parser.add_argument("--episodes", type=int, default=None, help="Number of episodes")
    parser.add_argument("--seed", type=int, default=None, help="Base seed")
    parser.add_argument("--alpha", type=float, default=None, help="Learning rate (q/sarsa); defaults to config agent_params or 0.1")
    parser.add_argument("--gamma", type=float, default=None, help="Discount factor; defaults to config agent_params or 0.99")
    parser.add_argument("--epsilon", type=float, default=None, help="Epsilon-greedy (q/sarsa); defaults to config agent_params or 0.1")
    parser.add_argument("--epsilon-decay", type=float, default=None, help="SARSA epsilon decay; defaults to config or 1.0")
    parser.add_argument("--epsilon-end", type=float, default=None, help="SARSA minimum epsilon; defaults to config or 0.0")
    parser.add_argument("--collection-episodes", type=int, default=None, help="VI collection episodes")
    parser.add_argument("--max-iterations", type=int, default=None, help="VI max iterations")
    parser.add_argument("--tolerance", type=float, default=None, help="VI tolerance")
    return parser.parse_args()


def _plot_learning_curve(returns: list[float], output_path: Path) -> None:
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, len(returns) + 1), returns, marker="o")
    plt.xlabel("episode")
    plt.ylabel("return")
    plt.title("Legacy smoke training returns")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def train_legacy_smoke(
    *,
    config_path: str = "configs/default.yaml",
    output_dir: str | None = None,
    episodes: int | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """Run a tiny non-learning random-policy collection and write legacy artifacts.

    The project now has explicit algorithm subcommands, but the original tests and
    older README examples expected ``scripts/train.py`` to work without a
    positional algorithm and to emit model.pkl/train_metrics.json/learning_curve.png.
    Keeping this path makes the CLI backward-compatible.
    """
    cfg = load_yaml(config_path)
    env_cfg_dict = cfg.get("env", {})
    train_cfg = cfg.get("train", {})
    paths_cfg = cfg.get("paths", {})
    if seed is not None:
        env_cfg_dict = {**env_cfg_dict, "seed": seed}
    env_cfg = build_env_config(env_cfg_dict)
    episode_count = int(episodes if episodes is not None else train_cfg.get("episodes", 20))
    policy_seed = int(train_cfg.get("policy_seed", env_cfg.seed))

    env = PacmanEnv(config=env_cfg, render_mode=None)
    policy = RandomPolicy(seed=policy_seed)
    returns: list[float] = []
    wins: list[bool] = []
    lengths: list[int] = []

    for episode_idx in range(episode_count):
        observation, info = env.reset(seed=env_cfg.seed + episode_idx)
        total_return = 0.0
        length = 0
        while True:
            action = policy.select_action(observation, info)
            observation, reward, terminated, truncated, info = env.step(action)
            total_return += float(reward)
            length += 1
            if terminated or truncated:
                returns.append(total_return)
                wins.append(bool(info.get("is_win", False)))
                lengths.append(length)
                break
    env.close()

    output_dir_path = ensure_directory(output_dir or paths_cfg.get("train_output_dir", "results/train"))
    model_payload = {"policy": "random", "policy_seed": policy_seed, "note": "Legacy smoke artifact."}
    save_pickle(model_payload, output_dir_path / "model.pkl")
    metrics = {
        "episodes": episode_count,
        "mean_return": sum(returns) / max(1, len(returns)),
        "win_rate": sum(1 for win in wins if win) / max(1, len(wins)),
        "avg_episode_length": sum(lengths) / max(1, len(lengths)),
        "returns": returns,
        "policy": "random",
        "base_seed": env_cfg.seed,
    }
    save_json(metrics, output_dir_path / "train_metrics.json")
    _plot_learning_curve(returns, output_dir_path / "learning_curve.png")
    return metrics


def main() -> None:
    args = parse_args()

    if args.algo is None:
        metrics = train_legacy_smoke(
            config_path=args.config or "configs/default.yaml",
            output_dir=args.output_dir,
            episodes=args.episodes,
            seed=args.seed,
        )
        print(
            "Legacy smoke training complete. "
            f"episodes={metrics['episodes']}, mean_return={metrics['mean_return']:.3f}"
        )
        return

    if args.algo == "pi":
        config = args.config or "configs/policy_iteration_obs.yaml"
        metrics = train_pi(
            config_path=config,
            output_dir=args.output_dir,
            episodes=args.episodes,
            seed=args.seed,
            log_every=100,
        )
        print(
            "PI training complete. "
            f"Episodes={metrics['episodes']}, states={metrics['state_count']}, transitions={metrics['transition_count']}"
        )
        return

    if args.algo == "vi":
        config = args.config or "configs/bitmask_value_iteration.yaml"
        result = train_food_bitmask_value_iteration(
            config_path=config,
            output_dir=args.output_dir,
            collection_episodes=args.collection_episodes,
            gamma=args.gamma,
            max_iterations=args.max_iterations,
            tolerance=args.tolerance,
            seed=args.seed,
        )
        metrics = result["metrics"]
        print(
            "VI training complete. "
            f"states={metrics['discovered_states']}, iterations={metrics['iterations']}, "
            f"total_seconds={metrics['total_seconds']:.3f}"
        )
        return

    if args.algo == "q_learning":
        config = args.config or "configs/q_learning.yaml"
        metrics = train_q_learning(
            config_path=config,
            output_dir=args.output_dir,
            episodes=args.episodes,
            alpha=args.alpha,
            gamma=args.gamma,
            epsilon=args.epsilon,
            epsilon_decay=args.epsilon_decay,
            epsilon_end=args.epsilon_end,
            seed=args.seed,
        )
        print(
            "Q-learning training complete. "
            f"episodes={metrics['episodes']}, mean_return={metrics['mean_return']:.3f}"
        )
        return

    if args.algo == "sarsa":
        config = args.config or "configs/sarsa.yaml"
        metrics = train_sarsa(
            config_path=config,
            output_dir=args.output_dir,
            episodes=args.episodes,
            alpha=args.alpha,
            gamma=args.gamma,
            epsilon=args.epsilon,
            epsilon_decay=args.epsilon_decay,
            epsilon_end=args.epsilon_end,
            seed=args.seed,
        )
        print(
            "SARSA training complete. "
            f"episodes={metrics['episodes']}, mean_return={metrics['mean_return']:.3f}"
        )
        return


if __name__ == "__main__":
    main()
