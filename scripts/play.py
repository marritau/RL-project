"""Play/visualization entrypoint for algorithms and policies."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pacman_rldp.agents import BaselineNearestFoodAvoidGhostPolicy, KeyboardPolicy, RandomPolicy
from pacman_rldp.agents.policies import Policy
from pacman_rldp.algorithms.policy_iteration.pi_runner import run_pi
from pacman_rldp.env import PacmanEnv, build_env_config
from pacman_rldp.pipelines_food_bitmask_vi import eval_food_bitmask_value_iteration
from pacman_rldp.pipelines_tabular_q import run_tabular_q
from pacman_rldp.utils import load_yaml
from pacman_rldp.visuals import run_keyboard_game


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Play script.")
    parser.add_argument(
        "algo",
        nargs="?",
        choices=["pi", "vi", "q_learning", "sarsa", "manual"],
        default=None,
        help="Algorithm name or manual. If omitted, runs --policy random/baseline.",
    )
    parser.add_argument("--config", default=None, help="Path to config file")
    parser.add_argument("--model", default=None, help="Path to model artifact")
    parser.add_argument("--policy", choices=["random", "baseline"], default="random", help="Policy mode")
    parser.add_argument("--render-mode", default="human", help="Render mode")
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes")
    parser.add_argument("--seed", type=int, default=None, help="Base seed")
    parser.add_argument("--gif-title", default=None, help="GIF title")
    parser.add_argument("--no-gif", action="store_true", help="Disable GIF")
    return parser.parse_args()


def run_manual(config_path: str, render_mode: str, episodes: int, seed: int | None) -> None:
    cfg = load_yaml(config_path)
    env_cfg_dict = cfg.get("env", {})
    if seed is not None:
        env_cfg_dict = {**env_cfg_dict, "seed": seed}
    env_cfg = build_env_config(env_cfg_dict)

    if render_mode == "human":
        final_score = run_keyboard_game(
            layout_name=env_cfg.layout_name,
            num_ghosts=env_cfg.num_ghosts,
            seed=env_cfg.seed,
            render_mode="human",
            zoom=env_cfg.zoom,
            frame_time=env_cfg.frame_time,
            ghost_policy=env_cfg.ghost_policy,
            ghost_loop_matrix=env_cfg.ghost_loop_matrix,
            ghost_loop_direction=env_cfg.ghost_loop_direction,
        )
        print(f"Manual game finished. Final score: {final_score:.2f}")
        return

    env = PacmanEnv(config=env_cfg, render_mode="ansi")
    policy = KeyboardPolicy()
    for episode_idx in range(episodes):
        observation, info = env.reset(seed=env_cfg.seed + episode_idx)
        print(f"Episode {episode_idx + 1} started.")
        while True:
            action = policy.select_action(observation, info)
            observation, reward, terminated, truncated, info = env.step(action)
            rendered = env.render()
            if rendered is not None:
                print(rendered)
            print(f"Reward: {reward:.2f} | Score: {info.get('score', 0.0):.2f}")
            if terminated or truncated:
                print(
                    f"Episode {episode_idx + 1} finished. "
                    f"win={info.get('is_win', False)} score={info.get('score', 0.0):.2f}"
                )
                break
    env.close()


def _build_policy(policy_name: str, *, seed: int) -> Policy:
    if policy_name == "baseline":
        return BaselineNearestFoodAvoidGhostPolicy()
    if policy_name == "random":
        return RandomPolicy(seed=seed)
    raise ValueError(f"Unsupported policy: {policy_name}")


def run_policy_play(
    *,
    config_path: str,
    policy_name: str,
    render_mode: str,
    episodes: int,
    seed: int | None,
) -> None:
    cfg = load_yaml(config_path)
    env_cfg_dict = cfg.get("env", {})
    if seed is not None:
        env_cfg_dict = {**env_cfg_dict, "seed": seed}
    env_cfg = build_env_config(env_cfg_dict)
    env = PacmanEnv(config=env_cfg, render_mode=render_mode)
    policy = _build_policy(policy_name, seed=env_cfg.seed)
    for episode_idx in range(episodes):
        observation, info = env.reset(seed=env_cfg.seed + episode_idx)
        print(f"Episode {episode_idx + 1} started. policy={policy_name}")
        while True:
            action = policy.select_action(observation, info)
            observation, reward, terminated, truncated, info = env.step(action)
            rendered = env.render()
            if rendered is not None:
                print(rendered)
            if render_mode == "ansi":
                print(f"Reward: {reward:.2f} | Score: {info.get('score', 0.0):.2f}")
            if terminated or truncated:
                print(
                    f"Episode {episode_idx + 1} finished. "
                    f"win={info.get('is_win', False)} score={info.get('score', 0.0):.2f}"
                )
                break
    env.close()


def main() -> None:
    args = parse_args()

    if args.algo is None:
        if (not args.no_gif) or args.gif_title is not None:
            print("GIF export is only implemented for algorithm play modes. Ignoring GIF options.")
        run_policy_play(
            config_path=args.config or "configs/default.yaml",
            policy_name=args.policy,
            render_mode=args.render_mode,
            episodes=args.episodes,
            seed=args.seed,
        )
        return

    if args.algo == "manual":
        if (not args.no_gif) or args.gif_title is not None:
            print("GIF export is not supported in manual mode. Ignoring GIF options.")
        config = args.config or "configs/default.yaml"
        run_manual(config, args.render_mode, args.episodes, args.seed)
        return

    if args.algo == "pi":
        config = args.config or "configs/policy_iteration_obs.yaml"
        run_pi(
            config_path=config,
            model_path=args.model,
            render_mode=args.render_mode,
            episodes=args.episodes,
            seed=args.seed,
            gif_title=args.gif_title,
            no_gif=args.no_gif,
        )
        return

    if args.algo == "vi":
        config = args.config or "configs/bitmask_value_iteration.yaml"
        eval_food_bitmask_value_iteration(
            config_path=config,
            model_path=args.model,
            output_dir=None,
            episodes=args.episodes,
            seed=args.seed,
            render_mode=args.render_mode,
            gif_title=args.gif_title,
            no_gif=args.no_gif,
        )
        return

    if args.algo in {"q_learning", "sarsa"}:
        config = args.config or ("configs/default.yaml" if args.algo == "q_learning" else "configs/sarsa.yaml")
        default_model = (
            "results/train_q_learning/q_table.pkl" if args.algo == "q_learning" else "results/train_sarsa/q_table.pkl"
        )
        run_tabular_q(
            config_path=config,
            model_path=args.model or default_model,
            render_mode=args.render_mode,
            episodes=args.episodes,
            seed=args.seed,
            gif_title=args.gif_title,
            no_gif=args.no_gif,
        )
        return


if __name__ == "__main__":
    main()
