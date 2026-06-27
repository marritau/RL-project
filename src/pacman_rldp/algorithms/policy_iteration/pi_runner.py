"""High-level helpers for training, evaluating, and running PI policies."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from .obs_encoding import encode_observation
from .obs_mdp import ObsMDPModel
from .policy_iteration_obs import PolicyIterationResult, policy_iteration
from ...agents import ObsPolicy
from ...env import PacmanEnv, build_env_config
from ...logging import configure_logging
from ...visuals.capture import capture_human_frame, save_gif
from ...utils import ensure_directory, load_pickle, load_yaml, save_json, save_pickle

_OBS_DIR_TO_ACTION = {1: 0, 2: 1, 3: 2, 4: 3}
_OBS_DIR_OPPOSITE = {1: 2, 2: 1, 3: 4, 4: 3}


def _direction_to_target(source: tuple[float, float], target: tuple[float, float]) -> int:
    """Encode coarse direction from source to target as integer id."""
    delta_x = float(target[0] - source[0])
    delta_y = float(target[1] - source[1])
    if delta_x == 0.0 and delta_y == 0.0:
        return 0
    if abs(delta_x) >= abs(delta_y):
        if delta_x > 0:
            return 3
        return 4
    if delta_y > 0:
        return 1
    return 2


def choose_action_baseline(
    info: dict[str, Any],
    rng: np.random.Generator,
    env: PacmanEnv,
    *,
    distance_bucket_size: int,
) -> int:
    """Baseline: go to food if ghost > 2 cells away, else move away from ghost."""
    legal_actions = info.get("legal_action_ids", [])
    if not legal_actions:
        return int(env.action_space.sample())

    state = env.runtime_state
    pacman_position = state.getPacmanPosition()
    food_positions = state.getFood().asList()
    ghost_positions = [
        ghost_state.getPosition()
        for ghost_state in state.getGhostStates()
        if ghost_state.getPosition() is not None
    ]

    nearest_food = min(
        food_positions,
        key=lambda pos: (abs(pos[0] - pacman_position[0]) + abs(pos[1] - pacman_position[1]), pos),
        default=None,
    )
    nearest_ghost = min(
        ghost_positions,
        key=lambda pos: (abs(pos[0] - pacman_position[0]) + abs(pos[1] - pacman_position[1]), pos),
        default=None,
    )

    ghost_distance = None
    if nearest_ghost is not None:
        ghost_distance = int(
            abs(nearest_ghost[0] - pacman_position[0]) + abs(nearest_ghost[1] - pacman_position[1])
        )
    ghost_bucket = -1 if ghost_distance is None else int(ghost_distance // max(1, distance_bucket_size))
    ghost_dir = 0 if nearest_ghost is None else _direction_to_target(pacman_position, nearest_ghost)
    food_dir = 0 if nearest_food is None else _direction_to_target(pacman_position, nearest_food)

    ghost_far = ghost_bucket < 0 or (ghost_bucket * max(1, distance_bucket_size) > 2)

    if not ghost_far and ghost_dir in _OBS_DIR_OPPOSITE:
        avoid_dir = _OBS_DIR_OPPOSITE[ghost_dir]
        avoid_action = _OBS_DIR_TO_ACTION.get(avoid_dir)
        if avoid_action in legal_actions:
            return int(avoid_action)

    food_action = _OBS_DIR_TO_ACTION.get(food_dir)
    if food_action in legal_actions:
        return int(food_action)

    return int(rng.choice(legal_actions))


def train_pi(
    *,
    config_path: str = "configs/policy_iteration_obs.yaml",
    output_dir: str | None = None,
    episodes: int | None = None,
    seed: int | None = None,
    log_every: int = 100,
) -> dict[str, Any]:
    """Collect empirical MDP data and run policy iteration."""
    configure_logging()

    cfg = load_yaml(config_path)
    env_cfg_dict = cfg.get("env", {})
    obs_mdp_cfg = cfg.get("obs_mdp", {})
    pi_cfg = cfg.get("policy_iteration", {})
    paths_cfg = cfg.get("paths", {})
    encoding_cfg = cfg.get("obs_encoding", {})
    empirical_mdp_path = obs_mdp_cfg.get("empirical_mdp_path")

    if seed is not None:
        env_cfg_dict = {**env_cfg_dict, "seed": seed}

    env_cfg = build_env_config(env_cfg_dict)
    env = PacmanEnv(config=env_cfg, render_mode=None)

    episodes = int(episodes if episodes is not None else obs_mdp_cfg.get("episodes", 60000))
    base_seed = int(obs_mdp_cfg.get("seed_base", env_cfg.seed))

    drop_keys = list(encoding_cfg.get("drop_keys", ["score", "step_count"]))
    float_round = int(encoding_cfg.get("float_round", 3))

    rng_seed = int(obs_mdp_cfg.get("policy_seed", env_cfg.seed))
    rng = np.random.default_rng(rng_seed)

    model = ObsMDPModel()

    default_output = paths_cfg.get("train_output_dir", "results/obs_policy_iteration")
    output_dir_path = ensure_directory(output_dir or default_output)

    returns: list[float] = []
    wins: list[bool] = []
    steps: list[int] = []

    distance_bucket_size = int(env_cfg.observation.distance_bucket_size)

    if empirical_mdp_path:
        model = load_pickle(Path(empirical_mdp_path))
        print(f"Loaded empirical MDP from: {empirical_mdp_path}")
    else:
        print(f"Collecting empirical MDP for {episodes} episodes...")
        for episode_idx in range(episodes):
            episode_seed = int(base_seed + episode_idx)
            observation, info = env.reset(seed=episode_seed)
            total_reward = 0.0
            step_count = 0

            while True:
                action = choose_action_baseline(
                    info,
                    rng,
                    env,
                    distance_bucket_size=distance_bucket_size,
                )

                next_observation, reward, terminated, truncated, next_info = env.step(action)

                model.update(
                    encode_observation(observation, drop_keys=drop_keys, float_round=float_round),
                    action,
                    float(reward),
                    encode_observation(next_observation, drop_keys=drop_keys, float_round=float_round),
                    bool(terminated),
                    bool(truncated),
                )

                total_reward += float(reward)
                step_count += 1
                observation = next_observation
                info = next_info

                if terminated or truncated:
                    returns.append(total_reward)
                    wins.append(bool(info.get("is_win", False)))
                    steps.append(step_count)
                    break

            if log_every > 0 and (episode_idx + 1) % log_every == 0:
                win_rate = sum(1 for value in wins if value) / max(1, len(wins))
                print(
                    "Collected "
                    f"{episode_idx + 1}/{episodes} episodes | wins={sum(wins)} | win_rate={win_rate:.4f}"
                )

    env.close()

    mini_eval_episodes = int(obs_mdp_cfg.get("mini_eval_episodes", 3))
    mini_eval_seed_base = int(obs_mdp_cfg.get("mini_eval_seed_base", env_cfg.seed + 20_000))
    eval_env = PacmanEnv(config=env_cfg, render_mode=None)
    mini_eval_returns: list[float] = []

    def run_mini_eval(iteration_idx: int, policy_table: dict[Any, int], _: dict[Any, float]) -> None:
        if mini_eval_episodes <= 0:
            return
        policy = ObsPolicy(
            policy_table,
            seed=env_cfg.seed,
            drop_keys=drop_keys,
            float_round=float_round,
            non_wall_coords=getattr(getattr(eval_env, "_observation_context", None), "non_wall_coords", []),
        )
        returns: list[float] = []
        for episode_idx in range(mini_eval_episodes):
            episode_seed = int(mini_eval_seed_base + iteration_idx * 100 + episode_idx)
            observation, info = eval_env.reset(seed=episode_seed)
            total_reward = 0.0
            while True:
                action = policy.select_action(observation, info)
                observation, reward, terminated, truncated, info = eval_env.step(action)
                total_reward += float(reward)
                if terminated or truncated:
                    returns.append(total_reward)
                    break
        mini_eval_returns.append(float(sum(returns) / max(1, len(returns))))

    model_path = output_dir_path / "empirical_mdp.pkl"
    if empirical_mdp_path:
        model_path = Path(empirical_mdp_path)
    else:
        save_pickle(model, model_path)

    print("Running policy iteration on empirical MDP...")
    pi_result: PolicyIterationResult = policy_iteration(
        model,
        gamma=float(pi_cfg.get("gamma", 0.95)),
        theta=float(pi_cfg.get("theta", 1e-4)),
        max_eval_iters=int(pi_cfg.get("max_eval_iters", 50)),
        max_policy_iters=int(pi_cfg.get("max_policy_iters", 50)),
        on_iteration=run_mini_eval,
    )
    print(
        "Policy iteration complete. "
        f"iterations={pi_result.policy_iterations}, eval_sweeps={pi_result.evaluation_sweeps}"
    )
    eval_env.close()

    policy_path = output_dir_path / "policy.pkl"
    values_path = output_dir_path / "values.pkl"
    save_pickle(pi_result.policy, policy_path)
    save_pickle(pi_result.values, values_path)

    curve_path = output_dir_path / "policy_iteration_curve.png"
    if pi_result.value_history:
        figure = plt.figure(figsize=(8, 4))
        axes = figure.add_subplot(111)
        axes.plot(range(1, len(pi_result.value_history) + 1), pi_result.value_history)
        axes.set_title("Policy Iteration Value (Mean V)")
        axes.set_xlabel("Policy Iteration Step")
        axes.set_ylabel("Mean V")
        axes.grid(True, alpha=0.3)
        figure.tight_layout()
        figure.savefig(curve_path)
        plt.close(figure)

    reward_curve_path = output_dir_path / "policy_iteration_reward_curve.png"
    if mini_eval_returns:
        figure = plt.figure(figsize=(8, 4))
        axes = figure.add_subplot(111)
        axes.plot(range(1, len(mini_eval_returns) + 1), mini_eval_returns)
        axes.set_title("Policy Iteration Mini-Eval Reward")
        axes.set_xlabel("Policy Iteration Step")
        axes.set_ylabel("Mean Return")
        axes.grid(True, alpha=0.3)
        figure.tight_layout()
        figure.savefig(reward_curve_path)
        plt.close(figure)

    mean_return = float(sum(returns) / max(1, len(returns)))
    win_rate = float(sum(1 for value in wins if value) / max(1, len(wins)))

    metrics: dict[str, Any] = {
        "episodes": episodes if not empirical_mdp_path else 0,
        "mean_return": mean_return,
        "win_rate": win_rate,
        "avg_steps": float(sum(steps) / max(1, len(steps))),
        "state_count": len(model.states()),
        "state_action_count": model.state_action_count(),
        "transition_count": model.transition_count(),
        "policy_iterations": pi_result.policy_iterations,
        "evaluation_sweeps": pi_result.evaluation_sweeps,
        "last_delta": pi_result.last_delta,
        "artifact_empirical_mdp": str(model_path),
        "artifact_policy": str(policy_path),
        "artifact_values": str(values_path),
        "artifact_policy_iteration_curve": str(curve_path),
        "artifact_policy_iteration_reward_curve": str(reward_curve_path),
    }
    save_json(metrics, output_dir_path / "train_metrics.json")

    return metrics


def eval_pi(
    *,
    config_path: str = "configs/policy_iteration_obs.yaml",
    model_path: str | None = None,
    output_dir: str | None = None,
    episodes: int | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """Evaluate a trained policy iteration artifact."""
    configure_logging()

    cfg = load_yaml(config_path)
    env_cfg_dict = cfg.get("env", {})
    eval_cfg = cfg.get("eval", {})
    paths_cfg = cfg.get("paths", {})
    encoding_cfg = cfg.get("obs_encoding", {})

    if seed is not None:
        env_cfg_dict = {**env_cfg_dict, "seed": seed}

    env_cfg = build_env_config(env_cfg_dict)
    env = PacmanEnv(config=env_cfg, render_mode=None)

    model_path = model_path or paths_cfg.get("model_path", "results/obs_policy_iteration/policy.pkl")
    policy_table = load_pickle(Path(model_path))

    drop_keys = list(encoding_cfg.get("drop_keys", ["score", "step_count"]))
    float_round = int(encoding_cfg.get("float_round", 3))
    policy = ObsPolicy(
        policy_table,
        seed=env_cfg.seed,
        drop_keys=drop_keys,
        float_round=float_round,
        non_wall_coords=getattr(getattr(env, "_observation_context", None), "non_wall_coords", []),
    )

    episodes = int(episodes if episodes is not None else eval_cfg.get("episodes", 100))
    base_seed = int(eval_cfg.get("seed_base", env_cfg.seed + 10_000))

    default_output = paths_cfg.get("eval_output_dir", "results/obs_policy_iteration_eval")
    output_dir_path = ensure_directory(output_dir or default_output)

    returns: list[float] = []
    wins: list[bool] = []
    steps: list[int] = []

    for episode_idx in range(episodes):
        episode_seed = int(base_seed + episode_idx)
        observation, info = env.reset(seed=episode_seed)
        total_reward = 0.0
        step_count = 0

        while True:
            action = policy.select_action(observation, info)
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            step_count += 1
            if terminated or truncated:
                returns.append(total_reward)
                wins.append(bool(info.get("is_win", False)))
                steps.append(step_count)
                break

    env.close()

    metrics: dict[str, Any] = {
        "episodes": episodes,
        "mean_return": float(sum(returns) / max(1, len(returns))),
        "win_rate": float(sum(1 for value in wins if value) / max(1, len(wins))),
        "avg_steps": float(sum(steps) / max(1, len(steps))),
        "returns": returns,
        "wins": wins,
        "steps": steps,
        "model_path": str(model_path),
        "unseen_state_fallbacks": int(getattr(policy, "fallback_count", 0)),
    }
    save_json(metrics, output_dir_path / "eval_metrics.json")

    return metrics


def run_pi(
    *,
    config_path: str = "configs/policy_iteration_obs.yaml",
    model_path: str | None = None,
    render_mode: str = "human",
    episodes: int = 1,
    seed: int | None = None,
    gif_title: str | None = None,
    no_gif: bool = False,
) -> None:
    """Run a trained PI policy in the live environment."""
    cfg = load_yaml(config_path)
    env_cfg_dict = cfg.get("env", {})
    paths_cfg = cfg.get("paths", {})
    encoding_cfg = cfg.get("obs_encoding", {})

    if seed is not None:
        env_cfg_dict = {**env_cfg_dict, "seed": seed}

    env_cfg = build_env_config(env_cfg_dict)
    env = PacmanEnv(config=env_cfg, render_mode=render_mode)

    model_path = model_path or paths_cfg.get("model_path", "results/obs_policy_iteration/policy.pkl")
    policy_table = load_pickle(Path(model_path))

    drop_keys = list(encoding_cfg.get("drop_keys", ["score", "step_count"]))
    float_round = int(encoding_cfg.get("float_round", 3))
    policy = ObsPolicy(
        policy_table,
        seed=env_cfg.seed,
        drop_keys=drop_keys,
        float_round=float_round,
        non_wall_coords=getattr(getattr(env, "_observation_context", None), "non_wall_coords", []),
    )

    save_gif_enabled = (not no_gif) and (render_mode == "human")
    gif_path: Path | None = None
    if save_gif_enabled:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"pi_run_{stamp}.gif" if not gif_title else (gif_title if gif_title.endswith(".gif") else f"{gif_title}.gif")
        gif_dir = ensure_directory("results/important")
        gif_path = gif_dir / filename

    gif_frames: list[Image.Image] = []

    for episode_idx in range(episodes):
        observation, info = env.reset(seed=env_cfg.seed + episode_idx)
        total_reward = 0.0
        if save_gif_enabled:
            frame = _capture_human_frame()
            if frame is not None:
                gif_frames.append(frame)
        while True:
            action = policy.select_action(observation, info)
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            if save_gif_enabled:
                frame = _capture_human_frame()
                if frame is not None:
                    gif_frames.append(frame)
            if terminated or truncated:
                print(f"Episode {episode_idx + 1} total reward: {total_reward:.2f}")
                break

    env.close()
    if save_gif_enabled and gif_path is not None:
        save_gif(gif_frames, gif_path, frame_time=env_cfg.frame_time)
        print(f"Saved GIF: {gif_path}")


def _capture_human_frame() -> Image.Image | None:
    """Capture only the Pacman Tk canvas for GIF export."""
    return capture_human_frame(max_size=(1600, 1200))
