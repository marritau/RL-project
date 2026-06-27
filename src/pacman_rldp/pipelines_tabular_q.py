"""Simple tabular Q-learning and SARSA pipelines over encoded observations."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .agents import BaselineNearestFoodAvoidGhostPolicy
from .algorithms.food_bitmask_value_iteration import FoodBitmaskHelper
from .algorithms.policy_iteration.obs_encoding import encode_observation
from .env import PacmanEnv, build_env_config
from .logging import configure_logging
from .utils import ensure_directory, load_pickle, load_yaml, save_json, save_pickle
from .visuals.capture import capture_human_frame, save_gif


class TabularAgent:
    """Base tabular agent with epsilon-greedy behavior over legal actions."""

    def __init__(self, action_size: int, alpha: float, gamma: float, epsilon: float) -> None:
        self.action_size = int(action_size)
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.epsilon = float(epsilon)
        self.q_table: dict[Any, np.ndarray] = {}
        self._rng = np.random.default_rng()

    def seed(self, seed: int) -> None:
        self._rng = np.random.default_rng(int(seed))

    def _get_q(self, state_key: Any) -> np.ndarray:
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_size, dtype=np.float32)
        return self.q_table[state_key]

    def select_action(self, state_key: Any, legal_actions: list[int]) -> int:
        legal_actions = [int(action) for action in legal_actions]
        if not legal_actions:
            return int(self._rng.integers(0, self.action_size))
        if self._rng.random() < self.epsilon:
            return int(self._rng.choice(legal_actions))
        q_values = self._get_q(state_key)
        return _greedy_action(q_values, legal_actions, self.action_size, self._rng)


class QLearningAgent(TabularAgent):
    """Off-policy one-step Q-learning."""

    def update(
        self,
        state_key: Any,
        action: int,
        reward: float,
        next_state_key: Any,
        next_legal: list[int],
        done: bool,
    ) -> None:
        q_values = self._get_q(state_key)
        current = float(q_values[int(action)])
        if done or not next_legal:
            target = float(reward)
        else:
            next_q = self._get_q(next_state_key)
            target = float(reward) + self.gamma * float(np.max(next_q[[int(a) for a in next_legal]]))
        q_values[int(action)] = current + self.alpha * (target - current)


class SarsaAgent(TabularAgent):
    """On-policy one-step SARSA."""

    def update(
        self,
        state_key: Any,
        action: int,
        reward: float,
        next_state_key: Any,
        next_action: int | None,
        done: bool,
    ) -> None:
        q_values = self._get_q(state_key)
        current = float(q_values[int(action)])
        if done or next_action is None:
            target = float(reward)
        else:
            next_q = self._get_q(next_state_key)
            # SARSA must bootstrap from the actually selected next action a', not
            # from max_a Q(s', a).  This is the main difference from Q-learning.
            target = float(reward) + self.gamma * float(next_q[int(next_action)])
        q_values[int(action)] = current + self.alpha * (target - current)


def _encode_obs(observation: dict[str, Any], drop_keys: list[str], float_round: int) -> Any:
    return encode_observation(observation, drop_keys=drop_keys, float_round=float_round)


def _resolve_agent_params(
    cfg: dict[str, Any],
    *,
    alpha: float | None,
    gamma: float | None,
    epsilon: float | None,
) -> tuple[float, float, float]:
    train_cfg = dict(cfg.get("train", {}))
    agent_params = dict(train_cfg.get("agent_params", {}))
    return (
        float(alpha if alpha is not None else agent_params.get("alpha", 0.1)),
        float(gamma if gamma is not None else agent_params.get("gamma", 0.99)),
        float(epsilon if epsilon is not None else agent_params.get("epsilon", 0.1)),
    )


def _resolve_episode_count(cfg: dict[str, Any], episodes: int | None, default: int) -> int:
    if episodes is not None:
        return int(episodes)
    return int(dict(cfg.get("train", {})).get("episodes", default))


def _resolve_seed(cfg: dict[str, Any], seed: int | None, env_seed: int) -> int:
    if seed is not None:
        return int(seed)
    train_cfg = dict(cfg.get("train", {}))
    return int(train_cfg.get("seed", env_seed))


def _greedy_action(
    q_values: np.ndarray,
    legal_actions: list[int],
    action_size: int,
    rng: np.random.Generator,
) -> int:
    if not legal_actions:
        return int(rng.integers(0, int(action_size)))
    masked = np.full(int(action_size), -np.inf, dtype=np.float32)
    for action in legal_actions:
        masked[int(action)] = q_values[int(action)]
    best = np.flatnonzero(masked == np.max(masked))
    return int(rng.choice(best))


def _action_from_q_table(
    q_table: dict[Any, np.ndarray],
    state_key: Any,
    legal_actions: list[int],
    action_size: int,
    rng: np.random.Generator,
) -> int:
    legal_actions = [int(action) for action in legal_actions]
    q_values = q_table.get(state_key)
    if q_values is None:
        return int(rng.choice(legal_actions)) if legal_actions else int(rng.integers(0, action_size))
    return _greedy_action(np.asarray(q_values), legal_actions, action_size, rng)


def train_q_learning(
    *,
    config_path: str = "configs/default.yaml",
    output_dir: str | None = None,
    episodes: int | None = None,
    alpha: float | None = None,
    gamma: float | None = None,
    epsilon: float | None = None,
    epsilon_decay: float | None = None,
    epsilon_end: float | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """Train a tabular Q-learning agent."""
    configure_logging()
    cfg = load_yaml(config_path)
    env_cfg_dict = dict(cfg.get("env", {}))
    paths_cfg = dict(cfg.get("paths", {}))
    encoding_cfg = dict(cfg.get("obs_encoding", {}))
    train_cfg = dict(cfg.get("train", {}))

    env_cfg = build_env_config(env_cfg_dict)
    base_seed = _resolve_seed(cfg, seed, env_cfg.seed)
    env_cfg_dict["seed"] = base_seed
    env_cfg = build_env_config(env_cfg_dict)
    env = PacmanEnv(config=env_cfg, render_mode=None)

    alpha_value, gamma_value, epsilon_value = _resolve_agent_params(
        cfg,
        alpha=alpha,
        gamma=gamma,
        epsilon=epsilon,
    )
    decay_value = float(epsilon_decay if epsilon_decay is not None else train_cfg.get("epsilon_decay", 1.0))
    epsilon_floor = float(epsilon_end if epsilon_end is not None else train_cfg.get("epsilon_end", 0.0))
    episode_count = _resolve_episode_count(cfg, episodes, 1000)

    drop_keys = list(encoding_cfg.get("drop_keys", ["score", "step_count"]))
    float_round = int(encoding_cfg.get("float_round", 3))

    agent = QLearningAgent(env.action_space.n, alpha=alpha_value, gamma=gamma_value, epsilon=epsilon_value)
    agent.seed(base_seed)

    returns: list[float] = []
    wins: list[bool] = []

    for episode_idx in range(episode_count):
        observation, info = env.reset(seed=base_seed + episode_idx)
        total_reward = 0.0
        while True:
            state_key = _encode_obs(observation, drop_keys, float_round)
            legal_actions = [int(a) for a in info.get("legal_action_ids", [])]
            action = agent.select_action(state_key, legal_actions)
            next_obs, reward, terminated, truncated, next_info = env.step(action)
            done = bool(terminated or truncated)
            next_state_key = _encode_obs(next_obs, drop_keys, float_round)
            next_legal = [int(a) for a in next_info.get("legal_action_ids", [])]
            agent.update(state_key, action, float(reward), next_state_key, next_legal, done)
            total_reward += float(reward)
            observation = next_obs
            info = next_info
            if done:
                wins.append(bool(info.get("is_win", False)))
                returns.append(total_reward)
                break

        agent.epsilon = max(epsilon_floor, agent.epsilon * decay_value)

    env.close()

    default_output = paths_cfg.get("train_q_learning_output_dir", "results/train_q_learning")
    output_dir_path = ensure_directory(output_dir or default_output)
    model_path = output_dir_path / "q_table.pkl"
    save_pickle(agent.q_table, model_path)

    metrics = {
        "algorithm": "q_learning",
        "episodes": episode_count,
        "mean_return": float(sum(returns) / max(1, len(returns))),
        "win_rate": float(sum(1 for value in wins if value) / max(1, len(wins))),
        "alpha": alpha_value,
        "gamma": gamma_value,
        "epsilon_start": float(epsilon_value),
        "epsilon_final": float(agent.epsilon),
        "epsilon_decay": float(decay_value),
        "epsilon_end": float(epsilon_floor),
        "q_table_size": int(len(agent.q_table)),
        "base_seed": int(base_seed),
        "model_path": str(model_path),
    }
    save_json(metrics, output_dir_path / "train_q_learning_metrics.json")
    return metrics


def train_sarsa(
    *,
    config_path: str = "configs/sarsa.yaml",
    output_dir: str | None = None,
    episodes: int | None = None,
    alpha: float | None = None,
    gamma: float | None = None,
    epsilon: float | None = None,
    epsilon_decay: float | None = None,
    epsilon_end: float | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """Train a tabular SARSA agent with optional epsilon decay from config."""
    configure_logging()
    cfg = load_yaml(config_path)
    env_cfg_dict = dict(cfg.get("env", {}))
    paths_cfg = dict(cfg.get("paths", {}))
    encoding_cfg = dict(cfg.get("obs_encoding", {}))
    train_cfg = dict(cfg.get("train", {}))

    env_cfg = build_env_config(env_cfg_dict)
    base_seed = _resolve_seed(cfg, seed, env_cfg.seed)
    env_cfg_dict["seed"] = base_seed
    env_cfg = build_env_config(env_cfg_dict)
    env = PacmanEnv(config=env_cfg, render_mode=None)

    alpha_value, gamma_value, epsilon_start = _resolve_agent_params(
        cfg,
        alpha=alpha,
        gamma=gamma,
        epsilon=epsilon,
    )
    decay_value = float(epsilon_decay if epsilon_decay is not None else train_cfg.get("epsilon_decay", 1.0))
    epsilon_floor = float(epsilon_end if epsilon_end is not None else train_cfg.get("epsilon_end", 0.0))
    episode_count = _resolve_episode_count(cfg, episodes, 1000)

    drop_keys = list(encoding_cfg.get("drop_keys", ["score", "step_count"]))
    float_round = int(encoding_cfg.get("float_round", 3))

    agent = SarsaAgent(env.action_space.n, alpha=alpha_value, gamma=gamma_value, epsilon=epsilon_start)
    agent.seed(base_seed)

    returns: list[float] = []
    wins: list[bool] = []

    for episode_idx in range(episode_count):
        observation, info = env.reset(seed=base_seed + episode_idx)
        total_reward = 0.0
        state_key = _encode_obs(observation, drop_keys, float_round)
        legal_actions = [int(a) for a in info.get("legal_action_ids", [])]
        action = agent.select_action(state_key, legal_actions)

        while True:
            next_obs, reward, terminated, truncated, next_info = env.step(action)
            done = bool(terminated or truncated)
            next_state_key = _encode_obs(next_obs, drop_keys, float_round)
            next_legal = [int(a) for a in next_info.get("legal_action_ids", [])]
            next_action = agent.select_action(next_state_key, next_legal) if not done else None
            agent.update(state_key, action, float(reward), next_state_key, next_action, done)

            total_reward += float(reward)
            state_key = next_state_key
            if next_action is not None:
                action = next_action

            if done:
                wins.append(bool(next_info.get("is_win", False)))
                returns.append(total_reward)
                break

        agent.epsilon = max(epsilon_floor, agent.epsilon * decay_value)

    env.close()

    default_output = paths_cfg.get("train_sarsa_output_dir", "results/train_sarsa")
    output_dir_path = ensure_directory(output_dir or default_output)
    model_path = output_dir_path / "q_table.pkl"
    save_pickle(agent.q_table, model_path)

    metrics = {
        "algorithm": "sarsa",
        "episodes": episode_count,
        "mean_return": float(sum(returns) / max(1, len(returns))),
        "win_rate": float(sum(1 for value in wins if value) / max(1, len(wins))),
        "alpha": alpha_value,
        "gamma": gamma_value,
        "epsilon_start": float(epsilon_start),
        "epsilon_final": float(agent.epsilon),
        "epsilon_decay": float(decay_value),
        "epsilon_end": float(epsilon_floor),
        "q_table_size": int(len(agent.q_table)),
        "base_seed": int(base_seed),
        "model_path": str(model_path),
    }
    save_json(metrics, output_dir_path / "train_sarsa_metrics.json")
    return metrics


def eval_tabular_q(
    *,
    config_path: str = "configs/default.yaml",
    model_path: str,
    output_dir: str | None = None,
    episodes: int = 200,
    seed: int | None = None,
    algo_name: str = "tabular_q",
    fallback_on_zero_q: bool = False,
) -> dict[str, Any]:
    """Evaluate a saved tabular Q/SARSA table greedily."""
    configure_logging()
    cfg = load_yaml(config_path)
    env_cfg_dict = dict(cfg.get("env", {}))
    encoding_cfg = dict(cfg.get("obs_encoding", {}))

    if seed is not None:
        env_cfg_dict["seed"] = int(seed)

    env_cfg = build_env_config(env_cfg_dict)
    env = PacmanEnv(config=env_cfg, render_mode=None)
    rng = np.random.default_rng(env_cfg.seed)

    drop_keys = list(encoding_cfg.get("drop_keys", ["score", "step_count"]))
    float_round = int(encoding_cfg.get("float_round", 3))

    q_table = load_pickle(model_path)
    non_wall_coords = getattr(getattr(env, "_observation_context", None), "non_wall_coords", [])
    food_helper = FoodBitmaskHelper(non_wall_coords)
    raw_baseline = BaselineNearestFoodAvoidGhostPolicy()
    unseen_state_fallbacks = 0

    returns: list[float] = []
    wins: list[bool] = []

    for episode_idx in range(episodes):
        observation, info = env.reset(seed=env_cfg.seed + episode_idx)
        total_reward = 0.0
        while True:
            state_key = _encode_obs(observation, drop_keys, float_round)
            legal_actions = [int(a) for a in info.get("legal_action_ids", [])]
            q_values = q_table.get(state_key)
            use_fallback = q_values is None
            if q_values is not None and fallback_on_zero_q:
                legal_values = np.asarray(q_values)[legal_actions] if legal_actions else np.asarray([])
                use_fallback = bool(legal_values.size and np.allclose(legal_values, 0.0))
            if use_fallback:
                unseen_state_fallbacks += 1
                action = _heuristic_fallback_action(
                    observation,
                    info,
                    legal_actions,
                    rng=rng,
                    food_helper=food_helper,
                    raw_baseline=raw_baseline,
                )
            else:
                q_values = q_table.get(state_key)
            if q_values is None:
                action = _heuristic_fallback_action(
                    observation,
                    info,
                    legal_actions,
                    rng=rng,
                    food_helper=food_helper,
                    raw_baseline=raw_baseline,
                )
            else:
                action = _action_from_q_table(q_table, state_key, legal_actions, env.action_space.n, rng)
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            if terminated or truncated:
                returns.append(total_reward)
                wins.append(bool(info.get("is_win", False)))
                break

    env.close()

    default_output = f"results/eval_{algo_name}"
    output_dir_path = ensure_directory(output_dir or default_output)

    metrics = {
        "episodes": int(episodes),
        "mean_return": float(sum(returns) / max(1, len(returns))),
        "win_rate": float(sum(1 for value in wins if value) / max(1, len(wins))),
        "model_path": str(model_path),
        "algorithm": str(algo_name),
        "unseen_state_fallbacks": int(unseen_state_fallbacks),
        "fallback_on_zero_q": bool(fallback_on_zero_q),
    }
    save_json(metrics, output_dir_path / f"eval_{algo_name}_metrics.json")
    return metrics


def _heuristic_fallback_action(
    observation: dict[str, Any],
    info: dict[str, Any],
    legal_actions: list[int],
    *,
    rng: np.random.Generator,
    food_helper: FoodBitmaskHelper,
    raw_baseline: BaselineNearestFoodAvoidGhostPolicy,
) -> int:
    if not legal_actions:
        return 4
    if {"walls", "food", "pacman_position", "ghost_positions", "ghost_present"}.issubset(observation.keys()):
        action = int(raw_baseline.select_action(observation, info))
        if action in legal_actions:
            return action
    if "food_bitmask" in observation:
        return int(
            food_helper.heuristic_action(
                observation,
                legal_actions,
                rng=rng,
                epsilon_random=0.0,
                danger_distance=2,
            )
        )
    non_stop = [action for action in legal_actions if action != 4]
    if non_stop:
        return int(rng.choice(non_stop))
    return int(rng.choice(legal_actions))


def run_tabular_q(
    *,
    config_path: str = "configs/default.yaml",
    model_path: str,
    render_mode: str = "human",
    episodes: int = 1,
    seed: int | None = None,
    gif_title: str | None = None,
    no_gif: bool = False,
) -> None:
    cfg = load_yaml(config_path)
    env_cfg_dict = dict(cfg.get("env", {}))
    encoding_cfg = dict(cfg.get("obs_encoding", {}))

    if seed is not None:
        env_cfg_dict["seed"] = int(seed)

    env_cfg = build_env_config(env_cfg_dict)
    rng = np.random.default_rng(env_cfg.seed)
    save_gif_enabled = (not no_gif) and (render_mode == "human")
    gif_path: Path | None = None
    if save_gif_enabled:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"tabular_q_{stamp}.gif" if not gif_title else (gif_title if gif_title.endswith(".gif") else f"{gif_title}.gif")
        gif_dir = ensure_directory("results/important")
        gif_path = gif_dir / filename

    env = PacmanEnv(config=env_cfg, render_mode=render_mode)

    drop_keys = list(encoding_cfg.get("drop_keys", ["score", "step_count"]))
    float_round = int(encoding_cfg.get("float_round", 3))

    q_table = load_pickle(model_path)
    non_wall_coords = getattr(getattr(env, "_observation_context", None), "non_wall_coords", [])
    food_helper = FoodBitmaskHelper(non_wall_coords)
    raw_baseline = BaselineNearestFoodAvoidGhostPolicy()

    gif_frames: list[Image.Image] = []
    for episode_idx in range(episodes):
        observation, info = env.reset(seed=env_cfg.seed + episode_idx)
        if save_gif_enabled:
            frame = _capture_human_frame()
            if frame is not None:
                gif_frames.append(frame)
        while True:
            state_key = _encode_obs(observation, drop_keys, float_round)
            legal_actions = [int(a) for a in info.get("legal_action_ids", [])]
            q_values = q_table.get(state_key)
            if q_values is None:
                action = _heuristic_fallback_action(
                    observation,
                    info,
                    legal_actions,
                    rng=rng,
                    food_helper=food_helper,
                    raw_baseline=raw_baseline,
                )
            else:
                action = _action_from_q_table(q_table, state_key, legal_actions, env.action_space.n, rng)
            observation, reward, terminated, truncated, info = env.step(action)
            if save_gif_enabled:
                frame = _capture_human_frame()
                if frame is not None:
                    gif_frames.append(frame)
            if terminated or truncated:
                break

    env.close()
    if save_gif_enabled and gif_path is not None:
        save_gif(gif_frames, gif_path, frame_time=env_cfg.frame_time)
        print(f"Saved GIF: {gif_path}")


def _capture_human_frame() -> Image.Image | None:
    """Capture only the Pacman Tk canvas for GIF export."""
    return capture_human_frame(max_size=(1600, 1200))
