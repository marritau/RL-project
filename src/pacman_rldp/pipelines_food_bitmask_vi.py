"""Train/eval entry points for food-bitmask approximate value iteration."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from .algorithms.food_bitmask_value_iteration import FoodBitmaskObservationModelBuilder, FoodBitmaskValueIterationPlanner, FoodBitmaskValueIterationPolicy
from .env import PacmanEnv, build_env_config
from .logging import configure_logging
from .visuals.capture import capture_human_frame, save_gif
from .utils import ensure_directory, load_pickle, load_yaml, save_json, save_pickle


def _build_gif_filename(gif_title: str | None) -> str:
    if gif_title is None or not gif_title.strip():
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"food_bitmask_vi_best_{stamp}.gif"
    normalized = gif_title.strip()
    if not normalized.lower().endswith(".gif"):
        normalized = f"{normalized}.gif"
    return normalized


def _save_diagnostic_training_plot(output_dir: Path, collection_summary: dict[str, Any], residual_history: list[float]) -> Path:
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(collection_summary["states_seen_by_episode"], label="States discovered")
    axes[0].plot(collection_summary["samples_seen_by_episode"], label="Transition samples")
    axes[0].set_title("Empirical model coverage")
    axes[0].set_xlabel("Collection episode")
    axes[0].grid(True)
    axes[0].legend()

    axes[1].plot(residual_history)
    axes[1].set_title("Value iteration residual")
    axes[1].set_xlabel("VI iteration")
    axes[1].set_ylabel("Max Bellman residual")
    axes[1].set_yscale("log")
    axes[1].grid(True)

    plot_path = output_dir / "training_curves.png"
    figure.tight_layout()
    figure.savefig(plot_path, dpi=160)
    plt.close(figure)
    return plot_path


def _save_training_reward_plot(important_dir: Path, exploratory_scores: list[float]) -> Path:
    figure, ax = plt.subplots(figsize=(8, 4))
    ax.plot(exploratory_scores, linewidth=1.2, label="Episode score")
    if exploratory_scores:
        window = min(50, len(exploratory_scores))
        if window >= 2:
            moving = []
            running = 0.0
            for idx, score in enumerate(exploratory_scores):
                running += float(score)
                if idx >= window:
                    running -= float(exploratory_scores[idx - window])
                moving.append(running / min(idx + 1, window))
            ax.plot(moving, linewidth=2.0, label=f"Moving average ({window})")
    ax.set_title("Pacman score per training episode")
    ax.set_xlabel("Training episode")
    ax.set_ylabel("Score")
    ax.grid(True)
    ax.legend()
    plot_path = important_dir / "train_bitmask_vi_reward_curve.png"
    figure.tight_layout()
    figure.savefig(plot_path, dpi=160)
    plt.close(figure)
    return plot_path


def _capture_human_frame_fullscreen() -> Image.Image | None:
    """Capture only the Pacman canvas, not the whole desktop.

    The old implementation used ImageGrab.grab(all_screens=True), which produces
    2520x1680 frames on your display and can make the game look cropped or
    shifted inside a huge GIF.  We keep the function name for compatibility, but
    delegate to the canvas-only capture helper.
    """
    return capture_human_frame(max_size=(1600, 1200))


def _run_eval_episode(env: PacmanEnv, policy: FoodBitmaskValueIterationPolicy, *, seed: int, render_mode: str, capture_frames: bool) -> tuple[float, bool, int, float, list[Image.Image]]:
    observation, info = env.reset(seed=seed)
    total_reward = 0.0
    step_count = 0
    final_score = float(info.get("score", 0.0))
    frames: list[Image.Image] = []

    if render_mode == "human" and capture_frames:
        frame = _capture_human_frame_fullscreen()
        if frame is not None:
            frames.append(frame.copy())

    while True:
        action = policy.select_action(observation, info)
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        step_count += 1
        final_score = float(info.get("score", final_score))

        if render_mode == "ansi":
            rendered = env.render()
            if rendered is not None:
                print(rendered)
        elif render_mode == "human" and capture_frames:
            frame = _capture_human_frame_fullscreen()
            if frame is not None:
                frames.append(frame.copy())

        if terminated or truncated:
            return total_reward, bool(info.get("is_win", False)), step_count, final_score, frames


def _is_better_episode(*, candidate_win: bool, candidate_score: float, candidate_return: float, candidate_steps: int, best_win: bool, best_score: float, best_return: float, best_steps: int) -> bool:
    if candidate_win != best_win:
        return candidate_win and not best_win
    if candidate_score != best_score:
        return candidate_score > best_score
    if candidate_return != best_return:
        return candidate_return > best_return
    if candidate_steps != best_steps:
        return candidate_steps < best_steps
    return False


def train_food_bitmask_value_iteration(
    *,
    config_path: str = "configs/bitmask_value_iteration.yaml",
    output_dir: str | None = None,
    collection_episodes: int | None = None,
    gamma: float | None = None,
    max_iterations: int | None = None,
    tolerance: float | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    configure_logging()
    cfg = load_yaml(config_path)
    env_cfg_dict = dict(cfg.get("env", {}))
    collection_cfg = dict(cfg.get("collection", {}))
    planner_cfg = dict(cfg.get("planner_food_bitmask_vi", {}))
    paths_cfg = dict(cfg.get("paths", {}))

    if seed is not None:
        env_cfg_dict["seed"] = int(seed)
        collection_cfg["base_seed"] = int(seed)

    env_cfg = build_env_config(env_cfg_dict)
    env = PacmanEnv(config=env_cfg, render_mode=None)

    collection_episodes = int(collection_episodes if collection_episodes is not None else collection_cfg.get("episodes", 2000))
    base_seed = int(collection_cfg.get("base_seed", env_cfg.seed))
    collection_progress = int(collection_cfg.get("progress_every", 100))
    max_steps_per_episode = collection_cfg.get("max_steps_per_episode")
    epsilon_random = float(collection_cfg.get("epsilon_random", 0.35))
    danger_distance = int(collection_cfg.get("danger_distance", 2))

    gamma = float(gamma if gamma is not None else planner_cfg.get("gamma", 0.99))
    tolerance = float(tolerance if tolerance is not None else planner_cfg.get("tolerance", 1.0e-6))
    max_iterations = int(max_iterations if max_iterations is not None else planner_cfg.get("max_iterations", 500))
    planner_progress = int(planner_cfg.get("progress_every", 25))

    output_dir_path = ensure_directory(output_dir or paths_cfg.get("train_food_bitmask_vi_output_dir", "results/train_food_bitmask_vi"))
    important_dir = ensure_directory(paths_cfg.get("important_output_dir", "results/important"))

    collector = FoodBitmaskObservationModelBuilder(
        env=env,
        base_seed=base_seed,
        collection_episodes=collection_episodes,
        progress_every=collection_progress,
        max_steps_per_episode=max_steps_per_episode,
        epsilon_random=epsilon_random,
        danger_distance=danger_distance,
    )
    collection_result = collector.collect()

    planner = FoodBitmaskValueIterationPlanner(
        collection_result,
        gamma=gamma,
        tolerance=tolerance,
        max_iterations=max_iterations,
        progress_every=planner_progress,
    )
    result = planner.solve()
    artifact = planner.build_artifact(
        result=result,
        collection_config=collection_cfg,
        env_config=env_cfg_dict,
        raw_config=cfg,
        non_wall_coords=env._observation_context.non_wall_coords,
    )

    model_path = output_dir_path / "model.pkl"
    save_pickle(artifact, model_path)
    diagnostic_plot_path = _save_diagnostic_training_plot(output_dir_path, artifact["collection_summary"], result.residual_history)
    reward_plot_path = _save_training_reward_plot(important_dir, artifact["collection_summary"].get("exploratory_scores", []))

    metrics = {
        "model_path": str(model_path),
        "training_plot": str(diagnostic_plot_path),
        "important_training_reward_plot": str(reward_plot_path),
        **artifact["summary"],
        "collection_episodes": collection_episodes,
        "collection_mean_return": float(sum(collection_result.exploratory_returns) / max(1, len(collection_result.exploratory_returns))),
        "collection_mean_score": float(sum(collection_result.exploratory_scores) / max(1, len(collection_result.exploratory_scores))),
        "collection_mean_steps": float(sum(collection_result.exploratory_lengths) / max(1, len(collection_result.exploratory_lengths))),
        "epsilon_random": epsilon_random,
        "danger_distance": danger_distance,
    }
    metrics_path = output_dir_path / "train_bitmask_vi_metrics.json"
    save_json(metrics, metrics_path)
    save_json(metrics, important_dir / "train_bitmask_vi_metrics.json")

    with (output_dir_path / "planner_summary.txt").open("w", encoding="utf-8") as handle:
        handle.write("Approximate food-bitmask empirical value iteration\n")
        handle.write(f"layout: {env_cfg.layout_name}\n")
        handle.write(f"ghost_policy: {env_cfg.ghost_policy}\n")
        handle.write(f"collection_episodes: {collection_episodes}\n")
        handle.write(f"gamma: {gamma}\n")
        handle.write(f"tolerance: {tolerance}\n")
        handle.write(f"epsilon_random: {epsilon_random}\n")
        handle.write(f"danger_distance: {danger_distance}\n")
        handle.write(f"discovered_states: {artifact['summary']['discovered_states']}\n")
        handle.write(f"transition_samples: {artifact['summary']['transition_samples']}\n")
        handle.write(f"iterations: {artifact['summary']['iterations']}\n")
        handle.write(f"final_residual: {artifact['summary']['final_residual']:.8f}\n")
        handle.write(f"total_seconds: {artifact['summary']['total_seconds']:.6f}\n")
        handle.write(f"training_plot: {diagnostic_plot_path}\n")
        handle.write(f"important_training_reward_plot: {reward_plot_path}\n")

    env.close()
    return {
        "artifact": artifact,
        "metrics": metrics,
        "model_path": str(model_path),
        "diagnostic_plot_path": str(diagnostic_plot_path),
        "important_reward_plot_path": str(reward_plot_path),
        "important_metrics_path": str(important_dir / "train_bitmask_vi_metrics.json"),
    }


def eval_food_bitmask_value_iteration(
    *,
    config_path: str = "configs/bitmask_value_iteration.yaml",
    model_path: str | None = None,
    output_dir: str | None = None,
    episodes: int | None = None,
    seed: int | None = None,
    render_mode: str | None = None,
    gif_title: str | None = None,
    no_gif: bool = False,
) -> dict[str, Any]:
    configure_logging()
    cfg = load_yaml(config_path)
    env_cfg_dict = dict(cfg.get("env", {}))
    eval_cfg = dict(cfg.get("eval_food_bitmask_vi", {}))
    paths_cfg = dict(cfg.get("paths", {}))

    base_seed = int(seed if seed is not None else eval_cfg.get("base_seed", env_cfg_dict.get("seed", 42)))
    episodes = int(episodes if episodes is not None else eval_cfg.get("episodes", 200))
    render_mode_value = render_mode if render_mode is not None else eval_cfg.get("render_mode", "none")
    eval_render_mode = None if render_mode_value == "none" else render_mode_value

    env_cfg_dict["seed"] = base_seed
    env_cfg = build_env_config(env_cfg_dict)
    env = PacmanEnv(config=env_cfg, render_mode=eval_render_mode)

    model_path = model_path or paths_cfg.get("model_food_bitmask_vi_path", "results/train_food_bitmask_vi/model.pkl")
    model_data = load_pickle(model_path)
    if not isinstance(model_data, dict):
        raise ValueError("Model artifact must be a dictionary.")

    policy = FoodBitmaskValueIterationPolicy(model_data=model_data, fallback_seed=base_seed)

    output_dir_path = ensure_directory(output_dir or paths_cfg.get("eval_food_bitmask_vi_output_dir", "results/eval_food_bitmask_vi"))
    important_dir = ensure_directory(paths_cfg.get("important_output_dir", "results/important"))

    save_gif_enabled = (not no_gif) and (render_mode_value == "human")
    gif_path: Path | None = None
    if save_gif_enabled:
        gif_path = important_dir / _build_gif_filename(gif_title or eval_cfg.get("gif_title"))

    returns: list[float] = []
    wins: list[bool] = []
    steps: list[int] = []
    scores: list[float] = []
    best_episode_index = -1
    best_win = False
    best_score = float("-inf")
    best_return = float("-inf")
    best_steps = 10**18
    best_frames: list[Image.Image] = []

    for episode_idx in range(episodes):
        episode_seed = base_seed + episode_idx
        total_reward, did_win, episode_steps, final_score, frames = _run_eval_episode(
            env,
            policy,
            seed=episode_seed,
            render_mode=render_mode_value,
            capture_frames=save_gif_enabled,
        )
        returns.append(total_reward)
        wins.append(did_win)
        steps.append(episode_steps)
        scores.append(final_score)

        if best_episode_index < 0 or _is_better_episode(
            candidate_win=did_win,
            candidate_score=final_score,
            candidate_return=total_reward,
            candidate_steps=episode_steps,
            best_win=best_win,
            best_score=best_score,
            best_return=best_return,
            best_steps=best_steps,
        ):
            best_episode_index = episode_idx
            best_win = did_win
            best_score = final_score
            best_return = total_reward
            best_steps = episode_steps
            best_frames = [frame.copy() for frame in frames]

    env.close()

    if save_gif_enabled and gif_path is not None and best_frames:
        save_gif(best_frames, gif_path, frame_time=env_cfg.frame_time)

    eval_plot_path = output_dir_path / "eval_curves.png"
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(returns)
    axes[0].set_title("Return per evaluation episode")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Return")
    axes[0].grid(True)
    axes[1].plot(steps)
    axes[1].set_title("Episode length")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Steps")
    axes[1].grid(True)
    figure.tight_layout()
    figure.savefig(eval_plot_path, dpi=160)
    plt.close(figure)

    metrics = {
        "episodes": episodes,
        "base_seed": base_seed,
        "seed_schedule": "base_seed + episode_index",
        "win_rate": float(sum(1 for value in wins if value) / max(1, len(wins))),
        "mean_return": float(sum(returns) / max(1, len(returns))),
        "mean_steps": float(sum(steps) / max(1, len(steps))),
        "mean_score": float(sum(scores) / max(1, len(scores))),
        "returns": returns,
        "wins": wins,
        "steps": steps,
        "scores": scores,
        "model_path": str(Path(model_path)),
        "model_summary": model_data.get("summary", {}),
        "unseen_state_fallbacks": int(policy.unseen_states),
        "eval_plot": str(eval_plot_path),
        "gif_path": str(gif_path) if gif_path is not None else None,
        "render_mode": render_mode_value,
        "best_episode_index": int(best_episode_index) if best_episode_index >= 0 else None,
        "best_episode_seed": int(base_seed + best_episode_index) if best_episode_index >= 0 else None,
        "best_episode_win": bool(best_win) if best_episode_index >= 0 else None,
        "best_episode_score": float(best_score) if best_episode_index >= 0 else None,
        "best_episode_return": float(best_return) if best_episode_index >= 0 else None,
        "best_episode_steps": int(best_steps) if best_episode_index >= 0 else None,
    }
    metrics_path = output_dir_path / "eval_bitmask_vi_metrics.json"
    save_json(metrics, metrics_path)
    save_json(metrics, important_dir / "eval_bitmask_vi_metrics.json")
    return {
        "metrics": metrics,
        "metrics_path": str(metrics_path),
        "important_metrics_path": str(important_dir / "eval_bitmask_vi_metrics.json"),
        "gif_path": str(gif_path) if gif_path is not None else None,
        "eval_plot_path": str(eval_plot_path),
    }
