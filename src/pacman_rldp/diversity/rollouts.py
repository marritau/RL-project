"""Rollout collection for trajectory-diversity evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable

import numpy as np

from pacman_rldp.agents.policies import Policy
from pacman_rldp.env import PacmanEnv


GridPosition = tuple[int, int]


@dataclass
class Trajectory:
    """One Pacman episode represented as an ordered trajectory.

    ``states`` are the Pacman grid positions, including the initial state and the
    post-step state after each action. We intentionally use Pacman's path through
    the maze as the low-dimensional state projection for TVS, matching the maze
    setting in the paper and keeping exact time-to-reach distances tractable.
    """

    states: list[GridPosition]
    actions: list[int]
    rewards: list[float]
    total_return: float
    score: float
    win: bool
    lose: bool
    truncated: bool
    seed: int

    @property
    def length(self) -> int:
        return len(self.actions)

    def to_json_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["states"] = [[int(x), int(y)] for x, y in self.states]
        payload["actions"] = [int(action) for action in self.actions]
        payload["rewards"] = [float(reward) for reward in self.rewards]
        return payload


def pacman_position_from_observation(observation: dict[str, np.ndarray]) -> GridPosition:
    """Extract integer Pacman grid coordinate from a raw observation."""
    if "pacman_position" not in observation:
        raise ValueError("TVS rollout collection requires raw observations with 'pacman_position'.")
    position = observation["pacman_position"]
    return int(round(float(position[0]))), int(round(float(position[1])))


def collect_trajectories(
    *,
    env: PacmanEnv,
    policy: Policy,
    episodes: int,
    base_seed: int,
) -> list[Trajectory]:
    """Collect deterministic-seed rollouts from a policy."""
    trajectories: list[Trajectory] = []
    for episode_idx in range(int(episodes)):
        seed = int(base_seed + episode_idx)
        observation, info = env.reset(seed=seed)
        states = [pacman_position_from_observation(observation)]
        actions: list[int] = []
        rewards: list[float] = []
        total_return = 0.0
        terminated = False
        truncated = False
        while True:
            action = int(policy.select_action(observation, info))
            observation, reward, terminated, truncated, info = env.step(action)
            actions.append(action)
            rewards.append(float(reward))
            total_return += float(reward)
            states.append(pacman_position_from_observation(observation))
            if terminated or truncated:
                break
        trajectories.append(
            Trajectory(
                states=states,
                actions=actions,
                rewards=rewards,
                total_return=float(total_return),
                score=float(info.get("score", 0.0)),
                win=bool(info.get("is_win", False)),
                lose=bool(info.get("is_lose", False)),
                truncated=bool(truncated and not terminated),
                seed=seed,
            )
        )
    return trajectories


def filter_trajectories(
    trajectories: Iterable[Trajectory],
    mode: str = "wins_or_all",
    top_fraction: float = 0.5,
) -> list[Trajectory]:
    """Select the trajectories used for diversity scoring.

    The paper focuses on *diverse high-quality trajectories*. For Pacman, a win is
    the cleanest quality marker. ``wins_or_all`` uses successful trajectories when
    any exist and falls back to all rollouts otherwise so the script remains useful
    for weak/random policies.
    """
    items = list(trajectories)
    if not items:
        return []

    normalized_mode = mode.lower().strip()
    if normalized_mode == "all":
        return items
    if normalized_mode in {"wins", "successful"}:
        return [trajectory for trajectory in items if trajectory.win]
    if normalized_mode in {"wins_or_all", "successful_or_all"}:
        winners = [trajectory for trajectory in items if trajectory.win]
        return winners if winners else items
    if normalized_mode in {"non_loss", "nonloss"}:
        return [trajectory for trajectory in items if not trajectory.lose]
    if normalized_mode in {"top_return", "top"}:
        if not 0.0 < top_fraction <= 1.0:
            raise ValueError("top_fraction must be in (0, 1].")
        ordered = sorted(items, key=lambda trajectory: trajectory.total_return, reverse=True)
        keep = max(1, int(round(len(ordered) * top_fraction)))
        return ordered[:keep]
    raise ValueError(
        "Unknown quality-filter mode. Use one of: all, wins, wins_or_all, non_loss, top_return."
    )
