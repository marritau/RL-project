"""Policy wrappers useful for diversity evaluation and trained-policy rollouts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from pacman_rldp.agents.policies import Policy
from pacman_rldp.algorithms.food_bitmask_value_iteration import FoodBitmaskHelper
from pacman_rldp.algorithms.policy_iteration.obs_encoding import encode_observation
from pacman_rldp.utils import load_pickle


@dataclass
class NoisyPolicy(Policy):
    """Epsilon-random wrapper around any base policy.

    This is useful for controlled TVS experiments: the base policy remains the
    high-quality solver, while epsilon controls small amounts of stochasticity.
    The paper emphasizes that q=2 TVS discounts superficial action noise, so this
    wrapper is mainly a diagnostic for whether additional stochasticity creates
    genuinely different trajectories or just noisy variants.
    """

    base_policy: Policy
    epsilon: float = 0.0
    seed: int = 0

    def __post_init__(self) -> None:
        if not 0.0 <= float(self.epsilon) <= 1.0:
            raise ValueError("epsilon must be in [0, 1].")
        self._rng = np.random.default_rng(self.seed)

    def select_action(self, observation: dict[str, np.ndarray], info: dict[str, Any]) -> int:
        legal_action_ids = [int(action_id) for action_id in info.get("legal_action_ids", [])]
        if not legal_action_ids:
            return 4
        if self.epsilon > 0.0 and self._rng.random() < self.epsilon:
            return int(self._rng.choice(legal_action_ids))
        return int(self.base_policy.select_action(observation, info))


@dataclass
class FoodBitmaskHeuristicPolicy(Policy):
    """Nearest-food / avoid-ghost heuristic for food-bitmask observations.

    This is used only as a fallback for tabular policies when a state was not
    observed during training. It prevents an unseen state from degenerating into
    a random walk, while still letting learned table entries take priority.
    """

    non_wall_coords: list[tuple[int, int]] | tuple[tuple[int, int], ...]
    seed: int = 0
    danger_distance: int = 2

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(int(self.seed))
        self._helper = FoodBitmaskHelper(self.non_wall_coords)

    def select_action(self, observation: dict[str, Any], info: dict[str, Any]) -> int:
        legal_action_ids = [int(action_id) for action_id in info.get("legal_action_ids", [])]
        if not legal_action_ids:
            return 4
        if "food_bitmask" not in observation:
            non_stop = [action for action in legal_action_ids if action != 4]
            return int(non_stop[0] if non_stop else legal_action_ids[0])
        return int(
            self._helper.heuristic_action(
                observation,
                legal_action_ids,
                rng=self._rng,
                epsilon_random=0.0,
                danger_distance=int(self.danger_distance),
            )
        )


@dataclass
class TabularQGreedyPolicy(Policy):
    """Greedy policy wrapper for saved Q-learning/SARSA Q-tables."""

    q_table: dict[Any, np.ndarray]
    action_size: int
    drop_keys: list[str]
    float_round: int = 3
    seed: int = 0
    fallback_policy: Policy | None = None
    fallback_on_zero: bool = False

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(int(self.seed))

    @classmethod
    def from_model_path(
        cls,
        model_path: str | Path,
        *,
        action_size: int,
        drop_keys: list[str],
        float_round: int = 3,
        seed: int = 0,
        fallback_policy: Policy | None = None,
        fallback_on_zero: bool = False,
    ) -> "TabularQGreedyPolicy":
        q_table = load_pickle(Path(model_path))
        if not isinstance(q_table, dict):
            raise ValueError(f"Q-table model must be a dictionary: {model_path}")
        return cls(
            q_table=q_table,
            action_size=action_size,
            drop_keys=drop_keys,
            float_round=float_round,
            seed=seed,
            fallback_policy=fallback_policy,
            fallback_on_zero=fallback_on_zero,
        )

    def select_action(self, observation: dict[str, Any], info: dict[str, Any]) -> int:
        legal_actions = [int(action_id) for action_id in info.get("legal_action_ids", [])]
        if not legal_actions:
            return 4
        state_key = encode_observation(
            observation,
            drop_keys=self.drop_keys,
            float_round=int(self.float_round),
        )
        q_values = self.q_table.get(state_key)
        if q_values is None:
            return self._fallback(observation, info, legal_actions)

        q_array = np.asarray(q_values, dtype=np.float64)
        legal_values = q_array[legal_actions]
        if bool(self.fallback_on_zero) and np.allclose(legal_values, 0.0):
            return self._fallback(observation, info, legal_actions)

        masked = np.full(int(self.action_size), -np.inf, dtype=np.float64)
        for action in legal_actions:
            masked[int(action)] = q_array[int(action)]
        best = np.flatnonzero(masked == np.max(masked))
        return int(self._rng.choice(best))

    def _fallback(self, observation: dict[str, Any], info: dict[str, Any], legal_actions: list[int]) -> int:
        if self.fallback_policy is not None:
            action = int(self.fallback_policy.select_action(observation, info))
            if action in legal_actions:
                return action
        return int(self._rng.choice(legal_actions))
