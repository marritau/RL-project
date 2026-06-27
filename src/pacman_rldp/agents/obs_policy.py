"""Policy wrapper that acts on aggregated observation keys."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from ..algorithms.food_bitmask_value_iteration import FoodBitmaskHelper
from ..algorithms.policy_iteration.obs_encoding import encode_observation


class ObsPolicy:
    """Policy that selects actions from a precomputed table over observation keys.

    Unknown aggregated states are common in Pacman because the empirical MDP is
    built from sampled rollouts.  Falling back to pure random actions makes PI
    look much worse than the learned table deserves, so unknown food-bitmask
    states use the same nearest-food/avoid-ghost heuristic used by the VI helper.
    """

    def __init__(
        self,
        policy_table: dict[Any, int],
        *,
        seed: int | None = None,
        drop_keys: list[str] | None = None,
        float_round: int = 3,
        encoder: Callable[..., Any] | None = None,
        non_wall_coords: list[tuple[int, int]] | tuple[tuple[int, int], ...] | None = None,
        fallback_danger_distance: int = 2,
    ) -> None:
        self.policy_table = policy_table
        self.drop_keys = drop_keys or []
        self.float_round = float_round
        self.encoder = encoder or encode_observation
        self._rng = np.random.default_rng(seed)
        self.fallback_danger_distance = int(fallback_danger_distance)
        self.fallback_count = 0
        self._fallback_helper = FoodBitmaskHelper(non_wall_coords or [])

    def select_action(self, observation: dict[str, Any], info: dict[str, Any]) -> int:
        """Select action from policy table, with heuristic fallback for unseen states."""
        key = self.encoder(observation, drop_keys=self.drop_keys, float_round=self.float_round)
        action = self.policy_table.get(key)
        legal_actions = [int(a) for a in info.get("legal_action_ids", [])] if isinstance(info, dict) else []
        if action is not None and (not legal_actions or int(action) in legal_actions):
            return int(action)
        self.fallback_count += 1
        return self._fallback_action(observation, legal_actions)

    def _fallback_action(self, observation: dict[str, Any], legal_actions: list[int]) -> int:
        if not legal_actions:
            return 4
        if "food_bitmask" in observation:
            return int(
                self._fallback_helper.heuristic_action(
                    observation,
                    legal_actions,
                    rng=self._rng,
                    epsilon_random=0.0,
                    danger_distance=self.fallback_danger_distance,
                )
            )
        non_stop = [action for action in legal_actions if action != 4]
        if non_stop:
            return int(self._rng.choice(non_stop))
        return int(self._rng.choice(legal_actions))
