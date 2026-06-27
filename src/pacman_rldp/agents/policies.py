"""Baseline policy implementations for training, evaluation, and manual play."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np


class Policy(ABC):
    """Generic policy contract used by scripts and experiments."""

    @abstractmethod
    def select_action(self, observation: dict[str, np.ndarray], info: dict[str, Any]) -> int:
        """Select one action id from the current observation and metadata."""


@dataclass
class RandomPolicy(Policy):
    """Uniform random policy over legal actions."""

    seed: int = 0

    def __post_init__(self) -> None:
        """Initialize local RNG state."""
        self._rng = np.random.default_rng(self.seed)

    def select_action(self, observation: dict[str, np.ndarray], info: dict[str, Any]) -> int:
        """Sample uniformly from provided legal action ids."""
        del observation
        legal_action_ids = info.get("legal_action_ids", [])
        if not legal_action_ids:
            return 4
        index = int(self._rng.integers(low=0, high=len(legal_action_ids)))
        return int(legal_action_ids[index])


@dataclass
class KeyboardPolicy(Policy):
    """Terminal-input policy for manual control in ANSI mode."""

    north_key: str = "w"
    south_key: str = "s"
    east_key: str = "d"
    west_key: str = "a"
    stop_key: str = "q"

    def select_action(self, observation: dict[str, np.ndarray], info: dict[str, Any]) -> int:
        """Prompt user for a key and map it to an action id."""
        del observation
        legal_action_ids = set(int(action_id) for action_id in info.get("legal_action_ids", []))
        key_to_action = {
            self.north_key: 0,
            self.south_key: 1,
            self.east_key: 2,
            self.west_key: 3,
            self.stop_key: 4,
        }
        chosen = input("Action [w/a/s/d, q=stop]: ").strip().lower()
        action = key_to_action.get(chosen, 4)
        if action not in legal_action_ids and legal_action_ids:
            return int(sorted(legal_action_ids)[0])
        return int(action)
