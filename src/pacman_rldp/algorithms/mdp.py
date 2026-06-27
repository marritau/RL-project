"""Abstract interfaces for tabular dynamic programming models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Hashable


@dataclass(frozen=True)
class TransitionOutcome:
    """One possible result of taking an action in an MDP model."""

    probability: float
    next_state: Hashable
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any]


class MDPModel(ABC):
    """Abstract model contract used by DP planners and experiments."""

    @abstractmethod
    def encode_state(self, state: Any) -> Hashable:
        """Convert a runtime state object to a hashable key."""

    @abstractmethod
    def available_actions(self, state: Any) -> list[int]:
        """Return legal action identifiers from the provided state."""

    @abstractmethod
    def transition_outcomes(self, state: Any, action: int) -> list[TransitionOutcome]:
        """Enumerate probabilistic transition outcomes for one action."""

    @abstractmethod
    def reward(self, state: Any, action: int, next_state: Any) -> float:
        """Compute immediate reward for a state-action-next_state triple."""

    @abstractmethod
    def is_terminal(self, state: Any) -> bool:
        """Report whether the provided state is terminal."""
