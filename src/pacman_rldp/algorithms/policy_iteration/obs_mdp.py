"""Empirical MDP model over aggregated observations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, DefaultDict, Hashable
from collections import defaultdict

from ..mdp import TransitionOutcome


ObsKey = Hashable


@dataclass
class TransitionStats:
    """Accumulator for one (s, a, s') transition bucket."""

    count: int = 0
    reward_sum: float = 0.0
    terminated_count: int = 0
    truncated_count: int = 0

    def update(self, reward: float, terminated: bool, truncated: bool) -> None:
        self.count += 1
        self.reward_sum += float(reward)
        if terminated:
            self.terminated_count += 1
        if truncated:
            self.truncated_count += 1


class ObsMDPModel:
    """Empirical transition model using observation keys as states."""

    def __init__(self) -> None:
        self._data: DefaultDict[tuple[ObsKey, int], dict[ObsKey, TransitionStats]] = defaultdict(dict)
        self._actions_by_state: DefaultDict[ObsKey, set[int]] = defaultdict(set)

    def update(
        self,
        state: ObsKey,
        action: int,
        reward: float,
        next_state: ObsKey,
        terminated: bool,
        truncated: bool,
    ) -> None:
        """Record one observed transition."""
        key = (state, int(action))
        self._actions_by_state[state].add(int(action))
        bucket = self._data[key].get(next_state)
        if bucket is None:
            bucket = TransitionStats()
            self._data[key][next_state] = bucket
        bucket.update(reward=reward, terminated=terminated, truncated=truncated)

    def states(self) -> list[ObsKey]:
        """Return list of known states."""
        return list(self._actions_by_state.keys())

    def actions_for_state(self, state: ObsKey) -> list[int]:
        """Return list of observed actions for a state."""
        return sorted(self._actions_by_state.get(state, set()))

    def get_outcomes(self, state: ObsKey, action: int) -> list[TransitionOutcome]:
        """Return empirical TransitionOutcome list for (state, action)."""
        key = (state, int(action))
        transitions = self._data.get(key)
        if not transitions:
            return []

        total = sum(stats.count for stats in transitions.values())
        if total <= 0:
            return []

        outcomes: list[TransitionOutcome] = []
        for next_state, stats in transitions.items():
            probability = stats.count / total
            reward = stats.reward_sum / max(1, stats.count)
            terminated_fraction = stats.terminated_count / max(1, stats.count)
            truncated_fraction = stats.truncated_count / max(1, stats.count)
            outcomes.append(
                TransitionOutcome(
                    probability=float(probability),
                    next_state=next_state,
                    reward=float(reward),
                    terminated=terminated_fraction >= 1.0,
                    truncated=truncated_fraction >= 1.0,
                    info={
                        "terminated_fraction": float(terminated_fraction),
                        "truncated_fraction": float(truncated_fraction),
                        "count": int(stats.count),
                    },
                )
            )
        return outcomes

    def transition_count(self) -> int:
        """Total number of observed transitions."""
        total = 0
        for transitions in self._data.values():
            for stats in transitions.values():
                total += stats.count
        return total

    def state_action_count(self) -> int:
        """Number of observed (state, action) pairs."""
        return len(self._data)
