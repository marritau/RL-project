"""Pacman-specific MDP adapter for one-step transition extraction."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Hashable

from ..env import PacmanEnv
from ..third_party.bk import pacman as runtime_pacman
from .mdp import MDPModel, TransitionOutcome


class PacmanMDPAdapter(MDPModel):
    """Model adapter that exposes exact one-step transitions for random ghosts."""

    def __init__(self, env: PacmanEnv) -> None:
        """Bind adapter to a configured environment instance."""
        self.env = env

    def encode_state(self, state: runtime_pacman.GameState) -> Hashable:
        """Encode runtime state to a deterministic, hashable tuple key."""
        pacman_pos = state.getPacmanPosition()
        pacman_key = (round(float(pacman_pos[0]), 3), round(float(pacman_pos[1]), 3))

        ghost_key = []
        for ghost_state in state.getGhostStates():
            ghost_pos = ghost_state.getPosition()
            if ghost_pos is None:
                ghost_key.append((-1.0, -1.0, int(ghost_state.scaredTimer)))
                continue
            ghost_key.append(
                (
                    round(float(ghost_pos[0]), 3),
                    round(float(ghost_pos[1]), 3),
                    int(ghost_state.scaredTimer),
                )
            )

        food_key = tuple(int(cell) for column in state.getFood().data for cell in column)
        capsules_key = tuple(sorted((int(x), int(y)) for x, y in state.getCapsules()))
        return (
            pacman_key,
            tuple(ghost_key),
            food_key,
            capsules_key,
            bool(state.isWin()),
            bool(state.isLose()),
        )

    def available_actions(self, state: runtime_pacman.GameState) -> list[int]:
        """Return legal action ids for Pacman from a concrete runtime state."""
        return self.env.legal_action_ids(state)

    def transition_outcomes(
        self,
        state: runtime_pacman.GameState,
        action: int,
    ) -> list[TransitionOutcome]:
        """Enumerate all stochastic outcomes under uniform random ghost actions."""
        if self.is_terminal(state):
            return []

        direction = self.env.action_id_to_direction(action)
        legal_directions = state.getLegalActions(0)
        if direction not in legal_directions:
            if self.env.config.invalid_action_mode == "raise":
                raise ValueError(f"Illegal action '{direction}'. Legal actions: {legal_directions}")
            stop_direction = self.env.action_id_to_direction(4)
            direction = stop_direction if stop_direction in legal_directions else legal_directions[0]

        base_state = self.env.clone_state(state)
        after_pacman = base_state.generateSuccessor(0, direction)

        leaf_states: list[tuple[float, runtime_pacman.GameState]] = []

        def recurse_ghosts(
            current_state: runtime_pacman.GameState,
            ghost_index: int,
            probability: float,
        ) -> None:
            """Enumerate sequential ghost responses recursively."""
            if current_state.isWin() or current_state.isLose() or ghost_index >= current_state.getNumAgents():
                leaf_states.append((probability, current_state))
                return

            legal_actions = current_state.getLegalActions(ghost_index)
            if not legal_actions:
                recurse_ghosts(current_state, ghost_index + 1, probability)
                return

            branch_prob = probability / len(legal_actions)
            for ghost_action in legal_actions:
                ghost_next_state = current_state.generateSuccessor(ghost_index, ghost_action)
                recurse_ghosts(ghost_next_state, ghost_index + 1, branch_prob)

        recurse_ghosts(after_pacman, 1, 1.0)

        aggregated: dict[Hashable, dict[str, Any]] = defaultdict(
            lambda: {
                "probability": 0.0,
                "weighted_reward": 0.0,
                "terminated": False,
                "count": 0,
            }
        )

        for probability, next_state in leaf_states:
            next_key = self.encode_state(next_state)
            reward_value = self.reward(state, action, next_state)
            slot = aggregated[next_key]
            slot["probability"] += probability
            slot["weighted_reward"] += probability * reward_value
            slot["terminated"] = slot["terminated"] or bool(next_state.isWin() or next_state.isLose())
            slot["count"] += 1

        outcomes: list[TransitionOutcome] = []
        for next_key, summary in aggregated.items():
            probability = float(summary["probability"])
            mean_reward = float(summary["weighted_reward"] / probability)
            outcomes.append(
                TransitionOutcome(
                    probability=probability,
                    next_state=next_key,
                    reward=mean_reward,
                    terminated=bool(summary["terminated"]),
                    truncated=False,
                    info={"aggregated_paths": int(summary["count"])},
                )
            )
        return outcomes

    def reward(
        self,
        state: runtime_pacman.GameState,
        action: int,
        next_state: runtime_pacman.GameState,
    ) -> float:
        """Compute reward using the same configured mapping as PacmanEnv."""
        del action
        reward_value, _ = self.env.compute_reward_from_transition(
            state_before=state,
            state_after=next_state,
            invalid_action=False,
        )
        return reward_value

    def is_terminal(self, state: runtime_pacman.GameState) -> bool:
        """Return whether state is terminal in the Berkeley runtime."""
        return bool(state.isWin() or state.isLose())
