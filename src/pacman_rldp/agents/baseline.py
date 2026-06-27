"""Deterministic baseline policy for Pacman evaluation."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np

from .policies import Policy


_MOVEMENT_ACTIONS: tuple[int, ...] = (0, 1, 2, 3)
_ACTION_TO_DELTA: dict[int, tuple[int, int]] = {
    0: (0, 1),   # North
    1: (0, -1),  # South
    2: (1, 0),   # East
    3: (-1, 0),  # West
    4: (0, 0),   # Stop
}


@dataclass
class BaselineNearestFoodAvoidGhostPolicy(Policy):
    """Policy: chase nearest food, but flee when a ghost is too close."""

    ghost_danger_distance: int = 1

    def select_action(self, observation: dict[str, np.ndarray], info: dict[str, Any]) -> int:
        """Pick one legal action based on danger-first heuristic logic."""
        self._validate_raw_observation(observation)
        legal_action_ids = sorted(int(action_id) for action_id in info.get("legal_action_ids", []))
        if not legal_action_ids:
            return 4

        pacman_pos = self._to_grid_position(observation["pacman_position"])
        walls = observation["walls"]
        food = observation["food"]
        ghost_positions = self._active_ghost_positions(observation)

        nearest_ghost_distance = self._nearest_ghost_distance(pacman_pos, ghost_positions)
        if nearest_ghost_distance is not None and nearest_ghost_distance <= self.ghost_danger_distance:
            return self._escape_action(
                pacman_pos=pacman_pos,
                legal_action_ids=legal_action_ids,
                ghost_positions=ghost_positions,
            )

        food_action = self._first_step_to_nearest_food(
            pacman_pos=pacman_pos,
            walls=walls,
            food=food,
            legal_action_ids=legal_action_ids,
        )
        if food_action is not None:
            return food_action

        non_stop_actions = [action for action in legal_action_ids if action != 4]
        if non_stop_actions:
            return non_stop_actions[0]
        return 4

    @staticmethod
    def _validate_raw_observation(observation: dict[str, np.ndarray]) -> None:
        """Ensure required raw observation keys are present."""
        required_keys = {
            "pacman_position",
            "ghost_positions",
            "ghost_present",
            "walls",
            "food",
        }
        missing = sorted(required_keys.difference(observation.keys()))
        if missing:
            raise ValueError(
                "BaselineNearestFoodAvoidGhostPolicy requires raw observation fields. "
                f"Missing keys: {missing}"
            )

    @staticmethod
    def _to_grid_position(position: np.ndarray) -> tuple[int, int]:
        """Convert float position array into integer grid coordinate."""
        return int(round(float(position[0]))), int(round(float(position[1])))

    @staticmethod
    def _active_ghost_positions(observation: dict[str, np.ndarray]) -> list[tuple[int, int]]:
        """Extract integer positions of currently present ghosts."""
        ghost_positions = observation["ghost_positions"]
        ghost_present = observation["ghost_present"]
        active_positions: list[tuple[int, int]] = []
        for ghost_idx in range(len(ghost_present)):
            if int(ghost_present[ghost_idx]) != 1:
                continue
            ghost_x = int(round(float(ghost_positions[ghost_idx][0])))
            ghost_y = int(round(float(ghost_positions[ghost_idx][1])))
            active_positions.append((ghost_x, ghost_y))
        return active_positions

    @staticmethod
    def _manhattan_distance(pos_a: tuple[int, int], pos_b: tuple[int, int]) -> int:
        """Compute Manhattan distance between two integer grid positions."""
        return abs(pos_a[0] - pos_b[0]) + abs(pos_a[1] - pos_b[1])

    def _nearest_ghost_distance(
        self,
        pacman_pos: tuple[int, int],
        ghost_positions: list[tuple[int, int]],
    ) -> int | None:
        """Return distance to nearest active ghost if any exists."""
        if not ghost_positions:
            return None
        return min(self._manhattan_distance(pacman_pos, ghost_pos) for ghost_pos in ghost_positions)

    def _escape_action(
        self,
        pacman_pos: tuple[int, int],
        legal_action_ids: list[int],
        ghost_positions: list[tuple[int, int]],
    ) -> int:
        """Select legal action that maximizes distance to nearest ghost."""
        candidate_actions = [action for action in legal_action_ids if action in _ACTION_TO_DELTA]
        best_score = -1
        best_action = 4
        for action_id in candidate_actions:
            next_pos = self._apply_action(pacman_pos, action_id)
            if ghost_positions:
                nearest_after_step = min(
                    self._manhattan_distance(next_pos, ghost_pos) for ghost_pos in ghost_positions
                )
            else:
                nearest_after_step = 0
            if nearest_after_step > best_score or (
                nearest_after_step == best_score and action_id < best_action
            ):
                best_score = nearest_after_step
                best_action = action_id
        return best_action

    def _first_step_to_nearest_food(
        self,
        pacman_pos: tuple[int, int],
        walls: np.ndarray,
        food: np.ndarray,
        legal_action_ids: list[int],
    ) -> int | None:
        """Run BFS to nearest food and return first step action id."""
        width, height = int(walls.shape[0]), int(walls.shape[1])
        legal_movement_actions = [action for action in legal_action_ids if action in _MOVEMENT_ACTIONS]

        queue: deque[tuple[tuple[int, int], int | None]] = deque([(pacman_pos, None)])
        visited: set[tuple[int, int]] = {pacman_pos}
        while queue:
            current_pos, first_action = queue.popleft()
            x_coord, y_coord = current_pos
            if 0 <= x_coord < width and 0 <= y_coord < height and int(food[x_coord][y_coord]) == 1:
                if first_action is not None and first_action in legal_movement_actions:
                    return first_action
                break

            for action_id in _MOVEMENT_ACTIONS:
                next_pos = self._apply_action(current_pos, action_id)
                next_x, next_y = next_pos
                if not (0 <= next_x < width and 0 <= next_y < height):
                    continue
                if int(walls[next_x][next_y]) == 1:
                    continue
                if next_pos in visited:
                    continue
                visited.add(next_pos)
                root_action = action_id if first_action is None else first_action
                queue.append((next_pos, root_action))
        return None

    @staticmethod
    def _apply_action(position: tuple[int, int], action_id: int) -> tuple[int, int]:
        """Apply one discrete action id to integer grid coordinate."""
        delta_x, delta_y = _ACTION_TO_DELTA[action_id]
        return position[0] + delta_x, position[1] + delta_y
