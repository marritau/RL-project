"""Unit tests for heuristic baseline Pacman policy."""

from __future__ import annotations

import numpy as np

from pacman_rldp.agents import BaselineNearestFoodAvoidGhostPolicy


def _build_observation(
    *,
    pacman: tuple[int, int],
    ghosts: list[tuple[int, int]],
    food_cells: list[tuple[int, int]],
    width: int = 5,
    height: int = 5,
) -> dict[str, np.ndarray]:
    """Build minimal raw observation dictionary required by baseline policy."""
    walls = np.zeros((width, height), dtype=np.int8)
    food = np.zeros((width, height), dtype=np.int8)
    for food_x, food_y in food_cells:
        food[food_x, food_y] = 1
    ghost_positions = np.full((len(ghosts), 2), fill_value=-1.0, dtype=np.float32)
    ghost_present = np.zeros((len(ghosts),), dtype=np.int8)
    for ghost_idx, (ghost_x, ghost_y) in enumerate(ghosts):
        ghost_positions[ghost_idx] = np.array([float(ghost_x), float(ghost_y)], dtype=np.float32)
        ghost_present[ghost_idx] = 1
    return {
        "pacman_position": np.array([float(pacman[0]), float(pacman[1])], dtype=np.float32),
        "ghost_positions": ghost_positions,
        "ghost_present": ghost_present,
        "walls": walls,
        "food": food,
    }


def test_safe_mode_moves_towards_nearest_food() -> None:
    """Move north when the nearest food is above and no ghost is dangerously close."""
    policy = BaselineNearestFoodAvoidGhostPolicy()
    observation = _build_observation(
        pacman=(1, 1),
        ghosts=[(4, 4)],
        food_cells=[(1, 3)],
    )
    info = {"legal_action_ids": [0, 1, 2, 3, 4]}
    assert policy.select_action(observation, info) == 0


def test_escape_mode_moves_away_from_close_ghost() -> None:
    """Escape when ghost is within Manhattan distance <= 2."""
    policy = BaselineNearestFoodAvoidGhostPolicy()
    observation = _build_observation(
        pacman=(2, 2),
        ghosts=[(2, 3)],
        food_cells=[(4, 4)],
    )
    info = {"legal_action_ids": [0, 1, 2, 3, 4]}
    assert policy.select_action(observation, info) == 1


def test_escape_mode_tie_breaks_by_smallest_action_id() -> None:
    """Use deterministic smallest-id tie break for equally good escape actions."""
    policy = BaselineNearestFoodAvoidGhostPolicy()
    observation = _build_observation(
        pacman=(2, 2),
        ghosts=[(2, 3)],
        food_cells=[(0, 0)],
    )
    info = {"legal_action_ids": [2, 3, 4]}
    assert policy.select_action(observation, info) == 2


def test_stop_is_returned_when_only_stop_is_legal() -> None:
    """Return stop action when it is the only available legal move."""
    policy = BaselineNearestFoodAvoidGhostPolicy()
    observation = _build_observation(
        pacman=(2, 2),
        ghosts=[(4, 4)],
        food_cells=[],
    )
    info = {"legal_action_ids": [4]}
    assert policy.select_action(observation, info) == 4
