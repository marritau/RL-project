"""Manual gameplay helpers using Berkeley keyboard control and displays."""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Any

from ..third_party.bk import ghostAgents
from ..third_party.bk import game as runtime_game
from ..third_party.bk import graphicsDisplay
from ..third_party.bk import keyboardAgents
from ..third_party.bk import layout as runtime_layout
from ..third_party.bk import pacman as runtime_pacman
from ..third_party.bk import textDisplay
from ..third_party.bk import util as runtime_util


class MarkovianGhost(ghostAgents.GhostAgent):
    """Ghost policy with uniform transitions from current state only."""

    def getDistribution(self, state: runtime_pacman.GameState) -> runtime_util.Counter:
        """Return uniform distribution over legal runtime moves for manual gameplay."""
        move_actions = [
            action
            for action in state.getLegalActions(self.index)
            if action != runtime_game.Directions.STOP
        ]

        distribution = runtime_util.Counter()
        if not move_actions:
            distribution[runtime_game.Directions.STOP] = 1.0
            return distribution

        probability = 1.0 / len(move_actions)
        for action in move_actions:
            distribution[action] = probability
        distribution.normalize()
        return distribution


@dataclass
class LoopPathManager:
    """Shared loop-path state across all loop-policy ghosts."""

    cycle: list[tuple[int, int]]
    ghost_count: int
    ghost_index_by_agent: dict[int, int]
    coord_to_index: dict[tuple[int, int], int]
    initialized: bool = False

    def ensure_initialized(self, state: runtime_pacman.GameState) -> None:
        """Initialize per-ghost loop offsets once from initial runtime positions."""
        if self.initialized:
            return
        cycle_length = len(self.cycle)
        spacing = max(1, cycle_length // max(1, self.ghost_count))
        anchor_index = 0
        first_ghost_pos = state.getGhostPosition(1) if self.ghost_count > 0 else None
        if first_ghost_pos is not None:
            anchor_index = self.nearest_cycle_index(first_ghost_pos)
        for ghost_agent_index in range(1, self.ghost_count + 1):
            self.ghost_index_by_agent[ghost_agent_index] = int(
                (anchor_index + (ghost_agent_index - 1) * spacing) % cycle_length
            )
        self.initialized = True

    def nearest_cycle_index(self, position: tuple[float, float]) -> int:
        """Return nearest loop index for a floating-point runtime coordinate."""
        best_distance = float("inf")
        best_coord: tuple[int, int] | None = None
        for coord in self.cycle:
            distance = abs(coord[0] - position[0]) + abs(coord[1] - position[1])
            if distance < best_distance or (
                distance == best_distance and (best_coord is None or coord < best_coord)
            ):
                best_distance = distance
                best_coord = coord
        if best_coord is None:
            raise ValueError("Failed to resolve nearest loop index.")
        return self.coord_to_index[best_coord]

    def sync_from_position(self, ghost_index: int, position: tuple[float, float]) -> None:
        """Update stored loop index from actual runtime position."""
        if position in self.coord_to_index:
            self.ghost_index_by_agent[ghost_index] = self.coord_to_index[position]
            return
        self.ghost_index_by_agent[ghost_index] = self.nearest_cycle_index(position)


class LoopPathGhost(ghostAgents.GhostAgent):
    """Deterministic ghost that follows one configured loop cycle."""

    def __init__(self, index: int, manager: LoopPathManager) -> None:
        """Bind one ghost index to a shared loop path manager."""
        super().__init__(index)
        self.manager = manager

    def getDistribution(self, state: runtime_pacman.GameState) -> runtime_util.Counter:
        """Return one-hot distribution for the next loop-following action."""
        self.manager.ensure_initialized(state)
        current_position = state.getGhostPosition(self.index)
        self.manager.sync_from_position(self.index, current_position)

        current_index = self.manager.ghost_index_by_agent[self.index]
        cycle_length = len(self.manager.cycle)
        target_index = int((current_index + 1) % cycle_length)
        target_coord = self.manager.cycle[target_index]

        legal_actions = state.getLegalActions(self.index)
        chosen_action = _choose_loop_action(
            current_position=current_position,
            target_coord=target_coord,
            legal_actions=legal_actions,
        )
        distribution = runtime_util.Counter()
        distribution[chosen_action] = 1.0
        return distribution


def _direction_towards_target(
    source: tuple[float, float],
    target: tuple[int, int],
) -> str:
    """Return primary grid direction from source to target."""
    delta_x = target[0] - source[0]
    delta_y = target[1] - source[1]
    if abs(delta_x) >= abs(delta_y):
        if delta_x > 0:
            return runtime_game.Directions.EAST
        if delta_x < 0:
            return runtime_game.Directions.WEST
    if delta_y > 0:
        return runtime_game.Directions.NORTH
    if delta_y < 0:
        return runtime_game.Directions.SOUTH
    return runtime_game.Directions.STOP


def _distance_after_action(
    source: tuple[float, float],
    action: str,
    target: tuple[int, int],
) -> float:
    """Compute Manhattan distance to target after a candidate action."""
    dx, dy = runtime_game.Actions.directionToVector(action, 1.0)
    next_x = source[0] + dx
    next_y = source[1] + dy
    return abs(target[0] - next_x) + abs(target[1] - next_y)


def _choose_loop_action(
    current_position: tuple[float, float],
    target_coord: tuple[int, int],
    legal_actions: list[str],
) -> str:
    """Select best legal action for loop progression with deterministic fallback."""
    intended = _direction_towards_target(current_position, target_coord)
    if intended in legal_actions:
        return intended
    if not legal_actions:
        return runtime_game.Directions.STOP
    return min(
        legal_actions,
        key=lambda action: (
            _distance_after_action(current_position, action, target_coord),
            action,
        ),
    )


def _path_neighbors(
    cell: tuple[int, int],
    path_cells: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    """Return 4-neighbor path cells for one coordinate."""
    x_coord, y_coord = cell
    candidates = [
        (x_coord + 1, y_coord),
        (x_coord - 1, y_coord),
        (x_coord, y_coord + 1),
        (x_coord, y_coord - 1),
    ]
    return [candidate for candidate in candidates if candidate in path_cells]


def _collect_component(
    path_cells: set[tuple[int, int]],
    start: tuple[int, int],
) -> set[tuple[int, int]]:
    """Collect connected component in 4-neighborhood for loop validation."""
    stack = [start]
    visited: set[tuple[int, int]] = set()
    while stack:
        cell = stack.pop()
        if cell in visited:
            continue
        visited.add(cell)
        for neighbor in _path_neighbors(cell, path_cells):
            if neighbor not in visited:
                stack.append(neighbor)
    return visited


def _walk_cycle(
    anchor: tuple[int, int],
    first_neighbor: tuple[int, int],
    path_cells: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    """Walk ordered loop cycle from anchor through one chosen neighbor."""
    ordered = [anchor]
    previous = anchor
    current = first_neighbor
    while current != anchor:
        ordered.append(current)
        neighbors = _path_neighbors(current, path_cells)
        next_candidates = [neighbor for neighbor in neighbors if neighbor != previous]
        if len(next_candidates) != 1:
            raise ValueError("Failed to build unique loop ordering from ghost_loop_matrix.")
        previous, current = current, next_candidates[0]
        if len(ordered) > len(path_cells) + 1:
            raise ValueError("Loop ordering exceeded expected path size.")
    if len(ordered) != len(path_cells):
        raise ValueError("Loop ordering does not cover all configured path cells.")
    return ordered


def _cycle_signed_area(cycle: list[tuple[int, int]]) -> float:
    """Compute signed area proxy to detect loop orientation."""
    area = 0.0
    for idx, (x_coord, y_coord) in enumerate(cycle):
        next_x, next_y = cycle[(idx + 1) % len(cycle)]
        area += x_coord * next_y - next_x * y_coord
    return area / 2.0


def _build_loop_cycle(
    chosen_layout: Any,
    ghost_loop_matrix: list[list[int]] | None,
    ghost_loop_direction: str,
) -> list[tuple[int, int]]:
    """Validate matrix and return ordered loop cycle coordinates."""
    if ghost_loop_matrix is None:
        raise ValueError("ghost_loop_matrix must be provided when ghost_policy='loop_path'.")
    width = int(chosen_layout.width)
    height = int(chosen_layout.height)
    if len(ghost_loop_matrix) != height:
        raise ValueError(
            f"ghost_loop_matrix row count {len(ghost_loop_matrix)} does not match layout height {height}."
        )
    if any(len(row) != width for row in ghost_loop_matrix):
        raise ValueError("ghost_loop_matrix width must match layout width.")

    path_cells: set[tuple[int, int]] = set()
    for row_idx, row in enumerate(ghost_loop_matrix):
        for col_idx, raw_value in enumerate(row):
            if raw_value not in (0, 1):
                raise ValueError("ghost_loop_matrix values must be 0 or 1.")
            if raw_value == 0:
                continue
            y_coord = height - 1 - row_idx
            x_coord = col_idx
            if chosen_layout.walls[x_coord][y_coord]:
                raise ValueError(f"ghost_loop_matrix marks wall cell (x={x_coord}, y={y_coord}).")
            path_cells.add((x_coord, y_coord))

    if not path_cells:
        raise ValueError("ghost_loop_matrix contains no path cells.")
    component = _collect_component(path_cells, next(iter(path_cells)))
    if component != path_cells:
        raise ValueError("ghost_loop_matrix must form one connected component.")
    for cell in path_cells:
        if len(_path_neighbors(cell, path_cells)) != 2:
            raise ValueError("ghost_loop_matrix must define a closed simple cycle.")

    anchor = min(path_cells, key=lambda coord: (-coord[1], coord[0]))
    neighbors = _path_neighbors(anchor, path_cells)
    first_cycle = _walk_cycle(anchor, neighbors[0], path_cells)
    second_cycle = _walk_cycle(anchor, neighbors[1], path_cells)
    if ghost_loop_direction not in {"clockwise", "anticlockwise"}:
        raise ValueError(
            "ghost_loop_direction must be one of {'clockwise', 'anticlockwise'}."
        )
    if ghost_loop_direction == "clockwise":
        return first_cycle if _cycle_signed_area(first_cycle) <= 0 else second_cycle
    return first_cycle if _cycle_signed_area(first_cycle) >= 0 else second_cycle


def build_manual_ghosts(
    ghost_policy: str,
    ghost_count: int,
    chosen_layout: Any,
    ghost_loop_matrix: list[list[int]] | None,
    ghost_loop_direction: str,
) -> list[ghostAgents.GhostAgent]:
    """Build manual-mode ghost agents according to configured ghost policy."""
    if ghost_policy == "random":
        return [ghostAgents.RandomGhost(index + 1) for index in range(ghost_count)]
    if ghost_policy == "markovian":
        return [MarkovianGhost(index + 1) for index in range(ghost_count)]
    if ghost_policy == "loop_path":
        cycle = _build_loop_cycle(
            chosen_layout=chosen_layout,
            ghost_loop_matrix=ghost_loop_matrix,
            ghost_loop_direction=ghost_loop_direction,
        )
        manager = LoopPathManager(
            cycle=cycle,
            ghost_count=ghost_count,
            ghost_index_by_agent={},
            coord_to_index={coord: idx for idx, coord in enumerate(cycle)},
        )
        return [LoopPathGhost(index + 1, manager) for index in range(ghost_count)]
    raise ValueError(f"Unsupported ghost policy '{ghost_policy}'.")


def run_keyboard_game(
    layout_name: str,
    num_ghosts: int,
    seed: int,
    render_mode: str,
    zoom: float,
    frame_time: float,
    ghost_policy: str,
    ghost_loop_matrix: list[list[int]] | None,
    ghost_loop_direction: str,
) -> float:
    """Run one interactive keyboard game and return the final score."""
    if render_mode != "human":
        raise ValueError("Keyboard-driven manual mode requires render_mode='human'.")

    random.seed(seed)
    chosen_layout = runtime_layout.getLayout(layout_name)
    if chosen_layout is None:
        raise ValueError(f"Unknown layout '{layout_name}'.")

    pacman_agent = keyboardAgents.KeyboardAgent(index=0)
    ghost_count = min(num_ghosts, chosen_layout.getNumGhosts())
    ghosts = build_manual_ghosts(
        ghost_policy=ghost_policy,
        ghost_count=ghost_count,
        chosen_layout=chosen_layout,
        ghost_loop_matrix=ghost_loop_matrix,
        ghost_loop_direction=ghost_loop_direction,
    )

    display = graphicsDisplay.PacmanGraphics(zoom=zoom, frameTime=frame_time)
    rules = runtime_pacman.ClassicGameRules()
    game = rules.newGame(
        layout=chosen_layout,
        pacmanAgent=pacman_agent,
        ghostAgents=ghosts,
        display=display,
        quiet=False,
        catchExceptions=False,
    )
    game.run()
    return float(game.state.getScore())


def build_text_display(frame_time: float) -> textDisplay.PacmanGraphics:
    """Construct Berkeley text display with explicit frame timing."""
    return textDisplay.PacmanGraphics(speed=frame_time)
