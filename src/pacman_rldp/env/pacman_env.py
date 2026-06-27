"""Gymnasium-style Pacman environment built on the runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
import random
from typing import Any

import gymnasium as gym
import numpy as np

from ..third_party.bk import game as runtime_game
from ..third_party.bk import graphicsDisplay
from ..third_party.bk import layout as runtime_layout
from ..third_party.bk import pacman as runtime_pacman
from .observations import (
    ObservationConfig,
    ObservationSpec,
    build_observation_context,
    get_observation_spec,
)


@dataclass(frozen=True)
class RewardConfig:
    """Event-to-reward mapping for environment transitions."""

    time_penalty: float = -1.0
    food: float = 10.0
    capsule: float = 0.0
    eat_ghost: float = 200.0
    win: float = 500.0
    lose: float = -500.0
    invalid_action: float = -5.0


@dataclass
class PacmanEnvConfig:
    """Configuration values that define environment dynamics and rendering."""

    layout_name: str = "smallClassic"
    num_ghosts: int = 2
    max_steps: int = 500
    seed: int = 42
    ghost_policy: str = "random"
    ghost_loop_matrix: list[list[int]] | None = None
    ghost_loop_direction: str = "clockwise"
    invalid_action_mode: str = "raise"
    render_mode: str | None = None
    zoom: float = 1.0
    frame_time: float = 0.1
    observation: ObservationConfig = field(default_factory=ObservationConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)


class PacmanEnv(gym.Env[dict[str, Any], int]):
    """Gymnasium wrapper around the refactored Pacman runtime."""

    metadata = {"render_modes": ["human", "ansi", None], "render_fps": 10}

    _ACTIONS: tuple[str, ...] = (
        runtime_game.Directions.NORTH,
        runtime_game.Directions.SOUTH,
        runtime_game.Directions.EAST,
        runtime_game.Directions.WEST,
        runtime_game.Directions.STOP,
    )

    def __init__(
        self,
        config: PacmanEnvConfig | None = None,
        render_mode: str | None = None,
        observation_spec: ObservationSpec | None = None,
    ) -> None:
        """Initialize the environment with validated config and spaces."""
        super().__init__()
        self.config = config or PacmanEnvConfig()
        self.render_mode = render_mode if render_mode is not None else self.config.render_mode

        self._layout = runtime_layout.getLayout(self.config.layout_name)
        if self._layout is None:
            raise ValueError(f"Unknown layout '{self.config.layout_name}'.")

        self._ghost_count = min(self.config.num_ghosts, self._layout.getNumGhosts())
        self.action_space = gym.spaces.Discrete(len(self._ACTIONS))
        self._observation_context = build_observation_context(
            layout=self._layout,
            ghost_count=self._ghost_count,
            max_steps=self.config.max_steps,
            config=self.config.observation,
        )
        self._observation_spec = (
            observation_spec
            if observation_spec is not None
            else get_observation_spec(self.config.observation.name)
        )
        self.observation_space = self._observation_spec.build_space(self._observation_context)

        self._seed_value = self.config.seed
        self._rng = np.random.default_rng(self._seed_value)
        self._state: runtime_pacman.GameState | None = None
        self._step_count = 0
        self._terminated = False
        self._truncated = False
        self._display: Any | None = None
        self._display_initialized = False
        self._ghost_loop_cycle: list[tuple[int, int]] = []
        self._ghost_loop_index_by_agent: dict[int, int] = {}
        self._ghost_loop_coord_to_index: dict[tuple[int, int], int] = {}
        self.seed(self._seed_value)

    def seed(self, seed: int | None = None) -> None:
        """Set deterministic random generators for environment and ghost sampling."""
        if seed is None:
            seed = self._seed_value
        self._seed_value = int(seed)
        self._rng = np.random.default_rng(self._seed_value)
        random.seed(self._seed_value)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Reset the episode and return initial observation and metadata."""
        del options
        if seed is not None:
            self.seed(seed)

        state = runtime_pacman.GameState()
        state.initialize(self._layout, numGhostAgents=self._ghost_count)

        self._state = state
        self._step_count = 0
        self._terminated = False
        self._truncated = False
        self._initialize_ghost_policy_state(state)

        if self.render_mode == "human":
            self._reset_human_display()
            self.render()

        observation = self._observation_spec.build_observation(
            self._observation_context,
            self._state,
            self._step_count,
        )
        info = self._build_info(self._state)
        return observation, info

    def step(self, action: int) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """Advance the environment by one Pacman action and one ghost-response phase."""
        if self._state is None:
            raise RuntimeError("Call reset() before step().")
        if self._terminated or self._truncated:
            raise RuntimeError("Episode is finished. Call reset() before stepping again.")
        if not self.action_space.contains(action):
            raise ValueError(f"Action id {action} is out of bounds.")

        requested_direction = self._ACTIONS[action]
        legal_actions = self._state.getLegalActions(0)
        invalid_action = requested_direction not in legal_actions

        selected_direction = requested_direction
        if invalid_action:
            if self.config.invalid_action_mode == "raise":
                raise ValueError(
                    f"Illegal action '{requested_direction}' for current state. Legal: {legal_actions}"
                )
            stop_direction = runtime_game.Directions.STOP
            selected_direction = stop_direction if stop_direction in legal_actions else legal_actions[0]

        state_before = self.clone_state(self._state)
        transition_states: list[runtime_pacman.GameState] = []
        next_state = self._state.generateSuccessor(0, selected_direction)
        transition_states.append(next_state)

        if not (next_state.isWin() or next_state.isLose()):
            for ghost_index in range(1, next_state.getNumAgents()):
                if next_state.isWin() or next_state.isLose():
                    break
                if self.config.ghost_policy == "markovian":
                    ghost_legal_actions = self._markovian_legal_actions(next_state, ghost_index)
                else:
                    ghost_legal_actions = next_state.getLegalActions(ghost_index)
                if not ghost_legal_actions:
                    continue
                ghost_action = self._sample_ghost_action(
                    state=next_state,
                    ghost_index=ghost_index,
                    legal_actions=ghost_legal_actions,
                )
                if self.config.ghost_policy == "markovian":
                    self._prepare_markovian_state_for_action(next_state, ghost_index)
                next_state = next_state.generateSuccessor(ghost_index, ghost_action)
                self._sync_loop_index_from_runtime_position(next_state, ghost_index)
                transition_states.append(next_state)

        reward, events = self.compute_reward_from_transition(
            state_before=state_before,
            state_after=next_state,
            invalid_action=invalid_action,
        )

        self._state = next_state
        self._step_count += 1
        self._terminated = bool(next_state.isWin() or next_state.isLose())
        self._truncated = bool((not self._terminated) and (self._step_count >= self.config.max_steps))

        if self.render_mode == "human":
            for transition_state in transition_states:
                self._render_runtime_state(transition_state)

        observation = self._observation_spec.build_observation(
            self._observation_context,
            self._state,
            self._step_count,
        )
        info = self._build_info(self._state)
        info["events"] = events
        info["selected_direction"] = selected_direction
        info["requested_direction"] = requested_direction
        info["invalid_action"] = invalid_action

        return observation, reward, self._terminated, self._truncated, info

    def render(self) -> str | None:
        """Render state using human (Tk) or ANSI text mode."""
        if self._state is None:
            return None
        if self.render_mode == "ansi":
            return str(self._state)
        if self.render_mode == "human":
            self._render_runtime_state(self._state)
            return None
        return None

    def close(self) -> None:
        """Release renderer resources if the display was initialized."""
        if self._display is not None:
            self._display.finish()
        self._display = None
        self._display_initialized = False

    @property
    def runtime_state(self) -> runtime_pacman.GameState:
        """Return a deep-copied runtime state for external planners."""
        if self._state is None:
            raise RuntimeError("Environment is not initialized. Call reset() first.")
        return self.clone_state(self._state)

    def legal_action_ids(self, state: runtime_pacman.GameState | None = None) -> list[int]:
        """Return legal action ids for Pacman in the provided or current state."""
        current = state if state is not None else self._state
        if current is None:
            raise RuntimeError("Environment is not initialized. Call reset() first.")
        legal_directions = current.getLegalActions(0)
        return [idx for idx, direction in enumerate(self._ACTIONS) if direction in legal_directions]

    def action_id_to_direction(self, action_id: int) -> str:
        """Convert integer action id into direction string."""
        if not self.action_space.contains(action_id):
            raise ValueError(f"Action id {action_id} is out of bounds.")
        return self._ACTIONS[action_id]

    def compute_reward_from_transition(
        self,
        state_before: runtime_pacman.GameState,
        state_after: runtime_pacman.GameState,
        invalid_action: bool,
    ) -> tuple[float, dict[str, int]]:
        """Compute configured reward using event markers from the runtime state."""
        del state_before
        eaten_flags = state_after.data._eaten if state_after.data._eaten is not None else []
        ghost_eaten = int(sum(1 for eaten in eaten_flags[1:] if eaten))
        events = {
            "food_eaten": int(state_after.data._foodEaten is not None),
            "capsule_eaten": int(state_after.data._capsuleEaten is not None),
            "ghost_eaten": ghost_eaten,
            "win": int(state_after.isWin()),
            "lose": int(state_after.isLose()),
            "invalid_action": int(invalid_action),
        }

        reward_cfg = self.config.reward
        reward = float(reward_cfg.time_penalty)
        reward += float(events["food_eaten"]) * reward_cfg.food
        reward += float(events["capsule_eaten"]) * reward_cfg.capsule
        reward += float(events["ghost_eaten"]) * reward_cfg.eat_ghost
        reward += float(events["win"]) * reward_cfg.win
        reward += float(events["lose"]) * reward_cfg.lose
        reward += float(events["invalid_action"]) * reward_cfg.invalid_action
        return reward, events

    @staticmethod
    def clone_state(state: runtime_pacman.GameState) -> runtime_pacman.GameState:
        """Create a deep-copied runtime game state."""
        return runtime_pacman.GameState(state)

    def _sample_ghost_action(
        self,
        state: runtime_pacman.GameState,
        ghost_index: int,
        legal_actions: list[str],
    ) -> str:
        """Sample a ghost action using configured policy."""
        if self.config.ghost_policy == "random":
            sampled_index = int(self._rng.integers(low=0, high=len(legal_actions)))
            return legal_actions[sampled_index]
        if self.config.ghost_policy == "markovian":
            del state, ghost_index
            return self._sample_markovian_action(legal_actions)
        if self.config.ghost_policy == "loop_path":
            return self._sample_loop_path_action(
                state=state,
                ghost_index=ghost_index,
                legal_actions=legal_actions,
            )
        raise ValueError(f"Unsupported ghost policy '{self.config.ghost_policy}'.")

    def _sample_markovian_action(self, legal_actions: list[str]) -> str:
        """Sample uniformly over legal neighbor transitions, excluding stationary moves."""
        move_actions = [
            direction
            for direction in legal_actions
            if direction != runtime_game.Directions.STOP
        ]
        if not move_actions:
            return runtime_game.Directions.STOP
        sampled_index = int(self._rng.integers(low=0, high=len(move_actions)))
        return move_actions[sampled_index]

    def _markovian_legal_actions(
        self,
        state: runtime_pacman.GameState,
        ghost_index: int,
    ) -> list[str]:
        """Return markovian legal moves based only on current position and walls."""
        configuration = state.getGhostState(ghost_index).configuration
        possible_actions = runtime_game.Actions.getPossibleActions(
            configuration,
            state.data.layout.walls,
        )
        return [
            action
            for action in possible_actions
            if action != runtime_game.Directions.STOP
        ]

    def _prepare_markovian_state_for_action(
        self,
        state: runtime_pacman.GameState,
        ghost_index: int,
    ) -> None:
        """Prepare ghost config so reverse move remains legal under runtime checks."""
        ghost_state = state.getGhostState(ghost_index)
        configuration = ghost_state.configuration
        if configuration.isInteger():
            configuration.direction = runtime_game.Directions.STOP

    def _initialize_ghost_policy_state(self, state: runtime_pacman.GameState) -> None:
        """Initialize policy-specific ghost state at episode reset."""
        self._ghost_loop_cycle = []
        self._ghost_loop_index_by_agent = {}
        self._ghost_loop_coord_to_index = {}
        if self.config.ghost_policy != "loop_path":
            return
        self._ghost_loop_cycle = self._build_ghost_loop_cycle()
        self._ghost_loop_coord_to_index = {
            coord: idx for idx, coord in enumerate(self._ghost_loop_cycle)
        }
        self._ghost_loop_index_by_agent = self._build_initial_loop_indices(state)

    def _build_ghost_loop_cycle(self) -> list[tuple[int, int]]:
        """Validate and convert configured loop matrix into ordered cycle coordinates."""
        loop_matrix = self.config.ghost_loop_matrix
        if loop_matrix is None:
            raise ValueError("ghost_loop_matrix must be provided when ghost_policy='loop_path'.")

        width = int(self._layout.width)
        height = int(self._layout.height)
        if len(loop_matrix) != height:
            raise ValueError(
                f"ghost_loop_matrix row count {len(loop_matrix)} does not match layout height {height}."
            )
        if any(len(row) != width for row in loop_matrix):
            raise ValueError("ghost_loop_matrix must have consistent row width equal to layout width.")

        path_cells: set[tuple[int, int]] = set()
        for row_idx, row in enumerate(loop_matrix):
            for col_idx, raw_value in enumerate(row):
                if raw_value not in (0, 1):
                    raise ValueError("ghost_loop_matrix values must be 0 or 1.")
                if raw_value == 0:
                    continue
                y_coord = height - 1 - row_idx
                x_coord = col_idx
                if self._layout.walls[x_coord][y_coord]:
                    raise ValueError(
                        f"ghost_loop_matrix marks wall cell as path: (x={x_coord}, y={y_coord})."
                    )
                path_cells.add((x_coord, y_coord))

        if not path_cells:
            raise ValueError("ghost_loop_matrix contains no path cells.")

        component = self._collect_component(path_cells, next(iter(path_cells)))
        if component != path_cells:
            raise ValueError("ghost_loop_matrix must form one connected path component.")

        for cell in path_cells:
            if len(self._path_neighbors(cell, path_cells)) != 2:
                raise ValueError(
                    "ghost_loop_matrix must define a closed simple cycle "
                    "(each path node degree must be exactly 2)."
                )

        anchor = min(path_cells, key=lambda coord: (-coord[1], coord[0]))
        neighbors = self._path_neighbors(anchor, path_cells)
        if len(neighbors) != 2:
            raise ValueError("Loop anchor must have degree 2.")

        first_cycle = self._walk_cycle(anchor, neighbors[0], path_cells)
        second_cycle = self._walk_cycle(anchor, neighbors[1], path_cells)

        if self.config.ghost_loop_direction not in {"clockwise", "anticlockwise"}:
            raise ValueError(
                "ghost_loop_direction must be one of {'clockwise', 'anticlockwise'}."
            )

        return self._select_oriented_cycle(
            cycle_a=first_cycle,
            cycle_b=second_cycle,
            direction=self.config.ghost_loop_direction,
        )

    def _build_initial_loop_indices(self, state: runtime_pacman.GameState) -> dict[int, int]:
        """Build initial per-ghost loop index mapping with even spacing."""
        if not self._ghost_loop_cycle:
            return {}
        cycle_length = len(self._ghost_loop_cycle)
        spacing = max(1, cycle_length // max(1, self._ghost_count))

        index_by_agent: dict[int, int] = {}
        ghost_states = state.getGhostStates()[: self._ghost_count]
        anchor_index = 0
        if ghost_states:
            first_position = ghost_states[0].getPosition()
            if first_position is not None:
                anchor_index = self._nearest_cycle_index(first_position)
        for ghost_offset, ghost_state in enumerate(ghost_states, start=1):
            del ghost_state
            index_by_agent[ghost_offset] = int((anchor_index + (ghost_offset - 1) * spacing) % cycle_length)
        return index_by_agent

    def _sample_loop_path_action(
        self,
        state: runtime_pacman.GameState,
        ghost_index: int,
        legal_actions: list[str],
    ) -> str:
        """Choose deterministic action that advances ghost along configured loop."""
        if not self._ghost_loop_cycle:
            raise ValueError("loop_path policy is active but no loop cycle is initialized.")

        current_position = state.getGhostPosition(ghost_index)
        current_index = self._ghost_loop_index_by_agent.get(ghost_index)
        if current_index is None:
            current_index = self._nearest_cycle_index(current_position)

        cycle_length = len(self._ghost_loop_cycle)
        if current_position in self._ghost_loop_coord_to_index:
            self._ghost_loop_index_by_agent[ghost_index] = self._ghost_loop_coord_to_index[current_position]
            target_index = int((self._ghost_loop_index_by_agent[ghost_index] + 1) % cycle_length)
        else:
            target_index = int(current_index)
        target_coord = self._ghost_loop_cycle[target_index]

        intended_direction = self._direction_towards_target(current_position, target_coord)
        if intended_direction in legal_actions:
            return intended_direction

        best_action = min(
            legal_actions,
            key=lambda direction: (
                self._distance_after_action(current_position, direction, target_coord),
                direction,
            ),
        )
        return best_action

    def _sync_loop_index_from_runtime_position(
        self,
        state: runtime_pacman.GameState,
        ghost_index: int,
    ) -> None:
        """Synchronize loop index from runtime ghost position after movement."""
        if self.config.ghost_policy != "loop_path" or not self._ghost_loop_cycle:
            return
        ghost_position = state.getGhostPosition(ghost_index)
        if ghost_position in self._ghost_loop_coord_to_index:
            self._ghost_loop_index_by_agent[ghost_index] = self._ghost_loop_coord_to_index[ghost_position]
            return
        self._ghost_loop_index_by_agent[ghost_index] = self._nearest_cycle_index(ghost_position)

    def _collect_component(
        self,
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
            for neighbor in self._path_neighbors(cell, path_cells):
                if neighbor not in visited:
                    stack.append(neighbor)
        return visited

    @staticmethod
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

    def _walk_cycle(
        self,
        anchor: tuple[int, int],
        first_neighbor: tuple[int, int],
        path_cells: set[tuple[int, int]],
    ) -> list[tuple[int, int]]:
        """Walk ordered cycle from anchor using chosen first neighbor."""
        ordered = [anchor]
        previous = anchor
        current = first_neighbor
        while current != anchor:
            ordered.append(current)
            neighbors = self._path_neighbors(current, path_cells)
            next_candidates = [neighbor for neighbor in neighbors if neighbor != previous]
            if len(next_candidates) != 1:
                raise ValueError("Failed to build unique loop ordering from matrix path.")
            previous, current = current, next_candidates[0]
            if len(ordered) > len(path_cells) + 1:
                raise ValueError("Loop ordering exceeded expected path size.")
        if len(ordered) != len(path_cells):
            raise ValueError("Loop ordering does not cover all configured path cells.")
        return ordered

    @staticmethod
    def _cycle_signed_area(cycle: list[tuple[int, int]]) -> float:
        """Compute polygon signed area proxy for cycle orientation."""
        area = 0.0
        for idx, (x_coord, y_coord) in enumerate(cycle):
            next_x, next_y = cycle[(idx + 1) % len(cycle)]
            area += x_coord * next_y - next_x * y_coord
        return area / 2.0

    def _select_oriented_cycle(
        self,
        cycle_a: list[tuple[int, int]],
        cycle_b: list[tuple[int, int]],
        direction: str,
    ) -> list[tuple[int, int]]:
        """Choose ordered cycle for requested direction with stable anchor."""
        candidate = cycle_a
        if direction == "clockwise":
            if self._cycle_signed_area(candidate) > 0:
                candidate = cycle_b
        else:
            if self._cycle_signed_area(candidate) < 0:
                candidate = cycle_b
        anchor = candidate[0]
        anchor_index = candidate.index(anchor)
        return candidate[anchor_index:] + candidate[:anchor_index]

    def _nearest_cycle_index(self, position: tuple[float, float]) -> int:
        """Return nearest loop index to floating-point runtime position."""
        best_distance = float("inf")
        best_coord: tuple[int, int] | None = None
        for coord in self._ghost_loop_cycle:
            distance = abs(coord[0] - position[0]) + abs(coord[1] - position[1])
            if distance < best_distance or (distance == best_distance and (best_coord is None or coord < best_coord)):
                best_distance = distance
                best_coord = coord
        if best_coord is None:
            raise ValueError("Failed to locate nearest loop coordinate.")
        return self._ghost_loop_coord_to_index[best_coord]

    @staticmethod
    def _direction_towards_target(
        source: tuple[float, float],
        target: tuple[int, int],
    ) -> str:
        """Return primary direction from source to target in grid space."""
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

    @staticmethod
    def _distance_after_action(
        source: tuple[float, float],
        direction: str,
        target: tuple[int, int],
    ) -> float:
        """Compute Manhattan distance to target after taking one action step."""
        dx, dy = runtime_game.Actions.directionToVector(direction, 1.0)
        next_x = source[0] + dx
        next_y = source[1] + dy
        return abs(target[0] - next_x) + abs(target[1] - next_y)

    def _build_info(self, state: runtime_pacman.GameState) -> dict[str, Any]:
        """Build step metadata that is useful for algorithms and debugging."""
        legal_action_ids = self.legal_action_ids(state)
        return {
            "legal_action_ids": legal_action_ids,
            "legal_directions": [self._ACTIONS[idx] for idx in legal_action_ids],
            "score": float(state.getScore()),
            "is_win": bool(state.isWin()),
            "is_lose": bool(state.isLose()),
            "step_count": int(self._step_count),
            "seed": int(self._seed_value),
        }

    def _ensure_human_display(self) -> None:
        """Initialize graphical renderer object on demand."""
        if self._display is None:
            self._display = graphicsDisplay.PacmanGraphics(
                self.config.zoom,
                frameTime=self.config.frame_time,
            )
            self._display_initialized = False

    def _reset_human_display(self) -> None:
        """Reset graphical renderer state at episode boundaries."""
        if self._display is not None:
            self._display.finish()
        self._display = None
        self._display_initialized = False

    def _render_runtime_state(self, state: runtime_pacman.GameState) -> None:
        """Render one concrete runtime state in the human display."""
        self._ensure_human_display()
        if not self._display_initialized:
            self._display.initialize(state.data)
            self._display_initialized = True
        else:
            self._display.update(state.data)


def build_env_config(config_dict: dict[str, Any]) -> PacmanEnvConfig:
    """Build typed environment config from a raw dictionary."""
    reward_dict = config_dict.get("reward", {})
    reward = RewardConfig(
        time_penalty=float(reward_dict.get("time_penalty", -1.0)),
        food=float(reward_dict.get("food", 10.0)),
        capsule=float(reward_dict.get("capsule", 0.0)),
        eat_ghost=float(reward_dict.get("eat_ghost", 200.0)),
        win=float(reward_dict.get("win", 500.0)),
        lose=float(reward_dict.get("lose", -500.0)),
        invalid_action=float(reward_dict.get("invalid_action", -5.0)),
    )
    raw_loop_matrix = config_dict.get("ghost_loop_matrix")
    ghost_loop_matrix: list[list[int]] | None
    if raw_loop_matrix is None:
        ghost_loop_matrix = None
    else:
        if not isinstance(raw_loop_matrix, list):
            raise ValueError("ghost_loop_matrix must be a list of rows.")
        ghost_loop_matrix = []
        for row in raw_loop_matrix:
            if not isinstance(row, list):
                raise ValueError("ghost_loop_matrix rows must be lists.")
            ghost_loop_matrix.append([int(value) for value in row])
    observation_dict = config_dict.get("observation", {})
    if observation_dict is None:
        observation_dict = {}
    if not isinstance(observation_dict, dict):
        raise ValueError("observation must be a mapping when provided.")
    observation = ObservationConfig(
        name=str(observation_dict.get("name", "raw")),
        chunk_w=int(observation_dict.get("chunk_w", 4)),
        chunk_h=int(observation_dict.get("chunk_h", 2)),
        distance_bucket_size=int(observation_dict.get("distance_bucket_size", 2)),
    )
    return PacmanEnvConfig(
        layout_name=str(config_dict.get("layout_name", "smallClassic")),
        num_ghosts=int(config_dict.get("num_ghosts", 2)),
        max_steps=int(config_dict.get("max_steps", 500)),
        seed=int(config_dict.get("seed", 42)),
        ghost_policy=str(config_dict.get("ghost_policy", "random")),
        ghost_loop_matrix=ghost_loop_matrix,
        ghost_loop_direction=str(config_dict.get("ghost_loop_direction", "clockwise")),
        invalid_action_mode=str(config_dict.get("invalid_action_mode", "raise")),
        render_mode=config_dict.get("render_mode"),
        zoom=float(config_dict.get("zoom", 1.0)),
        frame_time=float(config_dict.get("frame_time", 0.1)),
        observation=observation,
        reward=reward,
    )
