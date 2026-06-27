"""Observation specifications and registry for Pacman environments."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Any, Callable

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from ..third_party.bk import game as runtime_game
from ..third_party.bk import pacman as runtime_pacman


@dataclass(frozen=True)
class ObservationConfig:
    """Configuration for selecting and parameterizing observation builders."""

    name: str = "raw"
    chunk_w: int = 4
    chunk_h: int = 2
    distance_bucket_size: int = 2


@dataclass(frozen=True)
class ObservationContext:
    """Static observation metadata derived from layout and config."""

    width: int
    height: int
    ghost_count: int
    max_steps: int
    chunk_w: int
    chunk_h: int
    distance_bucket_size: int
    non_wall_coords: tuple[tuple[int, int], ...]


BuildSpaceFn = Callable[[ObservationContext], gym.Space]
BuildObservationFn = Callable[[ObservationContext, runtime_pacman.GameState, int], dict[str, Any]]


@dataclass(frozen=True)
class ObservationSpec:
    """Bundle of observation-space and observation-building functions."""

    name: str
    build_space: BuildSpaceFn
    build_observation: BuildObservationFn


class ObservationName(str, Enum):
    """Registry keys for built-in observation specifications."""

    RAW = "raw"
    CHUNKED_FOOD = "chunked_food"
    FOOD_BITMASK = "food_bitmask"
    BITMASK_DISTANCE_BUCKETS = "bitmask_distance_buckets"
    SIMPLE_DISTANCE_BUCKETS = "simple_distance_buckets"


class NonNegativeIntSpace(gym.Space[int]):
    """Gymnasium space for non-negative Python integers."""

    def __init__(self, max_value: int | None = None) -> None:
        """Create integer space with optional upper bound."""
        super().__init__(shape=(), dtype=np.int64)
        self.max_value = max_value

    def sample(self, mask: Any = None) -> int:
        """Sample one value from the bounded integer domain."""
        del mask
        if self.max_value is None:
            return 0
        return int(self.np_random.integers(low=0, high=self.max_value + 1))

    def contains(self, x: Any) -> bool:
        """Check whether value belongs to integer domain."""
        if not isinstance(x, int):
            return False
        if x < 0:
            return False
        if self.max_value is not None and x > self.max_value:
            return False
        return True


def build_observation_context(
    layout: Any,
    ghost_count: int,
    max_steps: int,
    config: ObservationConfig,
) -> ObservationContext:
    """Build immutable observation context from layout and config."""
    width = int(layout.width)
    height = int(layout.height)
    non_wall_coords: list[tuple[int, int]] = []
    for y_coord in range(height - 1, -1, -1):
        for x_coord in range(width):
            if not layout.walls[x_coord][y_coord]:
                non_wall_coords.append((x_coord, y_coord))

    return ObservationContext(
        width=width,
        height=height,
        ghost_count=int(ghost_count),
        max_steps=int(max_steps),
        chunk_w=max(1, int(config.chunk_w)),
        chunk_h=max(1, int(config.chunk_h)),
        distance_bucket_size=max(1, int(config.distance_bucket_size)),
        non_wall_coords=tuple(non_wall_coords),
    )


def _build_raw_space(context: ObservationContext) -> gym.Space:
    """Build observation space for the raw observation format."""
    max_coord = float(max(context.width, context.height))
    return spaces.Dict(
        {
            "pacman_position": spaces.Box(
                low=np.array([-1.0, -1.0], dtype=np.float32),
                high=np.array([max_coord, max_coord], dtype=np.float32),
                dtype=np.float32,
            ),
            "ghost_positions": spaces.Box(
                low=-1.0,
                high=max_coord,
                shape=(context.ghost_count, 2),
                dtype=np.float32,
            ),
            "ghost_timers": spaces.Box(
                low=0,
                high=999,
                shape=(context.ghost_count,),
                dtype=np.int32,
            ),
            "ghost_present": spaces.MultiBinary(context.ghost_count),
            "walls": spaces.Box(
                low=0,
                high=1,
                shape=(context.width, context.height),
                dtype=np.int8,
            ),
            "food": spaces.Box(
                low=0,
                high=1,
                shape=(context.width, context.height),
                dtype=np.int8,
            ),
            "capsules": spaces.Box(
                low=0,
                high=1,
                shape=(context.width, context.height),
                dtype=np.int8,
            ),
            "score": spaces.Box(
                low=-1_000_000.0,
                high=1_000_000.0,
                shape=(1,),
                dtype=np.float32,
            ),
            "step_count": spaces.Box(
                low=0,
                high=context.max_steps,
                shape=(1,),
                dtype=np.int32,
            ),
        }
    )


def _build_chunked_food_space(context: ObservationContext) -> gym.Space:
    """Build observation space for chunked-food representation."""
    chunks_x = math.ceil(context.width / context.chunk_w)
    chunks_y = math.ceil(context.height / context.chunk_h)
    max_coord = float(max(context.width, context.height))
    return spaces.Dict(
        {
            "pacman_position": spaces.Box(
                low=np.array([-1.0, -1.0], dtype=np.float32),
                high=np.array([max_coord, max_coord], dtype=np.float32),
                dtype=np.float32,
            ),
            "ghost_positions": spaces.Box(
                low=-1.0,
                high=max_coord,
                shape=(context.ghost_count, 2),
                dtype=np.float32,
            ),
            "ghost_timers": spaces.Box(
                low=0,
                high=999,
                shape=(context.ghost_count,),
                dtype=np.int32,
            ),
            "ghost_present": spaces.MultiBinary(context.ghost_count),
            "chunk_food_presence": spaces.Box(
                low=0,
                high=1,
                shape=(chunks_x, chunks_y),
                dtype=np.int8,
            ),
            "pacman_chunk_coord": spaces.Box(
                low=np.array([0, 0], dtype=np.int32),
                high=np.array([max(0, chunks_x - 1), max(0, chunks_y - 1)], dtype=np.int32),
                dtype=np.int32,
            ),
            "pacman_chunk_food": spaces.Box(
                low=0,
                high=1,
                shape=(context.chunk_w, context.chunk_h),
                dtype=np.int8,
            ),
            "pacman_chunk_walls": spaces.Box(
                low=0,
                high=1,
                shape=(context.chunk_w, context.chunk_h),
                dtype=np.int8,
            ),
            "pacman_chunk_capsules": spaces.Box(
                low=0,
                high=1,
                shape=(context.chunk_w, context.chunk_h),
                dtype=np.int8,
            ),
            "score": spaces.Box(
                low=-1_000_000.0,
                high=1_000_000.0,
                shape=(1,),
                dtype=np.float32,
            ),
            "step_count": spaces.Box(
                low=0,
                high=context.max_steps,
                shape=(1,),
                dtype=np.int32,
            ),
        }
    )


def _build_food_bitmask_space(context: ObservationContext) -> gym.Space:
    """Build observation space for compact food bitmask representation."""
    max_coord = float(max(context.width, context.height))
    return spaces.Dict(
        {
            "pacman_position": spaces.Box(
                low=np.array([-1.0, -1.0], dtype=np.float32),
                high=np.array([max_coord, max_coord], dtype=np.float32),
                dtype=np.float32,
            ),
            "ghost_positions": spaces.Box(
                low=-1.0,
                high=max_coord,
                shape=(context.ghost_count, 2),
                dtype=np.float32,
            ),
            "ghost_timers": spaces.Box(
                low=0,
                high=999,
                shape=(context.ghost_count,),
                dtype=np.int32,
            ),
            "ghost_present": spaces.MultiBinary(context.ghost_count),
            "food_bitmask": NonNegativeIntSpace(),
            "score": spaces.Box(
                low=-1_000_000.0,
                high=1_000_000.0,
                shape=(1,),
                dtype=np.float32,
            ),
            "step_count": spaces.Box(
                low=0,
                high=context.max_steps,
                shape=(1,),
                dtype=np.int32,
            ),
        }
    )


def _build_bitmask_distance_buckets_space(context: ObservationContext) -> gym.Space:
    """Build observation space for bitmask plus distance-bucket features."""
    max_coord = float(max(context.width, context.height))
    max_distance = context.width + context.height
    max_bucket = math.ceil(max_distance / context.distance_bucket_size)
    return spaces.Dict(
        {
            "pacman_position": spaces.Box(
                low=np.array([-1.0, -1.0], dtype=np.float32),
                high=np.array([max_coord, max_coord], dtype=np.float32),
                dtype=np.float32,
            ),
            "ghost_positions": spaces.Box(
                low=-1.0,
                high=max_coord,
                shape=(context.ghost_count, 2),
                dtype=np.float32,
            ),
            "ghost_timers": spaces.Box(
                low=0,
                high=999,
                shape=(context.ghost_count,),
                dtype=np.int32,
            ),
            "ghost_present": spaces.MultiBinary(context.ghost_count),
            "food_bitmask": NonNegativeIntSpace(),
            "nearest_food_bucket": spaces.Box(
                low=-1,
                high=max_bucket,
                shape=(1,),
                dtype=np.int32,
            ),
            "nearest_ghost_bucket": spaces.Box(
                low=-1,
                high=max_bucket,
                shape=(1,),
                dtype=np.int32,
            ),
            "nearest_food_direction": spaces.Box(
                low=0,
                high=4,
                shape=(1,),
                dtype=np.int32,
            ),
            "nearest_ghost_direction": spaces.Box(
                low=0,
                high=4,
                shape=(1,),
                dtype=np.int32,
            ),
            "score": spaces.Box(
                low=-1_000_000.0,
                high=1_000_000.0,
                shape=(1,),
                dtype=np.float32,
            ),
            "step_count": spaces.Box(
                low=0,
                high=context.max_steps,
                shape=(1,),
                dtype=np.int32,
            ),
        }
    )


def _extract_common_agent_features(
    context: ObservationContext,
    state: runtime_pacman.GameState,
    step_count: int,
) -> dict[str, Any]:
    """Extract common position/timer/score fields shared by multiple specs."""
    pac_pos = state.getPacmanPosition()
    pacman_position = np.array([float(pac_pos[0]), float(pac_pos[1])], dtype=np.float32)

    ghost_positions = np.full((context.ghost_count, 2), fill_value=-1.0, dtype=np.float32)
    ghost_timers = np.zeros((context.ghost_count,), dtype=np.int32)
    ghost_present = np.zeros((context.ghost_count,), dtype=np.int8)

    for ghost_idx, ghost_state in enumerate(state.getGhostStates()[: context.ghost_count]):
        ghost_position = ghost_state.getPosition()
        if ghost_position is None:
            continue
        ghost_positions[ghost_idx] = np.array(
            [float(ghost_position[0]), float(ghost_position[1])],
            dtype=np.float32,
        )
        ghost_timers[ghost_idx] = int(ghost_state.scaredTimer)
        ghost_present[ghost_idx] = 1

    return {
        "pacman_position": pacman_position,
        "ghost_positions": ghost_positions,
        "ghost_timers": ghost_timers,
        "ghost_present": ghost_present,
        "score": np.array([float(state.getScore())], dtype=np.float32),
        "step_count": np.array([int(step_count)], dtype=np.int32),
    }


def _grid_to_binary_array(grid: runtime_game.Grid) -> np.ndarray:
    """Convert runtime boolean grid to int8 ndarray."""
    return np.asarray(grid.data, dtype=np.int8)


def _build_raw_observation(
    context: ObservationContext,
    state: runtime_pacman.GameState,
    step_count: int,
) -> dict[str, Any]:
    """Build raw observation equivalent to the legacy in-env implementation."""
    observation = _extract_common_agent_features(context=context, state=state, step_count=step_count)
    walls = _grid_to_binary_array(state.getWalls())
    food = _grid_to_binary_array(state.getFood())

    capsules = np.zeros((walls.shape[0], walls.shape[1]), dtype=np.int8)
    for capsule_x, capsule_y in state.getCapsules():
        capsules[int(capsule_x), int(capsule_y)] = 1

    observation.update(
        {
            "walls": walls,
            "food": food,
            "capsules": capsules,
        }
    )
    return observation


def _build_food_bitmask(context: ObservationContext, state: runtime_pacman.GameState) -> int:
    """Pack food occupancy over non-wall coordinates into one integer bitmask."""
    food_grid = state.getFood()
    bitmask = 0
    for bit_idx, (x_coord, y_coord) in enumerate(context.non_wall_coords):
        if food_grid[x_coord][y_coord]:
            bitmask |= 1 << bit_idx
    return int(bitmask)


def _build_food_bitmask_observation(
    context: ObservationContext,
    state: runtime_pacman.GameState,
    step_count: int,
) -> dict[str, Any]:
    """Build compact observation with one non-wall food bitmask integer."""
    observation = _extract_common_agent_features(context=context, state=state, step_count=step_count)
    observation["food_bitmask"] = _build_food_bitmask(context, state)
    return observation


def _direction_to_target(source: tuple[float, float], target: tuple[float, float]) -> int:
    """Encode coarse direction from source to target as integer id."""
    delta_x = float(target[0] - source[0])
    delta_y = float(target[1] - source[1])
    if delta_x == 0.0 and delta_y == 0.0:
        return 0
    if abs(delta_x) >= abs(delta_y):
        if delta_x > 0:
            return 3
        return 4
    if delta_y > 0:
        return 1
    return 2


def _bucket_distance(distance: int | None, bucket_size: int) -> int:
    """Convert scalar distance to coarse bucket id with empty sentinel -1."""
    if distance is None:
        return -1
    return int(distance // bucket_size)


def _build_bitmask_distance_bucket_observation(
    context: ObservationContext,
    state: runtime_pacman.GameState,
    step_count: int,
) -> dict[str, Any]:
    """Build bitmask observation with nearest-food/ghost bucket features."""
    observation = _build_food_bitmask_observation(context=context, state=state, step_count=step_count)

    pacman_position = state.getPacmanPosition()
    food_positions = state.getFood().asList()
    ghost_positions = [ghost_state.getPosition() for ghost_state in state.getGhostStates() if ghost_state.getPosition() is not None]

    nearest_food = min(
        food_positions,
        key=lambda pos: (abs(pos[0] - pacman_position[0]) + abs(pos[1] - pacman_position[1]), pos),
        default=None,
    )
    nearest_ghost = min(
        ghost_positions,
        key=lambda pos: (abs(pos[0] - pacman_position[0]) + abs(pos[1] - pacman_position[1]), pos),
        default=None,
    )

    nearest_food_distance = None
    if nearest_food is not None:
        nearest_food_distance = int(abs(nearest_food[0] - pacman_position[0]) + abs(nearest_food[1] - pacman_position[1]))

    nearest_ghost_distance = None
    if nearest_ghost is not None:
        nearest_ghost_distance = int(abs(nearest_ghost[0] - pacman_position[0]) + abs(nearest_ghost[1] - pacman_position[1]))

    observation.update(
        {
            "nearest_food_bucket": np.array(
                [_bucket_distance(nearest_food_distance, context.distance_bucket_size)],
                dtype=np.int32,
            ),
            "nearest_ghost_bucket": np.array(
                [_bucket_distance(nearest_ghost_distance, context.distance_bucket_size)],
                dtype=np.int32,
            ),
            "nearest_food_direction": np.array(
                [0 if nearest_food is None else _direction_to_target(pacman_position, nearest_food)],
                dtype=np.int32,
            ),
            "nearest_ghost_direction": np.array(
                [0 if nearest_ghost is None else _direction_to_target(pacman_position, nearest_ghost)],
                dtype=np.int32,
            ),
        }
    )
    return observation


def _build_chunked_food_observation(
    context: ObservationContext,
    state: runtime_pacman.GameState,
    step_count: int,
) -> dict[str, Any]:
    """Build chunked observation with per-chunk food flags and local chunk detail."""
    observation = _extract_common_agent_features(context=context, state=state, step_count=step_count)

    chunks_x = math.ceil(context.width / context.chunk_w)
    chunks_y = math.ceil(context.height / context.chunk_h)

    food_grid = state.getFood()
    walls_grid = state.getWalls()
    capsules_set = {(int(x_coord), int(y_coord)) for x_coord, y_coord in state.getCapsules()}

    chunk_food_presence = np.zeros((chunks_x, chunks_y), dtype=np.int8)
    for chunk_x in range(chunks_x):
        for chunk_y in range(chunks_y):
            x_start = chunk_x * context.chunk_w
            y_start = chunk_y * context.chunk_h
            x_stop = min(context.width, x_start + context.chunk_w)
            y_stop = min(context.height, y_start + context.chunk_h)
            has_food = 0
            for x_coord in range(x_start, x_stop):
                for y_coord in range(y_start, y_stop):
                    if food_grid[x_coord][y_coord]:
                        has_food = 1
                        break
                if has_food:
                    break
            chunk_food_presence[chunk_x, chunk_y] = has_food

    pacman_position = state.getPacmanPosition()
    pac_x = int(round(float(pacman_position[0])))
    pac_y = int(round(float(pacman_position[1])))
    pacman_chunk_x = max(0, min(chunks_x - 1, pac_x // context.chunk_w))
    pacman_chunk_y = max(0, min(chunks_y - 1, pac_y // context.chunk_h))

    chunk_food = np.zeros((context.chunk_w, context.chunk_h), dtype=np.int8)
    chunk_walls = np.ones((context.chunk_w, context.chunk_h), dtype=np.int8)
    chunk_capsules = np.zeros((context.chunk_w, context.chunk_h), dtype=np.int8)

    global_x_start = pacman_chunk_x * context.chunk_w
    global_y_start = pacman_chunk_y * context.chunk_h
    for local_x in range(context.chunk_w):
        for local_y in range(context.chunk_h):
            global_x = global_x_start + local_x
            global_y = global_y_start + local_y
            if global_x < 0 or global_x >= context.width or global_y < 0 or global_y >= context.height:
                continue
            chunk_walls[local_x, local_y] = int(walls_grid[global_x][global_y])
            chunk_food[local_x, local_y] = int(food_grid[global_x][global_y])
            chunk_capsules[local_x, local_y] = int((global_x, global_y) in capsules_set)

    observation.update(
        {
            "chunk_food_presence": chunk_food_presence,
            "pacman_chunk_coord": np.array([pacman_chunk_x, pacman_chunk_y], dtype=np.int32),
            "pacman_chunk_food": chunk_food,
            "pacman_chunk_walls": chunk_walls,
            "pacman_chunk_capsules": chunk_capsules,
        }
    )
    return observation


def _build_simple_distance_space(context: ObservationContext) -> gym.Space:
    """Build observation space for minimal distance/direction features."""
    max_dist = int(context.width + context.height)
    max_bucket = math.ceil(max_dist / context.distance_bucket_size)
    return spaces.Dict(
        {
            "nearest_food_bucket": spaces.Box(
                low=-1,
                high=max_bucket,
                shape=(1,),
                dtype=np.int32,
            ),
            "nearest_ghost_bucket": spaces.Box(
                low=-1,
                high=max_bucket,
                shape=(1,),
                dtype=np.int32,
            ),
            "nearest_food_direction": spaces.Box(
                low=0,
                high=4,
                shape=(1,),
                dtype=np.int32,
            ),
            "nearest_ghost_direction": spaces.Box(
                low=0,
                high=4,
                shape=(1,),
                dtype=np.int32,
            ),
        }
    )


def _build_simple_distance_observation(
    context: ObservationContext,
    state: runtime_pacman.GameState,
    step_count: int,
) -> dict[str, Any]:
    """Build minimal observation with nearest food/ghost distance buckets and directions."""
    del step_count
    pacman_position = state.getPacmanPosition()
    food_positions = state.getFood().asList()
    ghost_positions = [
        ghost_state.getPosition()
        for ghost_state in state.getGhostStates()
        if ghost_state.getPosition() is not None
    ]

    nearest_food = min(
        food_positions,
        key=lambda pos: (abs(pos[0] - pacman_position[0]) + abs(pos[1] - pacman_position[1]), pos),
        default=None,
    )
    nearest_ghost = min(
        ghost_positions,
        key=lambda pos: (abs(pos[0] - pacman_position[0]) + abs(pos[1] - pacman_position[1]), pos),
        default=None,
    )

    nearest_food_distance = None
    if nearest_food is not None:
        nearest_food_distance = int(
            abs(nearest_food[0] - pacman_position[0]) + abs(nearest_food[1] - pacman_position[1])
        )

    nearest_ghost_distance = None
    if nearest_ghost is not None:
        nearest_ghost_distance = int(
            abs(nearest_ghost[0] - pacman_position[0]) + abs(nearest_ghost[1] - pacman_position[1])
        )

    return {
        "nearest_food_bucket": np.array(
            [_bucket_distance(nearest_food_distance, context.distance_bucket_size)],
            dtype=np.int32,
        ),
        "nearest_ghost_bucket": np.array(
            [_bucket_distance(nearest_ghost_distance, context.distance_bucket_size)],
            dtype=np.int32,
        ),
        "nearest_food_direction": np.array(
            [0 if nearest_food is None else _direction_to_target(pacman_position, nearest_food)],
            dtype=np.int32,
        ),
        "nearest_ghost_direction": np.array(
            [0 if nearest_ghost is None else _direction_to_target(pacman_position, nearest_ghost)],
            dtype=np.int32,
        ),
    }


_OBSERVATION_REGISTRY: dict[str, ObservationSpec] = {
    ObservationName.RAW.value: ObservationSpec(
        name=ObservationName.RAW.value,
        build_space=_build_raw_space,
        build_observation=_build_raw_observation,
    ),
    ObservationName.CHUNKED_FOOD.value: ObservationSpec(
        name=ObservationName.CHUNKED_FOOD.value,
        build_space=_build_chunked_food_space,
        build_observation=_build_chunked_food_observation,
    ),
    ObservationName.FOOD_BITMASK.value: ObservationSpec(
        name=ObservationName.FOOD_BITMASK.value,
        build_space=_build_food_bitmask_space,
        build_observation=_build_food_bitmask_observation,
    ),
    ObservationName.BITMASK_DISTANCE_BUCKETS.value: ObservationSpec(
        name=ObservationName.BITMASK_DISTANCE_BUCKETS.value,
        build_space=_build_bitmask_distance_buckets_space,
        build_observation=_build_bitmask_distance_bucket_observation,
    ),
    ObservationName.SIMPLE_DISTANCE_BUCKETS.value: ObservationSpec(
        name=ObservationName.SIMPLE_DISTANCE_BUCKETS.value,
        build_space=_build_simple_distance_space,
        build_observation=_build_simple_distance_observation,
    ),
}


def get_observation_spec(name_or_enum: str | ObservationName) -> ObservationSpec:
    """Resolve built-in observation spec by string name or enum attribute."""
    if isinstance(name_or_enum, ObservationName):
        key = name_or_enum.value
    else:
        key = str(name_or_enum)
    spec = _OBSERVATION_REGISTRY.get(key)
    if spec is None:
        available = ", ".join(sorted(_OBSERVATION_REGISTRY.keys()))
        raise ValueError(f"Unknown observation spec '{key}'. Available: {available}")
    return spec
