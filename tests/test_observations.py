"""Tests for observation registry and built-in observation formats."""

from __future__ import annotations

import math

import numpy as np
import pytest

from pacman_rldp.env import (
    ObservationConfig,
    ObservationName,
    PacmanEnv,
    PacmanEnvConfig,
    build_env_config,
    get_observation_spec,
)


def _first_legal_action(info: dict[str, object]) -> int:
    """Return deterministic legal action id from env info payload."""
    legal_ids = info.get("legal_action_ids", [])
    if not isinstance(legal_ids, list) or not legal_ids:
        return 4
    return int(sorted(int(action_id) for action_id in legal_ids)[0])


def _direction_to_target(source: tuple[float, float], target: tuple[float, float]) -> int:
    """Encode coarse direction ID from source to target coordinate."""
    delta_x = float(target[0] - source[0])
    delta_y = float(target[1] - source[1])
    if delta_x == 0.0 and delta_y == 0.0:
        return 0
    if abs(delta_x) >= abs(delta_y):
        return 3 if delta_x > 0 else 4
    return 1 if delta_y > 0 else 2


def test_observation_registry_resolves_by_enum_and_string() -> None:
    """Resolve observation spec by enum and by plain string key."""
    by_enum = get_observation_spec(ObservationName.RAW)
    by_string = get_observation_spec("raw")
    assert by_enum.name == "raw"
    assert by_string.name == "raw"


def test_observation_registry_raises_for_unknown_name() -> None:
    """Raise descriptive error for unknown observation key."""
    with pytest.raises(ValueError):
        get_observation_spec("missing_observation_key")


def test_chunked_food_observation_shapes_and_values() -> None:
    """Check chunked food map and local chunk tensors against runtime state."""
    cfg = PacmanEnvConfig(
        observation=ObservationConfig(name="chunked_food", chunk_w=3, chunk_h=2),
        seed=11,
        num_ghosts=1,
    )
    env = PacmanEnv(cfg)
    observation, _ = env.reset(seed=11)
    assert env.observation_space.contains(observation)

    context = env._observation_context
    state = env.runtime_state
    food_grid = state.getFood()
    walls_grid = state.getWalls()
    capsules = {(int(x_coord), int(y_coord)) for x_coord, y_coord in state.getCapsules()}

    chunks_x = math.ceil(context.width / context.chunk_w)
    chunks_y = math.ceil(context.height / context.chunk_h)
    expected_presence = np.zeros((chunks_x, chunks_y), dtype=np.int8)
    for chunk_x in range(chunks_x):
        for chunk_y in range(chunks_y):
            x_start = chunk_x * context.chunk_w
            y_start = chunk_y * context.chunk_h
            x_stop = min(context.width, x_start + context.chunk_w)
            y_stop = min(context.height, y_start + context.chunk_h)
            expected_presence[chunk_x, chunk_y] = int(
                any(food_grid[x_coord][y_coord] for x_coord in range(x_start, x_stop) for y_coord in range(y_start, y_stop))
            )
    np.testing.assert_array_equal(observation["chunk_food_presence"], expected_presence)

    pac_chunk_x = int(observation["pacman_chunk_coord"][0])
    pac_chunk_y = int(observation["pacman_chunk_coord"][1])
    expected_food = np.zeros((context.chunk_w, context.chunk_h), dtype=np.int8)
    expected_walls = np.ones((context.chunk_w, context.chunk_h), dtype=np.int8)
    expected_capsules = np.zeros((context.chunk_w, context.chunk_h), dtype=np.int8)
    x_origin = pac_chunk_x * context.chunk_w
    y_origin = pac_chunk_y * context.chunk_h
    for local_x in range(context.chunk_w):
        for local_y in range(context.chunk_h):
            x_coord = x_origin + local_x
            y_coord = y_origin + local_y
            if x_coord < 0 or x_coord >= context.width or y_coord < 0 or y_coord >= context.height:
                continue
            expected_walls[local_x, local_y] = int(walls_grid[x_coord][y_coord])
            expected_food[local_x, local_y] = int(food_grid[x_coord][y_coord])
            expected_capsules[local_x, local_y] = int((x_coord, y_coord) in capsules)
    np.testing.assert_array_equal(observation["pacman_chunk_food"], expected_food)
    np.testing.assert_array_equal(observation["pacman_chunk_walls"], expected_walls)
    np.testing.assert_array_equal(observation["pacman_chunk_capsules"], expected_capsules)
    env.close()


def test_food_bitmask_is_deterministic_and_uses_non_wall_cells() -> None:
    """Encode food into one deterministic integer over walkable cells only."""
    cfg = PacmanEnvConfig(observation=ObservationConfig(name="food_bitmask"), seed=13, num_ghosts=1)
    env_a = PacmanEnv(cfg)
    env_b = PacmanEnv(cfg)
    observation_a, _ = env_a.reset(seed=13)
    observation_b, _ = env_b.reset(seed=13)

    assert env_a.observation_space.contains(observation_a)
    assert env_b.observation_space.contains(observation_b)
    assert int(observation_a["food_bitmask"]) == int(observation_b["food_bitmask"])

    state = env_a.runtime_state
    food_grid = state.getFood()
    expected_bitmask = 0
    for bit_idx, (x_coord, y_coord) in enumerate(env_a._observation_context.non_wall_coords):
        if food_grid[x_coord][y_coord]:
            expected_bitmask |= 1 << bit_idx
    assert int(observation_a["food_bitmask"]) == int(expected_bitmask)

    env_a.close()
    env_b.close()


def test_bitmask_distance_bucket_features_are_correct() -> None:
    """Check distance buckets and coarse directions for nearest food/ghost."""
    cfg = PacmanEnvConfig(
        observation=ObservationConfig(name="bitmask_distance_buckets", distance_bucket_size=2),
        seed=17,
        num_ghosts=1,
    )
    env = PacmanEnv(cfg)
    observation, _ = env.reset(seed=17)
    assert env.observation_space.contains(observation)

    state = env.runtime_state
    pac_pos = state.getPacmanPosition()
    food_positions = state.getFood().asList()
    ghost_positions = [
        ghost_state.getPosition()
        for ghost_state in state.getGhostStates()
        if ghost_state.getPosition() is not None
    ]

    nearest_food = min(
        food_positions,
        key=lambda pos: (abs(pos[0] - pac_pos[0]) + abs(pos[1] - pac_pos[1]), pos),
        default=None,
    )
    nearest_ghost = min(
        ghost_positions,
        key=lambda pos: (abs(pos[0] - pac_pos[0]) + abs(pos[1] - pac_pos[1]), pos),
        default=None,
    )

    if nearest_food is None:
        expected_food_bucket = -1
        expected_food_direction = 0
    else:
        food_distance = int(abs(nearest_food[0] - pac_pos[0]) + abs(nearest_food[1] - pac_pos[1]))
        expected_food_bucket = food_distance // 2
        expected_food_direction = _direction_to_target(pac_pos, nearest_food)

    if nearest_ghost is None:
        expected_ghost_bucket = -1
        expected_ghost_direction = 0
    else:
        ghost_distance = int(abs(nearest_ghost[0] - pac_pos[0]) + abs(nearest_ghost[1] - pac_pos[1]))
        expected_ghost_bucket = ghost_distance // 2
        expected_ghost_direction = _direction_to_target(pac_pos, nearest_ghost)

    assert int(observation["nearest_food_bucket"][0]) == expected_food_bucket
    assert int(observation["nearest_ghost_bucket"][0]) == expected_ghost_bucket
    assert int(observation["nearest_food_direction"][0]) == expected_food_direction
    assert int(observation["nearest_ghost_direction"][0]) == expected_ghost_direction
    env.close()


def test_env_uses_observation_name_from_config() -> None:
    """Switch observation format from config and keep raw as fallback default."""
    bitmask_cfg = build_env_config({"observation": {"name": "food_bitmask"}, "seed": 23})
    bitmask_env = PacmanEnv(bitmask_cfg)
    bitmask_observation, _ = bitmask_env.reset(seed=23)
    assert "food_bitmask" in bitmask_observation
    assert "food" not in bitmask_observation
    bitmask_env.close()

    raw_cfg = build_env_config({"seed": 23})
    raw_env = PacmanEnv(raw_cfg)
    raw_observation, _ = raw_env.reset(seed=23)
    assert "food" in raw_observation
    assert "food_bitmask" not in raw_observation
    raw_env.close()


def test_custom_observation_spec_overrides_config_name() -> None:
    """Honor explicitly mounted spec over config-selected observation name."""
    mounted_spec = get_observation_spec("food_bitmask")
    cfg = PacmanEnvConfig(observation=ObservationConfig(name="raw"), seed=29)
    env = PacmanEnv(config=cfg, observation_spec=mounted_spec)
    observation, info = env.reset(seed=29)
    action = _first_legal_action(info)
    assert "food_bitmask" in observation
    assert "food" not in observation
    next_observation, _, _, _, _ = env.step(action)
    assert "food_bitmask" in next_observation
    env.close()
