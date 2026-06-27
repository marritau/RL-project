"""Tests for markovian ghost policy behavior."""

from __future__ import annotations

from pacman_rldp.env import PacmanEnv, PacmanEnvConfig
from pacman_rldp.third_party.bk import game as runtime_game


def _first_legal_action(info: dict[str, object]) -> int:
    """Return deterministic legal Pacman action id from info payload."""
    legal_action_ids = info.get("legal_action_ids", [])
    if not isinstance(legal_action_ids, list) or not legal_action_ids:
        return 4
    return int(sorted(int(action_id) for action_id in legal_action_ids)[0])


def test_markovian_policy_excludes_stop_when_moves_exist() -> None:
    """Ensure markovian ghost policy never samples Stop if move actions exist."""
    cfg = PacmanEnvConfig(ghost_policy="markovian", seed=7)
    env = PacmanEnv(cfg)

    legal_actions = [
        runtime_game.Directions.NORTH,
        runtime_game.Directions.WEST,
        runtime_game.Directions.STOP,
    ]
    for _ in range(50):
        sampled = env._sample_markovian_action(legal_actions)
        assert sampled != runtime_game.Directions.STOP

    env.close()


def test_markovian_policy_is_seed_reproducible() -> None:
    """Verify markovian stochastic policy is reproducible under the same seed."""
    cfg = PacmanEnvConfig(ghost_policy="markovian", seed=23, num_ghosts=1)
    env_a = PacmanEnv(cfg)
    env_b = PacmanEnv(cfg)
    _, info_a = env_a.reset(seed=23)
    _, info_b = env_b.reset(seed=23)

    trace_a: list[tuple[float, float]] = []
    trace_b: list[tuple[float, float]] = []
    for _ in range(6):
        action_a = _first_legal_action(info_a)
        action_b = _first_legal_action(info_b)
        _, _, term_a, trunc_a, info_a = env_a.step(action_a)
        _, _, term_b, trunc_b, info_b = env_b.step(action_b)

        ghost_a = env_a.runtime_state.getGhostPosition(1)
        ghost_b = env_b.runtime_state.getGhostPosition(1)
        trace_a.append((float(ghost_a[0]), float(ghost_a[1])))
        trace_b.append((float(ghost_b[0]), float(ghost_b[1])))

        if term_a or trunc_a or term_b or trunc_b:
            break

    assert trace_a == trace_b
    env_a.close()
    env_b.close()


def test_markovian_policy_allows_reverse_at_intersections() -> None:
    """Ensure markovian legal actions include reverse direction when geometrically valid."""
    env = PacmanEnv(PacmanEnvConfig(ghost_policy="markovian", seed=31, num_ghosts=1))
    env.reset(seed=31)
    state = env.runtime_state
    walls = state.getWalls()

    chosen_cell: tuple[int, int] | None = None
    for x_coord in range(1, int(walls.width) - 1):
        for y_coord in range(1, int(walls.height) - 1):
            if walls[x_coord][y_coord]:
                continue
            if walls[x_coord - 1][y_coord] or walls[x_coord + 1][y_coord]:
                continue
            neighbor_count = 0
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                if not walls[x_coord + dx][y_coord + dy]:
                    neighbor_count += 1
            if neighbor_count >= 3:
                chosen_cell = (x_coord, y_coord)
                break
        if chosen_cell is not None:
            break

    if chosen_cell is None:
        env.close()
        raise AssertionError("Failed to find intersection cell for markovian reverse test.")

    ghost_state = state.data.agentStates[1]
    ghost_state.configuration = runtime_game.Configuration(chosen_cell, runtime_game.Directions.EAST)

    runtime_legal = state.getLegalActions(1)
    markovian_legal = env._markovian_legal_actions(state, 1)

    assert runtime_game.Directions.WEST not in runtime_legal
    assert runtime_game.Directions.WEST in markovian_legal
    env.close()
