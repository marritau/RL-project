"""Tests for matrix-driven loop_path ghost policy."""

from __future__ import annotations

from pacman_rldp.env import PacmanEnv, PacmanEnvConfig


def perimeter_loop_matrix() -> list[list[int]]:
    """Return default perimeter-style loop matrix for smallClassic layout."""
    return [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
        [0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ]


def pick_first_legal_action(info: dict[str, object]) -> int:
    """Return deterministic first legal action id from info payload."""
    legal_action_ids = info.get("legal_action_ids", [])
    if not isinstance(legal_action_ids, list) or not legal_action_ids:
        return 4
    return int(sorted(int(action_id) for action_id in legal_action_ids)[0])


def build_loop_env(
    num_ghosts: int = 1,
    seed: int = 7,
    direction: str = "clockwise",
) -> PacmanEnv:
    """Construct environment configured for loop_path ghost policy."""
    cfg = PacmanEnvConfig(
        layout_name="smallClassic",
        num_ghosts=num_ghosts,
        seed=seed,
        ghost_policy="loop_path",
        ghost_loop_matrix=perimeter_loop_matrix(),
        ghost_loop_direction=direction,
    )
    return PacmanEnv(cfg)


def test_loop_policy_is_deterministic() -> None:
    """Verify loop_path ghost movement is deterministic under fixed seed."""
    env_a = build_loop_env(num_ghosts=2, seed=42)
    env_b = build_loop_env(num_ghosts=2, seed=42)

    _, info_a = env_a.reset(seed=42)
    _, info_b = env_b.reset(seed=42)

    trace_a: list[list[tuple[float, float]]] = []
    trace_b: list[list[tuple[float, float]]] = []

    for _ in range(6):
        action_a = pick_first_legal_action(info_a)
        action_b = pick_first_legal_action(info_b)
        _, _, term_a, trunc_a, info_a = env_a.step(action_a)
        _, _, term_b, trunc_b, info_b = env_b.step(action_b)

        ghosts_a = [
            (float(state.getPosition()[0]), float(state.getPosition()[1]))
            for state in env_a.runtime_state.getGhostStates()[:2]
        ]
        ghosts_b = [
            (float(state.getPosition()[0]), float(state.getPosition()[1]))
            for state in env_b.runtime_state.getGhostStates()[:2]
        ]
        trace_a.append(ghosts_a)
        trace_b.append(ghosts_b)
        if term_a or trunc_a or term_b or trunc_b:
            break

    assert trace_a == trace_b
    env_a.close()
    env_b.close()


def test_loop_policy_even_spacing_for_multiple_ghosts() -> None:
    """Verify ghosts are initialized with even spacing over one shared loop."""
    env = build_loop_env(num_ghosts=2, seed=13)
    env.reset(seed=13)

    cycle_len = len(env._ghost_loop_cycle)
    ghost_one_idx = env._ghost_loop_index_by_agent[1]
    ghost_two_idx = env._ghost_loop_index_by_agent[2]
    spacing = (ghost_two_idx - ghost_one_idx) % cycle_len

    assert cycle_len > 0
    assert spacing == cycle_len // 2
    env.close()


def test_loop_policy_snaps_ghost_to_loop_after_step() -> None:
    """Verify off-path spawn is snapped to nearest loop node and reaches loop deterministically."""
    env = build_loop_env(num_ghosts=1, seed=17)
    _, info = env.reset(seed=17)
    loop_set = set(env._ghost_loop_cycle)
    reached_loop = False
    for _ in range(20):
        action = pick_first_legal_action(info)
        _, _, terminated, truncated, info = env.step(action)
        ghost_pos = env.runtime_state.getGhostPosition(1)
        ghost_coord = (int(round(ghost_pos[0])), int(round(ghost_pos[1])))
        if ghost_coord in loop_set:
            reached_loop = True
            break
        if terminated or truncated:
            break
    assert reached_loop is True
    env.close()


def test_loop_policy_supports_anticlockwise_cycle_direction() -> None:
    """Verify anticlockwise cycle is reverse ordering of clockwise cycle."""
    env_clockwise = build_loop_env(num_ghosts=1, seed=41, direction="clockwise")
    env_anticlockwise = build_loop_env(num_ghosts=1, seed=41, direction="anticlockwise")
    env_clockwise.reset(seed=41)
    env_anticlockwise.reset(seed=41)

    clockwise_cycle = env_clockwise._ghost_loop_cycle
    anticlockwise_cycle = env_anticlockwise._ghost_loop_cycle

    assert len(clockwise_cycle) == len(anticlockwise_cycle)
    assert anticlockwise_cycle == [clockwise_cycle[0], *list(reversed(clockwise_cycle[1:]))]
    env_clockwise.close()
    env_anticlockwise.close()


def test_loop_policy_rejects_wrong_matrix_dimensions() -> None:
    """Verify matrix shape mismatches are rejected."""
    bad_matrix = [[1, 0], [0, 1]]
    cfg = PacmanEnvConfig(
        ghost_policy="loop_path",
        ghost_loop_matrix=bad_matrix,
    )
    env = PacmanEnv(cfg)
    try:
        env.reset(seed=5)
        raised = False
    except ValueError:
        raised = True
    assert raised is True
    env.close()


def test_loop_policy_rejects_non_binary_matrix_values() -> None:
    """Verify non-binary matrix values are rejected."""
    matrix = perimeter_loop_matrix()
    matrix[1][1] = 2
    cfg = PacmanEnvConfig(
        ghost_policy="loop_path",
        ghost_loop_matrix=matrix,
    )
    env = PacmanEnv(cfg)
    try:
        env.reset(seed=9)
        raised = False
    except ValueError:
        raised = True
    assert raised is True
    env.close()


def test_loop_policy_rejects_non_cycle_path() -> None:
    """Verify disconnected or non-cycle masks are rejected."""
    matrix = [[0 for _ in range(20)] for _ in range(7)]
    matrix[5][1] = 1
    matrix[5][2] = 1
    matrix[5][3] = 1

    cfg = PacmanEnvConfig(
        ghost_policy="loop_path",
        ghost_loop_matrix=matrix,
    )
    env = PacmanEnv(cfg)
    try:
        env.reset(seed=21)
        raised = False
    except ValueError:
        raised = True
    assert raised is True
    env.close()
