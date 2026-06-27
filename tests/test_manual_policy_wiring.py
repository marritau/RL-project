"""Tests for manual-mode ghost policy wiring."""

from __future__ import annotations

from pacman_rldp.third_party.bk import layout as runtime_layout
from pacman_rldp.third_party.bk import pacman as runtime_pacman
from pacman_rldp.visuals.manual import LoopPathGhost, MarkovianGhost, build_manual_ghosts


def _loop_matrix() -> list[list[int]]:
    """Return valid loop matrix for default smallClassic layout."""
    return [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0],
        [0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ]


def test_build_manual_ghosts_uses_requested_policy() -> None:
    """Ensure manual ghost factory creates policy-specific ghost agents."""
    chosen_layout = runtime_layout.getLayout("smallClassic")
    assert chosen_layout is not None

    random_ghosts = build_manual_ghosts(
        ghost_policy="random",
        ghost_count=1,
        chosen_layout=chosen_layout,
        ghost_loop_matrix=None,
        ghost_loop_direction="clockwise",
    )
    assert random_ghosts[0].__class__.__name__ == "RandomGhost"

    markovian_ghosts = build_manual_ghosts(
        ghost_policy="markovian",
        ghost_count=1,
        chosen_layout=chosen_layout,
        ghost_loop_matrix=None,
        ghost_loop_direction="clockwise",
    )
    assert isinstance(markovian_ghosts[0], MarkovianGhost)

    loop_path_ghosts = build_manual_ghosts(
        ghost_policy="loop_path",
        ghost_count=1,
        chosen_layout=chosen_layout,
        ghost_loop_matrix=_loop_matrix(),
        ghost_loop_direction="clockwise",
    )
    assert isinstance(loop_path_ghosts[0], LoopPathGhost)


def test_markovian_manual_distribution_uses_only_legal_actions() -> None:
    """Ensure manual markovian ghost never outputs action outside runtime legal set."""
    chosen_layout = runtime_layout.getLayout("smallClassic")
    assert chosen_layout is not None
    state = runtime_pacman.GameState()
    state.initialize(chosen_layout, numGhostAgents=1)

    ghost = MarkovianGhost(1)
    distribution = ghost.getDistribution(state)
    legal_actions = set(state.getLegalActions(1))

    assert distribution
    assert set(distribution.keys()).issubset(legal_actions)


def test_manual_loop_path_supports_anticlockwise_direction() -> None:
    """Ensure manual loop-path builder supports anticlockwise orientation."""
    chosen_layout = runtime_layout.getLayout("smallClassic")
    assert chosen_layout is not None
    clockwise_ghosts = build_manual_ghosts(
        ghost_policy="loop_path",
        ghost_count=1,
        chosen_layout=chosen_layout,
        ghost_loop_matrix=_loop_matrix(),
        ghost_loop_direction="clockwise",
    )
    anticlockwise_ghosts = build_manual_ghosts(
        ghost_policy="loop_path",
        ghost_count=1,
        chosen_layout=chosen_layout,
        ghost_loop_matrix=_loop_matrix(),
        ghost_loop_direction="anticlockwise",
    )

    clockwise_cycle = clockwise_ghosts[0].manager.cycle
    anticlockwise_cycle = anticlockwise_ghosts[0].manager.cycle
    assert anticlockwise_cycle == [clockwise_cycle[0], *list(reversed(clockwise_cycle[1:]))]
