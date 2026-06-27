"""MDP adapter tests for transition modeling behavior."""

from __future__ import annotations

from pacman_rldp.algorithms import PacmanMDPAdapter
from pacman_rldp.env import PacmanEnv, PacmanEnvConfig


def test_transition_probabilities_sum_to_one() -> None:
    """Ensure one-step outcome probabilities are normalized."""
    env = PacmanEnv(PacmanEnvConfig(num_ghosts=1, seed=5))
    _, info = env.reset(seed=5)
    adapter = PacmanMDPAdapter(env)

    state = env.runtime_state
    action = int(sorted(info["legal_action_ids"])[0])
    outcomes = adapter.transition_outcomes(state, action)

    assert outcomes
    probability_sum = sum(outcome.probability for outcome in outcomes)
    assert abs(probability_sum - 1.0) < 1e-9
    env.close()


def test_adapter_exposes_legal_actions() -> None:
    """Ensure adapter legal-action API mirrors environment action legality."""
    env = PacmanEnv(PacmanEnvConfig(num_ghosts=1, seed=2))
    _, info = env.reset(seed=2)
    adapter = PacmanMDPAdapter(env)

    state = env.runtime_state
    adapter_actions = sorted(adapter.available_actions(state))
    env_actions = sorted(int(action_id) for action_id in info["legal_action_ids"])

    assert adapter_actions == env_actions
    env.close()
