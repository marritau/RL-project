"""Algorithmic sanity checks for the four original RL/DP methods."""

from __future__ import annotations

import numpy as np

from pacman_rldp.algorithms.food_bitmask_value_iteration import (
    FoodBitmaskCollectionResult,
    FoodBitmaskValueIterationPlanner,
    TERMINAL_STATE,
)
from pacman_rldp.algorithms.policy_iteration.obs_mdp import ObsMDPModel
from pacman_rldp.algorithms.policy_iteration.policy_iteration_obs import policy_iteration
from pacman_rldp.pipelines_tabular_q import QLearningAgent, SarsaAgent


def test_q_learning_update_uses_max_next_legal_action() -> None:
    agent = QLearningAgent(action_size=3, alpha=1.0, gamma=0.5, epsilon=0.0)
    agent.q_table["next"] = np.array([10.0, 3.0, 8.0], dtype=np.float32)

    agent.update("s", 0, reward=2.0, next_state_key="next", next_legal=[1, 2], done=False)

    # max over legal next actions is action 2 with value 8, not illegal action 0.
    assert np.isclose(agent.q_table["s"][0], 2.0 + 0.5 * 8.0)


def test_sarsa_update_uses_selected_next_action_not_max() -> None:
    agent = SarsaAgent(action_size=3, alpha=1.0, gamma=0.5, epsilon=0.0)
    agent.q_table["next"] = np.array([0.0, 100.0, 4.0], dtype=np.float32)

    agent.update("s", 0, reward=2.0, next_state_key="next", next_action=2, done=False)

    # SARSA is on-policy: it must use Q(s', a'=2)=4, not max Q(s', a)=100.
    assert np.isclose(agent.q_table["s"][0], 2.0 + 0.5 * 4.0)


def test_sarsa_terminal_update_does_not_bootstrap() -> None:
    agent = SarsaAgent(action_size=2, alpha=1.0, gamma=0.99, epsilon=0.0)
    agent.q_table["next"] = np.array([999.0, 999.0], dtype=np.float32)

    agent.update("s", 1, reward=-5.0, next_state_key="next", next_action=0, done=True)

    assert np.isclose(agent.q_table["s"][1], -5.0)


def test_policy_iteration_finds_better_empirical_action() -> None:
    model = ObsMDPModel()
    # action 0 yields reward 1, action 1 yields reward 0 from the same state.
    model.update("s", 0, reward=1.0, next_state="terminal_like", terminated=True, truncated=False)
    model.update("s", 1, reward=0.0, next_state="terminal_like", terminated=True, truncated=False)

    result = policy_iteration(model, gamma=0.9, theta=1e-9, max_eval_iters=10, max_policy_iters=10)

    assert result.policy["s"] == 0
    assert result.values["s"] >= 1.0 - 1e-6


def test_food_bitmask_value_iteration_solves_tiny_empirical_mdp() -> None:
    collection = FoodBitmaskCollectionResult(
        state_action_next_counts={
            "s": {
                0: {TERMINAL_STATE: 1},
                1: {TERMINAL_STATE: 1},
            }
        },
        state_action_next_reward_sums={
            "s": {
                0: {TERMINAL_STATE: 2.0},
                1: {TERMINAL_STATE: -1.0},
            }
        },
        state_action_counts={"s": {0: 1, 1: 1}},
        state_action_legal_actions={"s": (0, 1)},
        exploratory_returns=[],
        exploratory_scores=[],
        exploratory_lengths=[],
        states_seen_by_episode=[],
        samples_seen_by_episode=[],
        transition_samples=2,
        unique_states=2,
    )

    result = FoodBitmaskValueIterationPlanner(
        collection,
        gamma=0.99,
        tolerance=1e-12,
        max_iterations=10,
        progress_every=0,
    ).solve()

    assert result.policy["s"] == 0
    assert np.isclose(result.values["s"], 2.0)
    assert result.q_values["s"][1] == -1.0
