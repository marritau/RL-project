"""Policy iteration over empirical observation MDPs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Hashable

from .obs_mdp import ObsMDPModel


ObsKey = Hashable


@dataclass
class PolicyIterationResult:
    """Output container for policy iteration."""

    policy: dict[ObsKey, int]
    values: dict[ObsKey, float]
    policy_iterations: int
    evaluation_sweeps: int
    last_delta: float
    value_history: list[float]


def _expected_return(
    values: dict[ObsKey, float],
    outcomes: list,
    gamma: float,
) -> float:
    expected = 0.0
    for outcome in outcomes:
        info = outcome.info or {}
        terminated_fraction = float(info.get("terminated_fraction", 1.0 if outcome.terminated else 0.0))
        truncated_fraction = float(info.get("truncated_fraction", 1.0 if outcome.truncated else 0.0))
        continuation = max(0.0, 1.0 - max(terminated_fraction, truncated_fraction))
        expected += outcome.probability * (outcome.reward + gamma * continuation * values.get(outcome.next_state, 0.0))
    return expected


def policy_evaluation(
    states: list[ObsKey],
    outcomes_cache: dict[tuple[ObsKey, int], list],
    policy: dict[ObsKey, int],
    values: dict[ObsKey, float],
    *,
    gamma: float,
    theta: float,
    max_iters: int,
) -> tuple[dict[ObsKey, float], float, int]:
    """Evaluate a fixed policy on an empirical model."""
    sweeps = 0
    while sweeps < max_iters:
        delta = 0.0
        for state in states:
            action = policy.get(state)
            if action is None:
                continue
            outcomes = outcomes_cache.get((state, action), [])
            new_value = _expected_return(values, outcomes, gamma)
            delta = max(delta, abs(new_value - values.get(state, 0.0)))
            values[state] = new_value
        sweeps += 1
        if delta < theta:
            return values, delta, sweeps
    return values, delta, sweeps


def policy_improvement(
    states: list[ObsKey],
    actions_by_state: dict[ObsKey, list[int]],
    outcomes_cache: dict[tuple[ObsKey, int], list],
    values: dict[ObsKey, float],
    *,
    gamma: float,
) -> tuple[dict[ObsKey, int], bool]:
    """Compute greedy policy w.r.t. current value function."""
    policy: dict[ObsKey, int] = {}
    stable = True
    for state in states:
        best_action = None
        best_value = float("-inf")
        for action in actions_by_state.get(state, []):
            outcomes = outcomes_cache.get((state, action), [])
            q_value = _expected_return(values, outcomes, gamma)
            if q_value > best_value:
                best_value = q_value
                best_action = action
        if best_action is None:
            continue
        policy[state] = best_action
    return policy, stable


def policy_iteration(
    model: ObsMDPModel,
    *,
    gamma: float = 0.95,
    theta: float = 1e-4,
    max_eval_iters: int = 50,
    max_policy_iters: int = 50,
    on_iteration: Callable[[int, dict[ObsKey, int], dict[ObsKey, float]], None] | None = None,
) -> PolicyIterationResult:
    """Run policy iteration on the empirical observation MDP model."""
    states = model.states()
    actions_by_state: dict[ObsKey, list[int]] = {state: model.actions_for_state(state) for state in states}
    outcomes_cache: dict[tuple[ObsKey, int], list] = {}
    for state, actions in actions_by_state.items():
        for action in actions:
            outcomes_cache[(state, action)] = model.get_outcomes(state, action)

    values: dict[ObsKey, float] = {state: 0.0 for state in states}
    policy: dict[ObsKey, int] = {
        state: (actions_by_state[state][0] if actions_by_state[state] else 0) for state in states
    }

    policy_iterations = 0
    evaluation_sweeps_total = 0
    last_delta = 0.0
    value_history: list[float] = []

    for _ in range(max_policy_iters):
        policy_iterations += 1
        values, last_delta, sweeps = policy_evaluation(
            states,
            outcomes_cache,
            policy,
            values,
            gamma=gamma,
            theta=theta,
            max_iters=max_eval_iters,
        )
        evaluation_sweeps_total += sweeps
        if values:
            value_history.append(float(sum(values.values()) / len(values)))
        else:
            value_history.append(0.0)

        new_policy, _ = policy_improvement(states, actions_by_state, outcomes_cache, values, gamma=gamma)
        if on_iteration is not None:
            on_iteration(policy_iterations, new_policy, values)
        if new_policy == policy:
            break
        policy = new_policy

    return PolicyIterationResult(
        policy=policy,
        values=values,
        policy_iterations=policy_iterations,
        evaluation_sweeps=evaluation_sweeps_total,
        last_delta=last_delta,
        value_history=value_history,
    )
