"""Approximate tabular value iteration over aggregated food-bitmask observations."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Hashable

import numpy as np

from ..env import PacmanEnv

TERMINAL_STATE: Hashable = ("__terminal__",)
_ACTION_DELTAS: dict[int, tuple[int, int]] = {
    0: (0, 1),   # north
    1: (0, -1),  # south
    2: (1, 0),   # east
    3: (-1, 0),  # west
    4: (0, 0),   # stop
}


def encode_food_bitmask_observation(observation: dict[str, Any]) -> Hashable:
    """Freeze food-bitmask observation into a hashable aggregated tabular state."""
    return (
        tuple(int(round(float(v))) for v in np.asarray(observation["pacman_position"]).reshape(-1).tolist()),
        tuple(round(float(v), 3) for v in np.asarray(observation["ghost_positions"]).reshape(-1).tolist()),
        tuple(int(v) for v in np.asarray(observation["ghost_timers"]).reshape(-1).tolist()),
        tuple(int(v) for v in np.asarray(observation["ghost_present"]).reshape(-1).tolist()),
        int(observation["food_bitmask"]),
    )


@dataclass
class FoodBitmaskCollectionResult:
    """Collected empirical transition model and coverage traces."""

    state_action_next_counts: dict[Hashable, dict[int, dict[Hashable, int]]]
    state_action_next_reward_sums: dict[Hashable, dict[int, dict[Hashable, float]]]
    state_action_counts: dict[Hashable, dict[int, int]]
    state_action_legal_actions: dict[Hashable, tuple[int, ...]]
    exploratory_returns: list[float]
    exploratory_scores: list[float]
    exploratory_lengths: list[int]
    states_seen_by_episode: list[int]
    samples_seen_by_episode: list[int]
    transition_samples: int
    unique_states: int


@dataclass
class FoodBitmaskValueIterationResult:
    """Outputs of approximate discounted value iteration."""

    values: dict[Hashable, float]
    policy: dict[Hashable, int]
    q_values: dict[Hashable, dict[int, float]]
    residual_history: list[float]
    summary: dict[str, Any]


class FoodBitmaskHelper:
    """Helpers for decoding food bitmask states and building simple heuristics."""

    def __init__(self, non_wall_coords: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> None:
        self.non_wall_coords = [tuple(map(int, coord)) for coord in non_wall_coords]

    def decode_food_positions(self, food_bitmask: int) -> list[tuple[int, int]]:
        positions: list[tuple[int, int]] = []
        mask = int(food_bitmask)
        bit_idx = 0
        while mask:
            if mask & 1:
                positions.append(self.non_wall_coords[bit_idx])
            mask >>= 1
            bit_idx += 1
        return positions

    @staticmethod
    def _manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
        return int(abs(int(a[0]) - int(b[0])) + abs(int(a[1]) - int(b[1])))

    def _nearest_food_distance(self, pos: tuple[int, int], food_positions: list[tuple[int, int]]) -> int:
        if not food_positions:
            return 0
        return min(self._manhattan(pos, food_pos) for food_pos in food_positions)

    def _active_ghost_positions(self, observation: dict[str, Any]) -> list[tuple[int, int]]:
        ghost_positions = np.asarray(observation["ghost_positions"]).reshape(-1, 2)
        ghost_timers = np.asarray(observation["ghost_timers"]).reshape(-1)
        ghost_present = np.asarray(observation["ghost_present"]).reshape(-1)
        active: list[tuple[int, int]] = []
        for ghost_pos, timer, present in zip(ghost_positions, ghost_timers, ghost_present):
            if int(present) != 1 or int(timer) > 0:
                continue
            active.append((int(round(float(ghost_pos[0]))), int(round(float(ghost_pos[1])))))
        return active

    def heuristic_action(self, observation: dict[str, Any], legal_action_ids: list[int], *, rng: np.random.Generator, epsilon_random: float = 0.0, danger_distance: int = 2) -> int:
        if not legal_action_ids:
            return 4
        if epsilon_random > 0.0 and float(rng.random()) < float(epsilon_random):
            candidate_actions = [a for a in legal_action_ids if a != 4] or legal_action_ids
            return int(rng.choice(np.asarray(candidate_actions, dtype=np.int64)))

        pacman_position = tuple(int(round(float(v))) for v in np.asarray(observation["pacman_position"]).reshape(-1).tolist())
        food_positions = self.decode_food_positions(int(observation["food_bitmask"]))
        active_ghosts = self._active_ghost_positions(observation)

        def score_action(action_id: int) -> tuple[float, float, float, int]:
            delta = _ACTION_DELTAS.get(int(action_id), (0, 0))
            next_pos = (pacman_position[0] + delta[0], pacman_position[1] + delta[1])
            nearest_food = self._nearest_food_distance(next_pos, food_positions)
            nearest_ghost = min((self._manhattan(next_pos, ghost_pos) for ghost_pos in active_ghosts), default=999)
            ghost_penalty = 0.0
            if nearest_ghost <= 0:
                ghost_penalty = -10_000.0
            elif nearest_ghost <= danger_distance:
                ghost_penalty = -500.0 / max(1.0, float(nearest_ghost))
            stop_penalty = -0.25 if int(action_id) == 4 else 0.0
            food_bonus = 5.0 if next_pos in food_positions else 0.0
            return (ghost_penalty + food_bonus - float(nearest_food) + stop_penalty, float(nearest_ghost), -float(nearest_food), -int(action_id))

        best_action = max(legal_action_ids, key=score_action)
        return int(best_action)


class FoodBitmaskObservationModelBuilder:
    """Collect an empirical MDP over aggregated food-bitmask observations."""

    def __init__(
        self,
        env: PacmanEnv,
        *,
        base_seed: int = 42,
        collection_episodes: int = 2000,
        progress_every: int = 100,
        max_steps_per_episode: int | None = None,
        epsilon_random: float = 0.35,
        danger_distance: int = 2,
    ) -> None:
        self.env = env
        self.base_seed = int(base_seed)
        self.collection_episodes = int(collection_episodes)
        self.progress_every = max(0, int(progress_every))
        self.max_steps_per_episode = None if max_steps_per_episode is None else int(max_steps_per_episode)
        self.epsilon_random = float(epsilon_random)
        self.danger_distance = int(danger_distance)
        self.helper = FoodBitmaskHelper(env._observation_context.non_wall_coords)

    def _select_exploration_action(self, observation: dict[str, Any], info: dict[str, Any], rng: np.random.Generator) -> int:
        legal_action_ids = [int(a) for a in info.get("legal_action_ids", [])]
        return self.helper.heuristic_action(
            observation,
            legal_action_ids,
            rng=rng,
            epsilon_random=self.epsilon_random,
            danger_distance=self.danger_distance,
        )

    def collect(self) -> FoodBitmaskCollectionResult:
        start_time = perf_counter()
        rng = np.random.default_rng(self.base_seed)

        state_action_next_counts: dict[Hashable, dict[int, dict[Hashable, int]]] = {}
        state_action_next_reward_sums: dict[Hashable, dict[int, dict[Hashable, float]]] = {}
        state_action_counts: dict[Hashable, dict[int, int]] = {}
        state_action_legal_actions: dict[Hashable, tuple[int, ...]] = {}

        exploratory_returns: list[float] = []
        exploratory_scores: list[float] = []
        exploratory_lengths: list[int] = []
        states_seen_by_episode: list[int] = []
        samples_seen_by_episode: list[int] = []
        unique_states: set[Hashable] = {TERMINAL_STATE}
        transition_samples = 0

        for episode_idx in range(self.collection_episodes):
            seed = self.base_seed + episode_idx
            observation, info = self.env.reset(seed=seed)
            state_key = encode_food_bitmask_observation(observation)
            unique_states.add(state_key)
            total_reward = 0.0
            final_score = float(info.get("score", 0.0))
            episode_steps = 0

            while True:
                legal_action_ids = tuple(int(a) for a in info.get("legal_action_ids", []))
                if not legal_action_ids:
                    break

                state_action_legal_actions[state_key] = legal_action_ids
                action = self._select_exploration_action(observation, info, rng)
                next_observation, reward, terminated, truncated, next_info = self.env.step(action)
                next_state_key = TERMINAL_STATE if (terminated or truncated) else encode_food_bitmask_observation(next_observation)
                unique_states.add(next_state_key)

                state_action_counts.setdefault(state_key, {}).setdefault(action, 0)
                state_action_counts[state_key][action] += 1

                state_action_next_counts.setdefault(state_key, {}).setdefault(action, {}).setdefault(next_state_key, 0)
                state_action_next_counts[state_key][action][next_state_key] += 1

                state_action_next_reward_sums.setdefault(state_key, {}).setdefault(action, {}).setdefault(next_state_key, 0.0)
                state_action_next_reward_sums[state_key][action][next_state_key] += float(reward)

                transition_samples += 1
                total_reward += float(reward)
                final_score = float(next_info.get("score", final_score))
                episode_steps += 1

                if terminated or truncated:
                    break
                if self.max_steps_per_episode is not None and episode_steps >= self.max_steps_per_episode:
                    break

                observation = next_observation
                info = next_info
                state_key = next_state_key

            exploratory_returns.append(float(total_reward))
            exploratory_scores.append(float(final_score))
            exploratory_lengths.append(int(episode_steps))
            states_seen_by_episode.append(len(unique_states))
            samples_seen_by_episode.append(transition_samples)

            if self.progress_every > 0 and ((episode_idx + 1) % self.progress_every == 0 or episode_idx == 0):
                elapsed = perf_counter() - start_time
                print(
                    "[collect] "
                    f"episodes={episode_idx + 1}/{self.collection_episodes} "
                    f"states={len(unique_states)} samples={transition_samples} elapsed={elapsed:.2f}s"
                )

        return FoodBitmaskCollectionResult(
            state_action_next_counts=state_action_next_counts,
            state_action_next_reward_sums=state_action_next_reward_sums,
            state_action_counts=state_action_counts,
            state_action_legal_actions=state_action_legal_actions,
            exploratory_returns=exploratory_returns,
            exploratory_scores=exploratory_scores,
            exploratory_lengths=exploratory_lengths,
            states_seen_by_episode=states_seen_by_episode,
            samples_seen_by_episode=samples_seen_by_episode,
            transition_samples=int(transition_samples),
            unique_states=int(len(unique_states)),
        )


class FoodBitmaskValueIterationPlanner:
    """Discounted value iteration on an empirical aggregated MDP."""

    def __init__(
        self,
        collection: FoodBitmaskCollectionResult,
        *,
        gamma: float = 0.99,
        tolerance: float = 1e-6,
        max_iterations: int = 500,
        progress_every: int = 25,
    ) -> None:
        self.collection = collection
        self.gamma = float(gamma)
        self.tolerance = float(tolerance)
        self.max_iterations = int(max_iterations)
        self.progress_every = max(0, int(progress_every))

    def solve(self) -> FoodBitmaskValueIterationResult:
        start_time = perf_counter()
        states = set(self.collection.state_action_legal_actions.keys())
        states.add(TERMINAL_STATE)
        values: dict[Hashable, float] = {state: 0.0 for state in states}
        policy: dict[Hashable, int] = {}
        q_values: dict[Hashable, dict[int, float]] = {}
        residual_history: list[float] = []

        for iteration in range(1, self.max_iterations + 1):
            new_values = dict(values)
            max_residual = 0.0

            for state in states:
                if state == TERMINAL_STATE:
                    new_values[state] = 0.0
                    continue
                actions = self.collection.state_action_legal_actions.get(state, ())
                if not actions:
                    new_values[state] = 0.0
                    continue

                action_q: dict[int, float] = {}
                for action in actions:
                    total_count = self.collection.state_action_counts.get(state, {}).get(action, 0)
                    if total_count <= 0:
                        continue
                    q_val = 0.0
                    next_counts = self.collection.state_action_next_counts[state][action]
                    reward_sums = self.collection.state_action_next_reward_sums[state][action]
                    for next_state, count in next_counts.items():
                        probability = float(count) / float(total_count)
                        mean_reward = float(reward_sums[next_state]) / float(count)
                        q_val += probability * (mean_reward + self.gamma * values.get(next_state, 0.0))
                    action_q[int(action)] = float(q_val)

                if not action_q:
                    new_values[state] = 0.0
                    continue

                best_action, best_value = max(action_q.items(), key=lambda item: (item[1], -item[0]))
                q_values[state] = action_q
                policy[state] = int(best_action)
                new_values[state] = float(best_value)
                max_residual = max(max_residual, abs(new_values[state] - values[state]))

            values = new_values
            residual_history.append(float(max_residual))
            if self.progress_every > 0 and (iteration % self.progress_every == 0 or iteration == 1):
                print(f"[vi] iteration={iteration} residual={max_residual:.8f}")
            if max_residual <= self.tolerance:
                break

        total_seconds = perf_counter() - start_time
        summary = {
            "gamma": self.gamma,
            "tolerance": self.tolerance,
            "iterations": len(residual_history),
            "final_residual": float(residual_history[-1] if residual_history else 0.0),
            "total_seconds": float(total_seconds),
            "discovered_states": int(self.collection.unique_states),
            "transition_samples": int(self.collection.transition_samples),
            "state_action_pairs": int(sum(len(v) for v in self.collection.state_action_counts.values())),
            "policy_states": int(len(policy)),
        }
        return FoodBitmaskValueIterationResult(
            values=values,
            policy=policy,
            q_values=q_values,
            residual_history=residual_history,
            summary=summary,
        )

    def build_artifact(
        self,
        *,
        result: FoodBitmaskValueIterationResult,
        collection_config: dict[str, Any],
        env_config: dict[str, Any],
        raw_config: dict[str, Any],
        non_wall_coords: list[tuple[int, int]] | tuple[tuple[int, int], ...],
    ) -> dict[str, Any]:
        return {
            "model_type": "food_bitmask_empirical_value_iteration",
            "state_encoder": "food_bitmask_observation",
            "policy_table": dict(result.policy),
            "value_table": dict(result.values),
            "q_table": dict(result.q_values),
            "summary": dict(result.summary),
            "residual_history": list(result.residual_history),
            "collection_summary": {
                "transition_samples": self.collection.transition_samples,
                "unique_states": self.collection.unique_states,
                "states_seen_by_episode": list(self.collection.states_seen_by_episode),
                "samples_seen_by_episode": list(self.collection.samples_seen_by_episode),
                "exploratory_returns": list(self.collection.exploratory_returns),
                "exploratory_scores": list(self.collection.exploratory_scores),
                "exploratory_lengths": list(self.collection.exploratory_lengths),
            },
            "collection_config": dict(collection_config),
            "env_config": dict(env_config),
            "non_wall_coords": [tuple(map(int, coord)) for coord in non_wall_coords],
            "config": dict(raw_config),
        }


class FoodBitmaskValueIterationPolicy:
    """Greedy policy over food-bitmask aggregated observations with heuristic fallback."""

    def __init__(self, model_data: dict[str, Any], *, fallback_seed: int = 42) -> None:
        self.model_data = model_data
        self.policy_table = dict(model_data.get("policy_table", {}))
        self.unseen_states = 0
        self.rng = np.random.default_rng(int(fallback_seed))
        self.helper = FoodBitmaskHelper(model_data.get("non_wall_coords", []))
        collection_cfg = dict(model_data.get("collection_config", {}))
        self.fallback_danger_distance = int(collection_cfg.get("danger_distance", 2))

    def select_action(self, observation: dict[str, Any], info: dict[str, Any]) -> int:
        state_key = encode_food_bitmask_observation(observation)
        action = self.policy_table.get(state_key)
        legal_action_ids = [int(a) for a in info.get("legal_action_ids", [])]
        if action is not None and int(action) in legal_action_ids:
            return int(action)
        self.unseen_states += 1
        return self.helper.heuristic_action(
            observation,
            legal_action_ids,
            rng=self.rng,
            epsilon_random=0.0,
            danger_distance=self.fallback_danger_distance,
        )
