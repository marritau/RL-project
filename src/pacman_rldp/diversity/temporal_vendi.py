"""Temporal Vendi Score for Pacman trajectories.

The implementation follows the structure of the TVS paper:

1. sample trajectories from an RL agent;
2. compare ordered trajectories with a Global Alignment Kernel (GAK);
3. use an environment-grounded time-to-reach state cost;
4. normalize the similarity matrix and aggregate it with the q=2 Vendi score.

For this Pacman project we use the Pacman grid position as the state projection.
The exact time-to-reach distance is therefore the shortest-path length through the
layout walls. This is a faithful discrete-maze adaptation of the paper while
avoiding an intractable exact distance over full Pacman states (food map, ghosts,
scared timers, score, etc.).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import isfinite, log
from typing import Iterable, Sequence

import numpy as np

from .rollouts import GridPosition, Trajectory


@dataclass(frozen=True)
class GridTimeToReach:
    """All-pairs shortest-path time-to-reach distances on a Pacman layout."""

    positions: tuple[GridPosition, ...]
    distances: np.ndarray
    unreachable_distance: float

    @classmethod
    def from_walls(cls, walls: np.ndarray) -> "GridTimeToReach":
        """Precompute BFS distances between all non-wall cells.

        ``walls`` follows the project convention: shape ``(width, height)`` and
        ``1`` for walls. Moving one legal grid step costs one time unit.
        """
        walls_array = np.asarray(walls)
        if walls_array.ndim != 2:
            raise ValueError("walls must be a 2D array of shape (width, height).")
        width, height = int(walls_array.shape[0]), int(walls_array.shape[1])
        positions = tuple(
            (x_coord, y_coord)
            for x_coord in range(width)
            for y_coord in range(height)
            if int(walls_array[x_coord, y_coord]) == 0
        )
        index_by_position = {position: idx for idx, position in enumerate(positions)}
        n_positions = len(positions)
        distances = np.full((n_positions, n_positions), np.inf, dtype=np.float64)

        for source_idx, source in enumerate(positions):
            distances[source_idx, source_idx] = 0.0
            queue: deque[GridPosition] = deque([source])
            while queue:
                current = queue.popleft()
                current_idx = index_by_position[current]
                current_distance = distances[source_idx, current_idx]
                x_coord, y_coord = current
                for neighbor in (
                    (x_coord + 1, y_coord),
                    (x_coord - 1, y_coord),
                    (x_coord, y_coord + 1),
                    (x_coord, y_coord - 1),
                ):
                    nx, ny = neighbor
                    if not (0 <= nx < width and 0 <= ny < height):
                        continue
                    if int(walls_array[nx, ny]) == 1:
                        continue
                    neighbor_idx = index_by_position[neighbor]
                    if distances[source_idx, neighbor_idx] <= current_distance + 1.0:
                        continue
                    distances[source_idx, neighbor_idx] = current_distance + 1.0
                    queue.append(neighbor)

        finite = distances[np.isfinite(distances)]
        max_finite = float(np.max(finite)) if finite.size else 1.0
        unreachable_distance = max_finite + 1.0
        distances = np.where(np.isfinite(distances), distances, unreachable_distance)
        return cls(positions=positions, distances=distances, unreachable_distance=unreachable_distance)

    def lookup_matrix(
        self,
        trajectory_a: Sequence[GridPosition],
        trajectory_b: Sequence[GridPosition],
    ) -> np.ndarray:
        """Return pairwise shortest-path costs between two position sequences."""
        index_by_position = {position: idx for idx, position in enumerate(self.positions)}
        result = np.empty((len(trajectory_a), len(trajectory_b)), dtype=np.float64)
        for i, state_a in enumerate(trajectory_a):
            idx_a = index_by_position.get(state_a)
            for j, state_b in enumerate(trajectory_b):
                idx_b = index_by_position.get(state_b)
                if idx_a is None or idx_b is None:
                    result[i, j] = self.unreachable_distance
                else:
                    result[i, j] = self.distances[idx_a, idx_b]
        return result


def _logsumexp(values: Sequence[float]) -> float:
    finite_values = [value for value in values if isfinite(value)]
    if not finite_values:
        return -np.inf
    max_value = max(finite_values)
    return max_value + log(sum(float(np.exp(value - max_value)) for value in finite_values))


def global_alignment_log_kernel(
    distance_matrix: np.ndarray,
    *,
    sigma: float,
    band_ratio: float = 0.2,
) -> float:
    """Compute log GAK with a Sakoe-Chiba band.

    The recurrence is ``GA(i,j)=kappa(i,j)*(GA(i-1,j-1)+GA(i-1,j)+GA(i,j-1))``.
    We evaluate it in log-space for numerical stability. The default band ratio
    is 20%, as used in the paper.
    """
    if sigma <= 0.0:
        raise ValueError("sigma must be positive.")
    distances = np.asarray(distance_matrix, dtype=np.float64)
    if distances.ndim != 2:
        raise ValueError("distance_matrix must be 2D.")
    length_a, length_b = int(distances.shape[0]), int(distances.shape[1])
    if length_a == 0 or length_b == 0:
        return -np.inf

    band = int(np.ceil(float(band_ratio) * max(length_a, length_b)))
    band = max(0, band)
    dp = np.full((length_a + 1, length_b + 1), -np.inf, dtype=np.float64)
    dp[0, 0] = 0.0

    for i in range(1, length_a + 1):
        j_start = 1
        j_stop = length_b
        if band_ratio < 1.0:
            # Paper Eq. (Sakoe-Chiba restriction): |i - j| <= band.
            # Long trajectories whose endpoints are outside the band are treated
            # as very dissimilar because no valid alignment reaches dp[T, T'].
            j_start = max(1, i - band)
            j_stop = min(length_b, i + band)
        for j in range(j_start, j_stop + 1):
            previous = _logsumexp((dp[i - 1, j - 1], dp[i - 1, j], dp[i, j - 1]))
            if not isfinite(previous):
                continue
            log_kappa = -float(distances[i - 1, j - 1]) / float(sigma)
            dp[i, j] = log_kappa + previous
    return float(dp[length_a, length_b])


def calibrate_sigma(
    trajectories: Sequence[Sequence[GridPosition]],
    time_to_reach: GridTimeToReach,
) -> float:
    """Choose the GAK bandwidth using the paper's Appendix A.1 rule.

    The paper selects ``sigma`` so that a median trajectory pair has normalized
    similarity about 0.5: ``sigma = L * d_hat / ln(2)``, where ``L`` is the
    median trajectory length and ``d_hat`` is the median positive time-to-reach
    distance between sampled states from the rollout set.
    """
    non_empty = [list(trajectory) for trajectory in trajectories if trajectory]
    if not non_empty:
        return 1.0
    unique_states = sorted({state for trajectory in non_empty for state in trajectory})
    if len(unique_states) < 2:
        return 1.0
    matrix = time_to_reach.lookup_matrix(unique_states, unique_states)
    positive = matrix[(matrix > 0.0) & np.isfinite(matrix)]
    if positive.size == 0:
        return 1.0
    median_distance = float(np.median(positive))
    median_length = float(np.median([len(trajectory) for trajectory in non_empty]))
    return max(1e-6, float(median_length * median_distance / np.log(2.0)))


def _compress_consecutive_duplicates(states: Sequence[GridPosition]) -> list[GridPosition]:
    compressed: list[GridPosition] = []
    previous: GridPosition | None = None
    for state in states:
        if state != previous:
            compressed.append(state)
            previous = state
    return compressed


def _resample_sequence(states: Sequence[GridPosition], max_points: int | None) -> list[GridPosition]:
    if max_points is None or max_points <= 0 or len(states) <= max_points:
        return list(states)
    if max_points == 1:
        return [states[0]]
    indices = np.linspace(0, len(states) - 1, num=max_points)
    rounded = sorted({int(round(index)) for index in indices})
    return [states[index] for index in rounded]


def prepare_tvs_trajectories(
    trajectories: Iterable[Trajectory],
    *,
    max_points_per_trajectory: int | None = 160,
    compress_repeats: bool = False,
) -> list[list[GridPosition]]:
    """Convert rollout dataclasses to compact TVS state sequences."""
    prepared: list[list[GridPosition]] = []
    for trajectory in trajectories:
        states: list[GridPosition] = list(trajectory.states)
        if compress_repeats:
            states = _compress_consecutive_duplicates(states)
        states = _resample_sequence(states, max_points_per_trajectory)
        if states:
            prepared.append(states)
    return prepared


@dataclass
class TemporalVendiResult:
    """TVS computation output."""

    tvs: float
    similarity_matrix: np.ndarray
    eigenvalues: np.ndarray
    sigma: float
    band_ratio: float
    trajectory_count: int
    mean_prepared_length: float
    min_prepared_length: int
    max_prepared_length: int

    def metrics_dict(self) -> dict[str, float | int | list[float]]:
        return {
            "tvs": float(self.tvs),
            "sigma": float(self.sigma),
            "band_ratio": float(self.band_ratio),
            "trajectory_count": int(self.trajectory_count),
            "mean_prepared_length": float(self.mean_prepared_length),
            "min_prepared_length": int(self.min_prepared_length),
            "max_prepared_length": int(self.max_prepared_length),
            "eigenvalues": [float(value) for value in self.eigenvalues],
        }


def normalized_gak_similarity_matrix(
    trajectories: Sequence[Sequence[GridPosition]],
    time_to_reach: GridTimeToReach,
    *,
    sigma: float,
    band_ratio: float = 0.2,
) -> np.ndarray:
    """Build normalized trajectory similarity matrix K with K_ii = 1."""
    n_trajectories = len(trajectories)
    if n_trajectories == 0:
        raise ValueError("At least one trajectory is required.")
    log_kernel = np.full((n_trajectories, n_trajectories), -np.inf, dtype=np.float64)

    for i in range(n_trajectories):
        for j in range(i, n_trajectories):
            distances = time_to_reach.lookup_matrix(trajectories[i], trajectories[j])
            log_value = global_alignment_log_kernel(distances, sigma=sigma, band_ratio=band_ratio)
            log_kernel[i, j] = log_value
            log_kernel[j, i] = log_value

    similarity = np.eye(n_trajectories, dtype=np.float64)
    diagonal = np.diag(log_kernel)
    for i in range(n_trajectories):
        for j in range(i + 1, n_trajectories):
            log_normalized = log_kernel[i, j] - 0.5 * (diagonal[i] + diagonal[j])
            value = float(np.exp(log_normalized)) if isfinite(log_normalized) else 0.0
            # Tiny numerical overshoots can happen after normalization.
            value = min(1.0, max(0.0, value))
            similarity[i, j] = value
            similarity[j, i] = value
    return similarity


def temporal_vendi_score_from_similarity(similarity_matrix: np.ndarray) -> tuple[float, np.ndarray]:
    """Compute q=2 Vendi score from a normalized similarity matrix.

    For q=2, the generalized Vendi score is the inverse Simpson index:
    ``1 / sum(lambda_i^2)``, where eigenvalues are taken from ``K / N``.
    """
    matrix = np.asarray(similarity_matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("similarity_matrix must be square.")
    n_items = int(matrix.shape[0])
    if n_items == 0:
        raise ValueError("similarity_matrix cannot be empty.")
    eigenvalues = np.linalg.eigvalsh(matrix / float(n_items))
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    total = float(np.sum(eigenvalues))
    if total <= 0.0:
        return 0.0, eigenvalues
    eigenvalues = eigenvalues / total
    denominator = float(np.sum(eigenvalues**2))
    score = 0.0 if denominator <= 0.0 else 1.0 / denominator
    return float(score), eigenvalues


def compute_temporal_vendi_score(
    trajectories: Sequence[Trajectory],
    time_to_reach: GridTimeToReach,
    *,
    sigma: float | None = None,
    band_ratio: float = 0.2,
    max_points_per_trajectory: int | None = 160,
    compress_repeats: bool = False,
) -> TemporalVendiResult:
    """Compute TVS for a collection of Pacman rollouts."""
    prepared = prepare_tvs_trajectories(
        trajectories,
        max_points_per_trajectory=max_points_per_trajectory,
        compress_repeats=compress_repeats,
    )
    if not prepared:
        raise ValueError("No non-empty trajectories available for TVS.")
    if sigma is None:
        sigma = calibrate_sigma(prepared, time_to_reach)
    similarity_matrix = normalized_gak_similarity_matrix(
        prepared,
        time_to_reach,
        sigma=float(sigma),
        band_ratio=float(band_ratio),
    )
    tvs, eigenvalues = temporal_vendi_score_from_similarity(similarity_matrix)
    lengths = [len(trajectory) for trajectory in prepared]
    return TemporalVendiResult(
        tvs=float(tvs),
        similarity_matrix=similarity_matrix,
        eigenvalues=eigenvalues,
        sigma=float(sigma),
        band_ratio=float(band_ratio),
        trajectory_count=len(prepared),
        mean_prepared_length=float(np.mean(lengths)),
        min_prepared_length=int(min(lengths)),
        max_prepared_length=int(max(lengths)),
    )


def prefix_temporal_vendi_scores(similarity_matrix: np.ndarray) -> dict[int, float]:
    """Compute TVS values for trajectory prefixes from an existing K matrix."""
    matrix = np.asarray(similarity_matrix, dtype=np.float64)
    n_items = int(matrix.shape[0])
    prefix_sizes = [4, 8, 16, 32, 64, 128, 256]
    result: dict[int, float] = {}
    for size in prefix_sizes:
        if size <= n_items:
            score, _ = temporal_vendi_score_from_similarity(matrix[:size, :size])
            result[int(size)] = float(score)
    if n_items not in result:
        score, _ = temporal_vendi_score_from_similarity(matrix)
        result[n_items] = float(score)
    return result


def state_coverage(trajectories: Iterable[Trajectory]) -> int:
    """Number of unique Pacman positions visited by the trajectory set."""
    return len({state for trajectory in trajectories for state in trajectory.states})


def occupancy_entropy(trajectories: Iterable[Trajectory]) -> float:
    """Shannon entropy of aggregate Pacman-position visitation counts."""
    counts: dict[GridPosition, int] = {}
    total = 0
    for trajectory in trajectories:
        for state in trajectory.states:
            counts[state] = counts.get(state, 0) + 1
            total += 1
    if total == 0:
        return 0.0
    probabilities = np.array([count / total for count in counts.values()], dtype=np.float64)
    return float(-np.sum(probabilities * np.log(probabilities + 1e-12)))


def action_entropy(trajectories: Iterable[Trajectory]) -> float:
    """Shannon entropy of aggregate action usage."""
    counts: dict[int, int] = {}
    total = 0
    for trajectory in trajectories:
        for action in trajectory.actions:
            counts[action] = counts.get(action, 0) + 1
            total += 1
    if total == 0:
        return 0.0
    probabilities = np.array([count / total for count in counts.values()], dtype=np.float64)
    return float(-np.sum(probabilities * np.log(probabilities + 1e-12)))
