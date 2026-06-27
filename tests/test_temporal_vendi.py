import numpy as np

from pacman_rldp.diversity.rollouts import Trajectory
from pacman_rldp.diversity.temporal_vendi import (
    GridTimeToReach,
    calibrate_sigma,
    compute_temporal_vendi_score,
    global_alignment_log_kernel,
    normalized_gak_similarity_matrix,
    temporal_vendi_score_from_similarity,
)


def test_vendi_score_is_one_for_identical_items():
    similarity = np.ones((4, 4), dtype=float)
    score, eigenvalues = temporal_vendi_score_from_similarity(similarity)

    assert np.isclose(score, 1.0)
    assert np.isclose(np.sum(eigenvalues), 1.0)


def test_vendi_score_equals_n_for_orthogonal_items():
    similarity = np.eye(4, dtype=float)
    score, eigenvalues = temporal_vendi_score_from_similarity(similarity)

    assert np.isclose(score, 4.0)
    assert np.allclose(eigenvalues, np.full(4, 0.25))


def test_grid_time_to_reach_respects_walls():
    walls = np.zeros((3, 3), dtype=int)
    walls[1, 1] = 1
    time_to_reach = GridTimeToReach.from_walls(walls)

    # Direct horizontal movement is blocked by the center wall, so the shortest
    # path must go around it: (0,1)->(0,0)->(1,0)->(2,0)->(2,1).
    distance = time_to_reach.lookup_matrix([(0, 1)], [(2, 1)])[0, 0]

    assert distance == 4.0



def test_gak_uses_paper_style_absolute_sakoe_chiba_band():
    # With |i-j| <= ceil(0.2 * max(3, 5)) = 1, a 3-step trajectory
    # cannot align to a 5-step trajectory endpoint, so the banded kernel is zero.
    # This tests the paper's literal band rule rather than a length-normalized band.
    distances = np.zeros((3, 5), dtype=float)
    log_value = global_alignment_log_kernel(distances, sigma=1.0, band_ratio=0.2)

    assert np.isneginf(log_value)


def test_sigma_calibration_matches_paper_appendix_rule():
    walls = np.zeros((3, 3), dtype=int)
    time_to_reach = GridTimeToReach.from_walls(walls)
    trajectories = [
        [(0, 0), (1, 0), (2, 0)],
        [(0, 1), (1, 1), (2, 1)],
    ]
    unique_states = sorted({state for trajectory in trajectories for state in trajectory})
    pairwise = time_to_reach.lookup_matrix(unique_states, unique_states)
    d_hat = np.median(pairwise[pairwise > 0.0])
    expected = 3 * d_hat / np.log(2.0)

    assert np.isclose(calibrate_sigma(trajectories, time_to_reach), expected)

def test_normalized_gak_similarity_is_highest_for_identical_trajectories():
    walls = np.zeros((3, 3), dtype=int)
    walls[1, 1] = 1
    time_to_reach = GridTimeToReach.from_walls(walls)

    lower_route = [(0, 0), (1, 0), (2, 0)]
    same_lower_route = [(0, 0), (1, 0), (2, 0)]
    upper_route = [(0, 2), (1, 2), (2, 2)]

    similarity = normalized_gak_similarity_matrix(
        [lower_route, same_lower_route, upper_route],
        time_to_reach,
        sigma=1.0,
        band_ratio=1.0,
    )

    assert np.isclose(similarity[0, 1], 1.0)
    assert similarity[0, 2] < similarity[0, 1]
    assert np.allclose(np.diag(similarity), 1.0)


def test_compute_temporal_vendi_score_end_to_end():
    walls = np.zeros((3, 3), dtype=int)
    walls[1, 1] = 1
    time_to_reach = GridTimeToReach.from_walls(walls)
    routes = [
        [(0, 0), (1, 0), (2, 0)],
        [(0, 0), (1, 0), (2, 0)],
        [(0, 2), (1, 2), (2, 2)],
    ]
    trajectories = [
        Trajectory(
            states=list(route),
            actions=[0, 0],
            rewards=[1.0, 1.0],
            total_return=2.0,
            score=2.0,
            win=True,
            lose=False,
            truncated=False,
            seed=idx,
        )
        for idx, route in enumerate(routes)
    ]

    result = compute_temporal_vendi_score(
        trajectories,
        time_to_reach,
        sigma=1.0,
        band_ratio=1.0,
        max_points_per_trajectory=None,
    )

    assert 1.0 <= result.tvs <= 3.0
    assert result.trajectory_count == 3
    assert result.similarity_matrix.shape == (3, 3)
