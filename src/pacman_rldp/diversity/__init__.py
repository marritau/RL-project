"""Trajectory-diversity evaluation utilities for Pacman RLDP.

This package implements a Pacman-grid adaptation of Temporal Vendi Score (TVS):
rollouts are compared as ordered sequences of Pacman grid positions, pairwise
state costs are shortest-path time-to-reach distances through the maze, trajectory
similarity is computed with a banded Global Alignment Kernel, and the final score
is the q=2 Vendi score of the normalized similarity matrix.
"""

from .policies import FoodBitmaskHeuristicPolicy, NoisyPolicy, TabularQGreedyPolicy
from .rollouts import Trajectory, collect_trajectories, filter_trajectories
from .temporal_vendi import (
    GridTimeToReach,
    TemporalVendiResult,
    compute_temporal_vendi_score,
    occupancy_entropy,
    action_entropy,
    prefix_temporal_vendi_scores,
    state_coverage,
    temporal_vendi_score_from_similarity,
)

__all__ = [
    "GridTimeToReach",
    "FoodBitmaskHeuristicPolicy",
    "NoisyPolicy",
    "TabularQGreedyPolicy",
    "TemporalVendiResult",
    "Trajectory",
    "collect_trajectories",
    "compute_temporal_vendi_score",
    "filter_trajectories",
    "occupancy_entropy",
    "action_entropy",
    "prefix_temporal_vendi_scores",
    "state_coverage",
    "temporal_vendi_score_from_similarity",
]
