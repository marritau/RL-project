"""Observation encoding helpers for aggregated-state DP."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np

DEFAULT_DROP_KEYS = {"score", "step_count"}


ObsKey = tuple[Any, ...]


def _encode_array(value: np.ndarray, float_round: int) -> tuple[str, tuple[int, ...], bytes]:
    """Convert numpy array into a stable, hashable representation."""
    array = np.asarray(value)
    if np.issubdtype(array.dtype, np.floating):
        array = np.round(array.astype(np.float32), float_round)
    return (str(array.dtype), tuple(int(dim) for dim in array.shape), array.tobytes())


def encode_observation(
    observation: dict[str, Any],
    *,
    drop_keys: Iterable[str] | None = None,
    float_round: int = 3,
) -> ObsKey:
    """Encode an observation dict into a compact, hashable key."""
    drop = set(DEFAULT_DROP_KEYS)
    if drop_keys is not None:
        drop.update(drop_keys)

    parts: list[Any] = []
    for key in sorted(k for k in observation.keys() if k not in drop):
        value = observation[key]
        if isinstance(value, np.ndarray):
            parts.append((key, *_encode_array(value, float_round)))
        elif isinstance(value, (list, tuple)):
            parts.append((key, *_encode_array(np.asarray(value), float_round)))
        else:
            if isinstance(value, float):
                value = round(value, float_round)
            parts.append((key, value))
    return tuple(parts)
