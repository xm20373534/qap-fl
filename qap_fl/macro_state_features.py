"""Deployment-visible state summaries for macro-action control.

The functions in this module deliberately operate on already available search
state.  They do not inspect future oracle outcomes, so the same feature
contract can be used by trajectory logging, offline training, and an online
controller.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np


EPS = 1e-12


def _summary(values: Sequence[float], prefix: str) -> dict[str, float]:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        ordered = [0.0]

    def quantile(fraction: float) -> float:
        position = fraction * (len(ordered) - 1)
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        weight = position - lower
        return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

    return {
        f"{prefix}_min": ordered[0],
        f"{prefix}_mean": float(sum(ordered) / len(ordered)),
        f"{prefix}_max": ordered[-1],
        f"{prefix}_q25": quantile(0.25),
        f"{prefix}_q75": quantile(0.75),
    }


def hamming_distance_features(
    perm: np.ndarray,
    elite_perms: Iterable[np.ndarray] | None,
    *,
    incumbent_perm: np.ndarray | None = None,
) -> dict[str, float]:
    """Return normalized Hamming-distance summaries to an elite pool.

    ``elite_perms`` may be empty during the first outer iteration.  In that
    case the summary is zero and the count feature records that no guide was
    available.  Distances are normalized by ``n`` and therefore transfer
    across instance sizes.
    """

    current = np.asarray(perm)
    n = max(int(current.size), 1)
    elites = [np.asarray(item) for item in (elite_perms or [])]
    distances = [float(np.count_nonzero(current != item)) / n for item in elites]
    output = _summary(distances, "state_hamming_elite")
    output["state_hamming_elite_count"] = float(len(elites))
    if incumbent_perm is None:
        output["state_hamming_incumbent"] = 0.0
    else:
        incumbent = np.asarray(incumbent_perm)
        output["state_hamming_incumbent"] = float(
            np.count_nonzero(current != incumbent) / n
        )
    return output


def decayed_macro_history_features(
    history: Iterable[tuple[str, float]] | None,
    actions: Sequence[str],
    *,
    decay: float = 0.85,
    max_events: int = 10,
) -> dict[str, float]:
    """Summarize the most recent macro utilities with exponential decay."""

    if not 0.0 < float(decay) <= 1.0:
        raise ValueError("decay must be in (0, 1]")
    events = list(history or [])[-int(max_events) :]
    output: dict[str, float] = {
        "state_macro_history_count": float(len(events)),
        "state_macro_history_decayed_mean": 0.0,
        "state_macro_history_last_utility": 0.0,
    }
    if not events:
        for action in actions:
            output[f"state_macro_history_{action}_utility"] = 0.0
            output[f"state_macro_history_{action}_count"] = 0.0
        return output
    weights = [float(decay) ** (len(events) - 1 - index) for index in range(len(events))]
    utilities = [float(item[1]) for item in events]
    output["state_macro_history_decayed_mean"] = float(
        sum(weight * utility for weight, utility in zip(weights, utilities))
        / max(sum(weights), EPS)
    )
    output["state_macro_history_last_utility"] = utilities[-1]
    for action in actions:
        selected = [float(value) for name, value in events if str(name) == str(action)]
        output[f"state_macro_history_{action}_utility"] = float(
            sum(selected) / len(selected) if selected else 0.0
        )
        output[f"state_macro_history_{action}_count"] = float(len(selected))
    return output


def basin_depth_features(
    recent_gains: Iterable[float] | None,
    *,
    decay: float = 0.85,
    max_events: int = 10,
) -> dict[str, float]:
    """Summarize recent positive descent improvements as basin depth."""

    if not 0.0 < float(decay) <= 1.0:
        raise ValueError("decay must be in (0, 1]")
    gains = [max(float(value), 0.0) for value in (recent_gains or [])]
    gains = gains[-int(max_events) :]
    output = _summary(gains, "state_basin_depth")
    output["state_basin_depth_count"] = float(len(gains))
    if gains:
        weights = [float(decay) ** (len(gains) - 1 - index) for index in range(len(gains))]
        output["state_basin_depth_decayed"] = float(
            sum(weight * value for weight, value in zip(weights, gains))
            / max(sum(weights), EPS)
        )
        output["state_basin_depth_last"] = gains[-1]
    else:
        output["state_basin_depth_decayed"] = 0.0
        output["state_basin_depth_last"] = 0.0
    return output


__all__ = [
    "basin_depth_features",
    "decayed_macro_history_features",
    "hamming_distance_features",
]
