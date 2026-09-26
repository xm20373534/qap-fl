from __future__ import annotations

import numpy as np


POOL_CONTEXT_BASE_FIELDS = [
    "feature_anchor_external_delta_estimate",
    "feature_anchor_delta_mean",
    "feature_anchor_delta_min",
    "feature_anchor_delta_max",
    "feature_node_contrib_sum",
    "feature_node_contrib_absdiff",
    "feature_facility_row_l1",
    "feature_location_row_l1",
]


def pool_context_fields(base_fields: list[str]) -> list[str]:
    missing = [field for field in POOL_CONTEXT_BASE_FIELDS if field not in base_fields]
    if missing:
        raise ValueError(f"pool-context base fields are missing: {missing}")
    added: list[str] = []
    for field in POOL_CONTEXT_BASE_FIELDS:
        added.extend([f"{field}_pool_z", f"{field}_pool_percentile"])
    return list(base_fields) + added


def _average_percentile(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if len(values) <= 1:
        return np.zeros(len(values), dtype=np.float32)
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    starts = np.cumsum(counts) - counts
    average_ranks = starts + 0.5 * (counts - 1)
    return (average_ranks[inverse] / float(len(values) - 1)).astype(np.float32)


def add_pool_context(features: np.ndarray, base_fields: list[str]) -> np.ndarray:
    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 2 or features.shape[1] != len(base_fields):
        raise ValueError("features must have shape [pool_size, len(base_fields)]")
    additions: list[np.ndarray] = []
    for field in POOL_CONTEXT_BASE_FIELDS:
        values = features[:, base_fields.index(field)].astype(np.float64)
        std = float(np.std(values))
        z = np.zeros(len(values), dtype=np.float32)
        if std >= 1e-6:
            z = ((values - float(np.mean(values))) / std).astype(np.float32)
        additions.extend([z, _average_percentile(values)])
    return np.concatenate([features, np.column_stack(additions)], axis=1).astype(np.float32)


__all__ = ["POOL_CONTEXT_BASE_FIELDS", "add_pool_context", "pool_context_fields"]
