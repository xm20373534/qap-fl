"""Exact relabeling transformations for QAP training augmentation."""

from __future__ import annotations

import numpy as np

from .qap import QAPInstance, is_valid_perm


def _validate_map(mapping: np.ndarray, n: int) -> np.ndarray:
    mapping = np.asarray(mapping, dtype=np.int64)
    if not is_valid_perm(mapping, n):
        raise ValueError("label map must be a permutation of 0..n-1")
    return mapping


def relabel_instance(
    instance: QAPInstance,
    facility_map: np.ndarray,
    location_map: np.ndarray,
) -> QAPInstance:
    """Relabel facility and location indices while preserving the QAP."""
    n = instance.n
    facility_map = _validate_map(facility_map, n)
    location_map = _validate_map(location_map, n)
    flow = np.empty_like(instance.F)
    distance = np.empty_like(instance.D)
    flow[np.ix_(facility_map, facility_map)] = instance.F
    distance[np.ix_(location_map, location_map)] = instance.D
    return QAPInstance(
        name=f"{instance.name}__sym",
        F=flow,
        D=distance,
        optimum=instance.optimum,
    )


def relabel_solution(
    perm: np.ndarray,
    facility_map: np.ndarray,
    location_map: np.ndarray,
) -> np.ndarray:
    """Map an assignment and return the assignment in relabeled indices."""
    perm = np.asarray(perm, dtype=np.int64)
    n = len(perm)
    facility_map = _validate_map(facility_map, n)
    location_map = _validate_map(location_map, n)
    if not is_valid_perm(perm, n):
        raise ValueError("perm must be a valid permutation")
    transformed = np.empty_like(perm)
    transformed[facility_map] = location_map[perm]
    return transformed


def relabel_swap(
    pair: tuple[int, int] | np.ndarray,
    facility_map: np.ndarray,
) -> tuple[int, int]:
    """Map a facility-indexed swap candidate to relabeled indices."""
    pair = np.asarray(pair, dtype=np.int64).reshape(-1)
    if len(pair) != 2:
        raise ValueError("swap pair must contain two indices")
    facility_map = _validate_map(facility_map, len(facility_map))
    a, b = int(facility_map[int(pair[0])]), int(facility_map[int(pair[1])])
    return (min(a, b), max(a, b))

