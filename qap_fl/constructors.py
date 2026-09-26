from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
from scipy.optimize import linear_sum_assignment

from .qap import QAPInstance, compute_cost, is_valid_perm
from .relaxation_features_v2 import _faq_soft_assignment, _spectral_cost_matrix


CONSTRUCTION_METHODS = ("random", "incremental_greedy", "strength_matching", "faq_relaxation")


@dataclass(frozen=True)
class ConstructedStart:
    method: str
    perm: np.ndarray
    cost: float
    runtime_sec: float


def random_start(instance: QAPInstance, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(int(seed)).permutation(instance.n).astype(np.int64)


def incremental_greedy_start(instance: QAPInstance) -> np.ndarray:
    """Assign high-flow facilities one at a time by exact partial QAP cost."""
    flow = np.asarray(instance.F, dtype=np.float64)
    distance = np.asarray(instance.D, dtype=np.float64)
    n = instance.n
    strength = np.sum(np.abs(flow), axis=0) + np.sum(np.abs(flow), axis=1)
    facility_order = np.argsort(-strength, kind="stable")
    perm = np.full(n, -1, dtype=np.int64)
    available = np.ones(n, dtype=bool)
    assigned: list[int] = []
    for facility in facility_order:
        locations = np.flatnonzero(available)
        partial_cost = flow[facility, facility] * distance[locations, locations]
        if assigned:
            other = np.asarray(assigned, dtype=np.int64)
            other_locations = perm[other]
            partial_cost = partial_cost + np.sum(
                flow[facility, other][None, :] * distance[np.ix_(locations, other_locations)]
                + flow[other, facility][None, :] * distance[np.ix_(other_locations, locations)].T,
                axis=1,
            )
        location = int(locations[int(np.argmin(partial_cost))])
        perm[int(facility)] = location
        available[location] = False
        assigned.append(int(facility))
    return perm


def strength_matching_start(instance: QAPInstance) -> np.ndarray:
    """Match high-interaction facilities to low-total-distance locations."""
    flow = np.asarray(instance.F, dtype=np.float64)
    distance = np.asarray(instance.D, dtype=np.float64)
    facility_strength = np.sum(np.abs(flow), axis=0) + np.sum(np.abs(flow), axis=1)
    location_remoteness = np.sum(np.abs(distance), axis=0) + np.sum(np.abs(distance), axis=1)
    facilities = np.argsort(-facility_strength, kind="stable")
    locations = np.argsort(location_remoteness, kind="stable")
    perm = np.empty(instance.n, dtype=np.int64)
    perm[facilities] = locations
    return perm


def faq_relaxation_start(instance: QAPInstance, faq_iters: int = 8) -> np.ndarray:
    """Round the project's deterministic FAQ-style doubly stochastic relaxation."""
    soft = _faq_soft_assignment(instance.F, instance.D, n_iters=int(faq_iters))
    rows, columns = linear_sum_assignment(-soft)
    perm = np.empty(instance.n, dtype=np.int64)
    perm[rows] = columns
    return perm


def spectral_relaxation_start(
    instance: QAPInstance,
    faq_iters: int = 8,
    spectral_dim: int = 4,
) -> np.ndarray:
    """Round a spectral alignment initialized by a FAQ soft assignment."""
    soft = _faq_soft_assignment(instance.F, instance.D, n_iters=int(faq_iters))
    cost = _spectral_cost_matrix(instance.F, instance.D, soft, dim=int(spectral_dim))
    rows, columns = linear_sum_assignment(cost)
    perm = np.empty(instance.n, dtype=np.int64)
    perm[rows] = columns
    return perm


def faq_spectral_portfolio(
    instance: QAPInstance,
    n_starts: int = 16,
    seed: int = 0,
) -> list[ConstructedStart]:
    """Build diverse deterministic FAQ/spectral starts, then fill duplicates locally."""
    started = time.perf_counter()
    candidates: list[tuple[str, np.ndarray]] = []
    for faq_iters in range(1, 9):
        candidates.append((f"faq_i{faq_iters}", faq_relaxation_start(instance, faq_iters=faq_iters)))
    for spectral_dim in range(1, 9):
        candidates.append(
            (
                f"spectral_d{spectral_dim}",
                spectral_relaxation_start(instance, faq_iters=8, spectral_dim=spectral_dim),
            )
        )

    unique: list[tuple[str, np.ndarray]] = []
    seen: set[tuple[int, ...]] = set()
    for label, perm in candidates:
        key = tuple(map(int, perm))
        if key not in seen:
            seen.add(key)
            unique.append((label, np.asarray(perm, dtype=np.int64).copy()))
        if len(unique) >= int(n_starts):
            break

    rng = np.random.default_rng(int(seed))
    base_count = len(unique)
    fill_index = 0
    while len(unique) < int(n_starts):
        label, base = unique[fill_index % max(base_count, 1)]
        perm = base.copy()
        swap_count = 1 + fill_index // max(base_count, 1)
        for _ in range(swap_count):
            i, j = rng.choice(instance.n, size=2, replace=False)
            perm[int(i)], perm[int(j)] = perm[int(j)], perm[int(i)]
        key = tuple(map(int, perm))
        fill_index += 1
        if key in seen:
            continue
        seen.add(key)
        unique.append((f"{label}_swap{swap_count}", perm))

    elapsed = time.perf_counter() - started
    per_start_runtime = elapsed / max(len(unique), 1)
    return [
        ConstructedStart(
            method=label,
            perm=perm,
            cost=compute_cost(perm, instance.F, instance.D),
            runtime_sec=per_start_runtime,
        )
        for label, perm in unique[: int(n_starts)]
    ]


def construct_start(
    instance: QAPInstance,
    method: str,
    seed: int = 0,
    faq_iters: int = 8,
) -> ConstructedStart:
    started = time.perf_counter()
    if method == "random":
        perm = random_start(instance, seed=seed)
    elif method == "incremental_greedy":
        perm = incremental_greedy_start(instance)
    elif method == "strength_matching":
        perm = strength_matching_start(instance)
    elif method == "faq_relaxation":
        perm = faq_relaxation_start(instance, faq_iters=faq_iters)
    else:
        raise ValueError(f"unknown construction method: {method}")
    runtime_sec = time.perf_counter() - started
    if not is_valid_perm(perm, instance.n):
        raise RuntimeError(f"{method} produced an invalid permutation")
    return ConstructedStart(
        method=method,
        perm=np.asarray(perm, dtype=np.int64),
        cost=compute_cost(perm, instance.F, instance.D),
        runtime_sec=float(runtime_sec),
    )
