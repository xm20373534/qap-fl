"""A direct Python reproduction of the official Benlic--Hao BLS-QAP code.

The implementation deliberately keeps the official control flow instead of
reusing the project-specific pruning BLS in :mod:`mlp_pruning.bls`:

* full best-improvement 2-swap descent;
* Taillard delta initialization/update;
* the official ``iter_without_improvement`` jump rule;
* directed, recency-based, and random perturbations.

The C++ source uses ``rand`` and a non-standard Fisher--Yates initialization.
The Python version exposes the same operations through an explicit RNG object
so experiments are reproducible, while keeping all search decisions identical
to the source algorithm.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
import ctypes
import math
import os
import time

import numpy as np
import torch

from .features import SwapFeatureCache, all_swap_pairs, build_swap_feature_cache, sample_swap_pairs
from .global_features import build_swap_features_for_fields
from .local_search import SelectorBundle
from .macro_state_features import basin_depth_features, hamming_distance_features
from .qap import QAPInstance, compute_cost


INFINITE = 999_999_999
EPS = 1e-12


class CStdRand:
    """Small wrapper around the platform C ``rand`` when available.

    MinGW's Windows build used by the supplied official executable resolves
    ``rand`` through the Microsoft C runtime.  Falling back to the familiar
    MSVC recurrence keeps deterministic behavior on environments without that
    DLL, although exact bit-for-bit matching is platform-dependent by design.
    """

    def __init__(self, seed: int = 1) -> None:
        self._lib = None
        self._state = int(seed) & 0x7FFFFFFF
        if os.name == "nt":
            try:
                lib = ctypes.CDLL("msvcrt.dll")
                lib.srand.argtypes = [ctypes.c_uint]
                lib.rand.restype = ctypes.c_int
                lib.srand(ctypes.c_uint(int(seed)))
                self._lib = lib
            except OSError:
                self._lib = None

    def rand(self) -> int:
        if self._lib is not None:
            return int(self._lib.rand())
        self._state = (214013 * self._state + 2_531_011) & 0x7FFFFFFF
        return (self._state >> 16) & 0x7FFF

    def uniform_int(self, low: int, high: int) -> int:
        if high < low:
            raise ValueError("high must be >= low")
        return int(low) + self.rand() % (int(high) - int(low) + 1)

    def unit_101(self) -> float:
        # This intentionally matches C++: rand()%101 / 100.0.
        return float(self.rand() % 101) / 100.0


@dataclass(frozen=True)
class OfficialBLSResult:
    perm: np.ndarray
    cost: int
    initial_cost: int
    runtime_sec: float
    n_outer_iters: int
    n_moves: int
    n_descent_decisions: int
    n_descent_moves: int
    n_perturb_moves: int
    n_delta_evals: int
    n_score_evals: int
    n_certification_checks: int
    n_certification_moves: int
    n_relink_calls: int
    n_relink_moves: int
    n_d28_relink_calls: int
    n_d28_path_moves: int
    n_d28_internal_moves: int
    n_d28_descent_moves: int
    n_d28_pool_rejections: int
    n_d28_relink_rejections: int
    n_directed_moves: int
    n_recency_moves: int
    n_random_moves: int
    n_noop_moves: int
    n_macro_official_actions: int
    n_macro_tabu_actions: int
    mean_relative_incumbent_gain: float
    macro_controller_overhead_sec: float
    n_macro_controller_updates: int
    n_macro_controller_switches: int
    hit_target: bool
    seed: int
    method: str
    delta_update_mode: str


@dataclass(frozen=True)
class OfficialBLSStateTrace:
    outer_iteration: int
    descent_step: int
    state_source: str
    perm: np.ndarray
    current_cost: int
    best_cost: int
    iteration: int
    iter_without_improvement: int
    perturb_strength: float
    last_swapped: np.ndarray
    best_perm: np.ndarray | None = None
    outer_previous_cost: int | None = None
    outer_descent_num: int = 0
    learned_decisions: int = 0
    rescue_used: bool = False


def _integer_matrices(instance: QAPInstance) -> tuple[np.ndarray, np.ndarray]:
    # QAPLIB data are integral.  The official code uses long arithmetic.
    flow = np.asarray(instance.F, dtype=np.int64)
    distance = np.asarray(instance.D, dtype=np.int64)
    if flow.shape != distance.shape or flow.ndim != 2 or flow.shape[0] != flow.shape[1]:
        raise ValueError("QAP matrices must be square and have matching shapes.")
    return flow, distance


def _compute_delta(perm: np.ndarray, flow: np.ndarray, distance: np.ndarray, i: int, j: int) -> int:
    """Exact 0-based translation of the official ``compute_delta``."""
    pi = int(perm[i])
    pj = int(perm[j])
    value = (
        (int(flow[i, i]) - int(flow[j, j]))
        * (int(distance[pj, pj]) - int(distance[pi, pi]))
        + (int(flow[i, j]) - int(flow[j, i]))
        * (int(distance[pj, pi]) - int(distance[pi, pj]))
    )
    for k in range(len(perm)):
        if k == i or k == j:
            continue
        value += (
            (int(flow[k, i]) - int(flow[k, j]))
            * (int(distance[int(perm[k]), pj]) - int(distance[int(perm[k]), pi]))
            + (int(flow[i, k]) - int(flow[j, k]))
            * (int(distance[pj, int(perm[k])]) - int(distance[pi, int(perm[k])]))
        )
    return int(value)


def _compute_deltas_for_pairs(
    perm: np.ndarray,
    flow: np.ndarray,
    distance: np.ndarray,
    pairs: np.ndarray,
) -> np.ndarray:
    """Vectorized exact swap deltas for an arbitrary list of pairs."""
    pairs = np.asarray(pairs, dtype=np.int64)
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError("pairs must have shape [num_pairs, 2]")
    if len(pairs) == 0:
        return np.zeros(0, dtype=np.int64)
    i = pairs[:, 0]
    j = pairs[:, 1]
    p_i = perm[i]
    p_j = perm[j]
    values = (
        (flow[i, i] - flow[j, j]) * (distance[p_j, p_j] - distance[p_i, p_i])
        + (flow[i, j] - flow[j, i]) * (distance[p_j, p_i] - distance[p_i, p_j])
    )
    k = np.arange(len(perm), dtype=np.int64)
    mask = (k[None, :] != i[:, None]) & (k[None, :] != j[:, None])
    incoming = (flow[:, i].T - flow[:, j].T) * (
        distance[perm[None, :], p_j[:, None]] - distance[perm[None, :], p_i[:, None]]
    )
    outgoing = (flow[i, :] - flow[j, :]) * (
        distance[p_j[:, None], perm[None, :]] - distance[p_i[:, None], perm[None, :]]
    )
    values = values + np.where(mask, incoming + outgoing, 0).sum(axis=1)
    return np.asarray(values, dtype=np.int64)


def _initialize_delta_scalar(perm: np.ndarray, flow: np.ndarray, distance: np.ndarray) -> np.ndarray:
    n = len(perm)
    delta = np.zeros((n, n), dtype=np.int64)
    for i in range(n - 1):
        for j in range(i + 1, n):
            value = _compute_delta(perm, flow, distance, i, j)
            delta[i, j] = value
            delta[j, i] = value
    np.fill_diagonal(delta, 0)
    return delta


def _initialize_delta(perm: np.ndarray, flow: np.ndarray, distance: np.ndarray) -> np.ndarray:
    """Vectorized equivalent of the official Taillard delta initialization."""
    n = len(perm)
    assigned_distance = distance[np.ix_(perm, perm)]
    flow_diag = np.diag(flow).astype(np.int64)
    distance_diag = np.diag(assigned_distance).astype(np.int64)
    delta = (flow_diag[:, None] - flow_diag[None, :]) * (distance_diag[None, :] - distance_diag[:, None])
    delta += (flow - flow.T) * (assigned_distance.T - assigned_distance)
    index = np.arange(n)
    first = (flow[:, :, None] - flow[:, None, :]) * (
        assigned_distance[:, None, :] - assigned_distance[:, :, None]
    )
    second = (flow[:, None, :] - flow[None, :, :]) * (
        assigned_distance[None, :, :] - assigned_distance[:, None, :]
    )
    first_mask = (index[:, None, None] != index[None, :, None]) & (index[:, None, None] != index[None, None, :])
    second_mask = (index[None, None, :] != index[:, None, None]) & (index[None, None, :] != index[None, :, None])
    delta += np.where(first_mask, first, 0).sum(axis=0) + np.where(second_mask, second, 0).sum(axis=2)
    np.fill_diagonal(delta, 0)
    return np.asarray(delta, dtype=np.int64)


def _compute_delta_part(
    perm: np.ndarray,
    flow: np.ndarray,
    distance: np.ndarray,
    delta_before: np.ndarray,
    i: int,
    j: int,
    r: int,
    s: int,
) -> int:
    p_i, p_j, p_r, p_s = (int(perm[i]), int(perm[j]), int(perm[r]), int(perm[s]))
    value = int(delta_before[i, j])
    value += (
        (int(flow[r, i]) - int(flow[r, j]) + int(flow[s, j]) - int(flow[s, i]))
        * (int(distance[p_s, p_i]) - int(distance[p_s, p_j]) + int(distance[p_r, p_j]) - int(distance[p_r, p_i]))
    )
    value += (
        (int(flow[i, r]) - int(flow[j, r]) + int(flow[j, s]) - int(flow[i, s]))
        * (int(distance[p_i, p_s]) - int(distance[p_j, p_s]) + int(distance[p_j, p_r]) - int(distance[p_i, p_r]))
    )
    return int(value)


def _update_delta(
    delta: np.ndarray,
    perm: np.ndarray,
    flow: np.ndarray,
    distance: np.ndarray,
    r: int,
    s: int,
    mode: str = "incremental",
) -> None:
    if mode == "rebuild":
        delta[:, :] = _initialize_delta(perm, flow, distance)
        return
    if mode != "incremental":
        raise ValueError("delta update mode must be rebuild or incremental")

    # perm is the state after swapping r and s. This is the vectorized form of
    # Taillard's O(1) correction for every pair not incident to r or s.
    p_r = int(perm[r])
    p_s = int(perm[s])
    flow_out = flow[r, :] - flow[s, :]
    distance_out = distance[p_s, perm] - distance[p_r, perm]
    flow_in = flow[:, r] - flow[:, s]
    distance_in = distance[perm, p_s] - distance[perm, p_r]
    delta += (
        (flow_out[:, None] - flow_out[None, :])
        * (distance_out[:, None] - distance_out[None, :])
        + (flow_in[:, None] - flow_in[None, :])
        * (distance_in[:, None] - distance_in[None, :])
    )

    n = len(perm)
    others = np.asarray([k for k in range(n) if k != r and k != s], dtype=np.int64)
    incident_pairs = np.vstack(
        [
            np.column_stack((np.minimum(r, others), np.maximum(r, others))),
            np.column_stack((np.minimum(s, others), np.maximum(s, others))),
            np.asarray([[min(r, s), max(r, s)]], dtype=np.int64),
        ]
    )
    incident_values = _compute_deltas_for_pairs(perm, flow, distance, incident_pairs)
    delta[incident_pairs[:, 0], incident_pairs[:, 1]] = incident_values
    delta[incident_pairs[:, 1], incident_pairs[:, 0]] = incident_values
    np.fill_diagonal(delta, 0)


def _official_random_perm(n: int, rng: CStdRand) -> np.ndarray:
    """Official Fisher--Yates variant translated from the 1-based C++ array."""
    perm = np.arange(n, dtype=np.int64)
    for i in range(0, n - 1):
        j = i + rng.rand() % (n - i)
        perm[i], perm[j] = perm[j], perm[i]
    return perm


def _best_improvement_move(
    instance: QAPInstance,
    perm: np.ndarray,
    delta: np.ndarray,
    current_cost: int,
    best_cost: int,
    best_perm: np.ndarray,
    last_swapped: np.ndarray,
    flow: np.ndarray,
    distance: np.ndarray,
    iter_without_improvement: int,
    iteration: int,
    method: str,
    candidate_rng: np.random.Generator,
    candidate_swaps: int | None,
    score_pool_swaps: int | None,
    selector: SelectorBundle | None,
    pair_score_callback: Callable[[QAPInstance, np.ndarray, np.ndarray, SwapFeatureCache], np.ndarray] | None,
    feature_cache: SwapFeatureCache | None,
    score_learned: bool = True,
    delta_update_mode: str = "rebuild",
    certification_allowed: bool = True,
    learned_variant: str = "plain",
    learned_random_rescue: int = 0,
    learned_delta_blend: float = 0.5,
    learned_rank_bias: float = 1.0,
) -> tuple[bool, int, int, int, int, int, int, bool, bool]:
    n = len(perm)
    if method == "full":
        candidate_pairs = all_swap_pairs(n)
        score_evals = 0
    else:
        pool = sample_swap_pairs(n, score_pool_swaps, candidate_rng)
        k = len(pool) if candidate_swaps is None else min(int(candidate_swaps), len(pool))
        if k <= 0:
            raise ValueError("candidate_swaps must be positive or None")
        if method in {"random", "random_certified", "random_one_rescue"}:
            if k >= len(pool):
                candidate_pairs = pool[candidate_rng.permutation(len(pool))]
            else:
                candidate_pairs = pool[candidate_rng.choice(len(pool), size=k, replace=False)]
            score_evals = 0
        elif method == "learned":
            if (selector is None and pair_score_callback is None) or feature_cache is None:
                raise ValueError("learned official BLS requires a selector or pair score callback")
            if not score_learned:
                if k >= len(pool):
                    candidate_pairs = pool[candidate_rng.permutation(len(pool))]
                else:
                    candidate_pairs = pool[candidate_rng.choice(len(pool), size=k, replace=False)]
                score_evals = 0
            else:
                if pair_score_callback is not None:
                    scores = np.asarray(
                        pair_score_callback(instance, perm, pool, feature_cache),
                        dtype=np.float64,
                    )
                else:
                    features = build_swap_features_for_fields(
                        instance=instance,
                        perm=perm,
                        pairs=pool,
                        feature_fields=selector.feature_fields,
                        feature_cache=feature_cache,
                    )
                    if selector.numpy_model is not None:
                        scores = selector.numpy_model(features)
                    else:
                        features = (features - selector.feature_mean) / selector.feature_std
                        device = next(selector.model.parameters()).device
                        selector.model.eval()
                        with torch.no_grad():
                            scores = selector.model(
                                torch.from_numpy(features).to(device=device, dtype=torch.float32)
                            ).detach().cpu().numpy().astype(np.float64)
                if scores.shape != (len(pool),) or not np.all(np.isfinite(scores)):
                    raise ValueError("pair score callback returned invalid scores")
                if learned_variant == "plain":
                    ranking_score = scores
                    if k >= len(pool):
                        selected = np.argsort(-ranking_score)
                    else:
                        selected = np.argpartition(-ranking_score, kth=k - 1)[:k]
                        selected = selected[np.argsort(-ranking_score[selected])]
                elif learned_variant == "random_rescue":
                    rescue = min(max(int(learned_random_rescue), 0), k)
                    model_k = k - rescue
                    if model_k:
                        if model_k >= len(pool):
                            model_selected = np.argsort(-scores)
                        else:
                            model_selected = np.argpartition(-scores, kth=model_k - 1)[:model_k]
                            model_selected = model_selected[np.argsort(-scores[model_selected])]
                    else:
                        model_selected = np.empty(0, dtype=np.int64)
                    remaining = np.setdiff1d(np.arange(len(pool), dtype=np.int64), model_selected, assume_unique=False)
                    if rescue and len(remaining):
                        rescue_selected = remaining[candidate_rng.choice(len(remaining), size=min(rescue, len(remaining)), replace=False)]
                    else:
                        rescue_selected = np.empty(0, dtype=np.int64)
                    selected = np.concatenate([model_selected, rescue_selected])
                    if len(selected) < k:
                        missing = np.setdiff1d(np.arange(len(pool), dtype=np.int64), selected, assume_unique=False)
                        selected = np.concatenate([selected, missing[: k - len(selected)]])
                elif learned_variant == "delta_rank_blend":
                    weight = float(np.clip(learned_delta_blend, 0.0, 1.0))
                    model_rank = np.empty(len(pool), dtype=np.float64)
                    model_rank[np.argsort(-scores, kind="stable")] = np.arange(len(pool), dtype=np.float64)
                    delta_values = delta[pool[:, 0], pool[:, 1]]
                    delta_rank = np.empty(len(pool), dtype=np.float64)
                    delta_rank[np.argsort(delta_values, kind="stable")] = np.arange(len(pool), dtype=np.float64)
                    denom = float(max(len(pool) - 1, 1))
                    ranking_score = -((1.0 - weight) * model_rank + weight * delta_rank) / denom
                    if k >= len(pool):
                        selected = np.argsort(-ranking_score)
                    else:
                        selected = np.argpartition(-ranking_score, kth=k - 1)[:k]
                        selected = selected[np.argsort(-ranking_score[selected])]
                elif learned_variant == "rank_sample":
                    model_rank = np.empty(len(pool), dtype=np.float64)
                    model_rank[np.argsort(-scores, kind="stable")] = np.arange(len(pool), dtype=np.float64)
                    normalized_rank = model_rank / float(max(len(pool) - 1, 1))
                    logits = -max(float(learned_rank_bias), 0.0) * normalized_rank
                    weights = np.exp(logits - logits.max())
                    probabilities = weights / weights.sum()
                    selected = candidate_rng.choice(len(pool), size=k, replace=False, p=probabilities)
                else:
                    raise ValueError(f"unsupported learned_variant: {learned_variant}")
                candidate_pairs = pool[selected]
                score_evals = len(pool)
        else:
            raise ValueError(f"unsupported official BLS descent method: {method}")
    values = delta[candidate_pairs[:, 0], candidate_pairs[:, 1]]
    selected_idx = int(np.argmin(values))
    delta_evals = len(candidate_pairs)
    certification_checked = False
    if (
        method == "random_certified"
        or (method == "random_one_rescue" and certification_allowed)
    ) and int(values[selected_idx]) >= 0:
        certification_checked = True
        candidate_pairs = all_swap_pairs(n)
        values = delta[candidate_pairs[:, 0], candidate_pairs[:, 1]]
        selected_idx = int(np.argmin(values))
        delta_evals += len(candidate_pairs)
    selected_i = int(candidate_pairs[selected_idx, 0])
    selected_j = int(candidate_pairs[selected_idx, 1])
    min_delta = int(values[selected_idx])
    if selected_i < 0 or current_cost + min_delta >= current_cost:
        return (
            False, current_cost, best_cost, iter_without_improvement, iteration,
            delta_evals, score_evals, certification_checked, False,
        )
    last_swapped[selected_i, selected_j] = iteration
    last_swapped[selected_j, selected_i] = iteration
    perm[selected_i], perm[selected_j] = perm[selected_j], perm[selected_i]
    current_cost += min_delta
    _update_delta(delta, perm, flow, distance, selected_i, selected_j, mode=delta_update_mode)
    if current_cost < best_cost:
        best_cost = current_cost
        best_perm[:] = perm
        iter_without_improvement = 0
    iteration += 1
    return (
        True, current_cost, best_cost, iter_without_improvement, iteration,
        delta_evals, score_evals, certification_checked, certification_checked,
    )


def _apply_move(
    perm: np.ndarray,
    delta: np.ndarray,
    current_cost: int,
    best_cost: int,
    best_perm: np.ndarray,
    last_swapped: np.ndarray,
    flow: np.ndarray,
    distance: np.ndarray,
    iter_without_improvement: int,
    iteration: int,
    i: int,
    j: int,
    delta_update_mode: str = "rebuild",
) -> tuple[int, int, int, int, int]:
    if i >= 0 and j >= 0:
        last_swapped[i, j] = iteration
        last_swapped[j, i] = iteration
        perm[i], perm[j] = perm[j], perm[i]
        current_cost += int(delta[i, j])
        _update_delta(delta, perm, flow, distance, i, j, mode=delta_update_mode)
        if current_cost < best_cost:
            best_cost = current_cost
            best_perm[:] = perm
            iter_without_improvement = 0
    iteration += 1
    return current_cost, best_cost, iter_without_improvement, iteration, int(i >= 0 and j >= 0)


def _directed_perturb(
    perm: np.ndarray,
    delta: np.ndarray,
    current_cost: int,
    best_cost: int,
    best_perm: np.ndarray,
    last_swapped: np.ndarray,
    flow: np.ndarray,
    distance: np.ndarray,
    init_cost: int,
    iteration: int,
    iter_without_improvement: int,
    rng: CStdRand,
    r1: float,
    r2: float,
    delta_update_mode: str = "rebuild",
) -> tuple[int, int, int, int, int, int]:
    n = len(perm)
    min_delta = INFINITE
    selected_i = -1
    selected_j = -1
    tenure_noise = max(int(n * r2), 1)
    for i in range(n - 1):
        for j in range(i + 1, n):
            next_cost = current_cost + int(delta[i, j])
            if next_cost != init_cost and int(delta[i, j]) < min_delta and (
                (int(last_swapped[i, j]) + n * r1 + rng.rand() % tenure_noise < iteration)
                or next_cost < best_cost
            ):
                selected_i, selected_j, min_delta = i, j, int(delta[i, j])
    current_cost, best_cost, iter_without_improvement, iteration, moved = _apply_move(
        perm, delta, current_cost, best_cost, best_perm, last_swapped, flow, distance,
        iter_without_improvement, iteration, selected_i, selected_j, delta_update_mode
    )
    return current_cost, best_cost, iter_without_improvement, iteration, moved, int(moved)


def _recency_perturb(
    perm: np.ndarray,
    delta: np.ndarray,
    current_cost: int,
    best_cost: int,
    best_perm: np.ndarray,
    last_swapped: np.ndarray,
    flow: np.ndarray,
    distance: np.ndarray,
    iteration: int,
    iter_without_improvement: int,
    delta_update_mode: str = "rebuild",
) -> tuple[int, int, int, int, int]:
    n = len(perm)
    min_age = INFINITE
    selected_i = -1
    selected_j = -1
    for i in range(n - 1):
        for j in range(i + 1, n):
            if int(last_swapped[i, j]) < min_age:
                selected_i, selected_j, min_age = i, j, int(last_swapped[i, j])
    return _apply_move(
        perm, delta, current_cost, best_cost, best_perm, last_swapped, flow, distance,
        iter_without_improvement, iteration, selected_i, selected_j, delta_update_mode
    )


def _random_perturb(
    perm: np.ndarray,
    delta: np.ndarray,
    current_cost: int,
    best_cost: int,
    best_perm: np.ndarray,
    last_swapped: np.ndarray,
    flow: np.ndarray,
    distance: np.ndarray,
    init_cost: int,
    iteration: int,
    iter_without_improvement: int,
    rng: CStdRand,
    delta_update_mode: str = "rebuild",
) -> tuple[int, int, int, int, int]:
    n = len(perm)
    i = rng.uniform_int(0, n - 1)
    j = rng.uniform_int(0, n - 1)
    if i > j:
        i, j = j, i
    attempts = 0
    while (i == j or current_cost + int(delta[i, j]) == init_cost) and attempts < max(n * n * 4, 16):
        j = rng.uniform_int(0, n - 1)
        if i > j:
            i, j = j, i
        attempts += 1
    if i == j or current_cost + int(delta[i, j]) == init_cost:
        return _apply_move(
            perm, delta, current_cost, best_cost, best_perm, last_swapped, flow, distance,
            iter_without_improvement, iteration, -1, -1, delta_update_mode
        )
    return _apply_move(
        perm, delta, current_cost, best_cost, best_perm, last_swapped, flow, distance,
        iter_without_improvement, iteration, i, j, delta_update_mode
    )


def _tabu_burst_perturb(
    perm: np.ndarray,
    delta: np.ndarray,
    current_cost: int,
    best_cost: int,
    best_perm: np.ndarray,
    last_swapped: np.ndarray,
    flow: np.ndarray,
    distance: np.ndarray,
    iteration: int,
    iter_without_improvement: int,
    rng: np.random.Generator,
    steps: int,
    delta_update_mode: str,
    deadline: float | None = None,
) -> tuple[int, int, int, int, int]:
    pairs = all_swap_pairs(len(perm))
    tabu_until = np.zeros((len(perm), len(perm)), dtype=np.int64)
    base_tenure = max(int(round(0.6 * len(perm))), 2)
    moved_total = 0
    for step in range(int(steps)):
        if deadline is not None and time.perf_counter() >= deadline:
            break
        values = delta[pairs[:, 0], pairs[:, 1]]
        next_costs = current_cost + values
        admissible = (tabu_until[pairs[:, 0], pairs[:, 1]] <= step) | (next_costs < best_cost)
        indices = np.flatnonzero(admissible)
        if len(indices) == 0:
            indices = np.arange(len(pairs), dtype=np.int64)
        best_value = np.min(values[indices])
        tied = indices[values[indices] == best_value]
        selected = int(tied[int(rng.integers(0, len(tied)))])
        i, j = int(pairs[selected, 0]), int(pairs[selected, 1])
        current_cost, best_cost, iter_without_improvement, iteration, moved = _apply_move(
            perm, delta, current_cost, best_cost, best_perm, last_swapped, flow, distance,
            iter_without_improvement, iteration, i, j, delta_update_mode,
        )
        tenure = base_tenure + int(rng.integers(0, max(len(perm) // 5, 1) + 1))
        tabu_until[i, j] = tabu_until[j, i] = step + tenure
        moved_total += int(moved)
    return current_cost, best_cost, iter_without_improvement, iteration, moved_total


def _update_elite_pool(
    elite_pool: list[tuple[int, np.ndarray]],
    perm: np.ndarray,
    cost: int,
    max_size: int,
) -> None:
    if any(np.array_equal(perm, elite_perm) for _, elite_perm in elite_pool):
        return
    elite_pool.append((int(cost), perm.copy()))
    elite_pool.sort(key=lambda item: item[0])
    del elite_pool[int(max_size):]


def _elite_relink_perturb(
    perm: np.ndarray,
    guide: np.ndarray,
    delta: np.ndarray,
    current_cost: int,
    best_cost: int,
    best_perm: np.ndarray,
    last_swapped: np.ndarray,
    flow: np.ndarray,
    distance: np.ndarray,
    iter_without_improvement: int,
    iteration: int,
    max_steps: int,
    delta_update_mode: str,
    deadline: float | None = None,
) -> tuple[int, int, int, int, int]:
    n = len(perm)
    moves = 0
    for _ in range(int(max_steps)):
        if deadline is not None and time.perf_counter() >= deadline:
            break
        mismatched = np.flatnonzero(perm != guide)
        if len(mismatched) < 2:
            break
        location_to_facility = np.empty(n, dtype=np.int64)
        location_to_facility[perm] = np.arange(n, dtype=np.int64)
        pairs = np.asarray(
            [
                sorted((int(i), int(location_to_facility[int(guide[i])])))
                for i in mismatched
            ],
            dtype=np.int64,
        )
        pairs = np.unique(pairs, axis=0)
        values = delta[pairs[:, 0], pairs[:, 1]]
        selected = int(np.argmin(values))
        i, j = int(pairs[selected, 0]), int(pairs[selected, 1])
        current_cost, best_cost, iter_without_improvement, iteration, moved = _apply_move(
            perm, delta, current_cost, best_cost, best_perm, last_swapped, flow, distance,
            iter_without_improvement, iteration, i, j, delta_update_mode,
        )
        moves += int(moved)
    return current_cost, best_cost, iter_without_improvement, iteration, moves


def official_bls(
    instance: QAPInstance,
    *,
    seed: int = 0,
    initial_perm: np.ndarray | None = None,
    max_time_sec: float | None = None,
    max_outer_iterations: int = 2_000_000_000,
    max_descent_decisions: int | None = None,
    stop_at_optimum: bool = False,
    target_cost: int | None = None,
    r1: float = 0.7,
    r2: float = 0.2,
    init_perturb_strength: float = 0.15,
    stagnation_threshold: int = 2500,
    p0: float = 0.75,
    q: float = 0.3,
    method: str = "full",
    candidate_swaps: int | None = None,
    score_pool_swaps: int | None = None,
    selector: SelectorBundle | None = None,
    pair_score_callback: Callable[[QAPInstance, np.ndarray, np.ndarray, SwapFeatureCache], np.ndarray] | None = None,
    learned_score_interval: int = 1,
    learned_score_schedule: str = "interval",
    learned_variant: str = "plain",
    learned_random_rescue: int = 0,
    learned_delta_blend: float = 0.5,
    learned_rank_bias: float = 1.0,
    delta_update_mode: str = "rebuild",
    elite_relink_interval: int | None = None,
    elite_relink_steps: int = 8,
    elite_pool_size: int = 4,
    d28_relink_interval: int | None = None,
    d28_elite_pool_size: int = 8,
    d28_min_distance_fraction: float = 0.25,
    d28_path_local_stride: int = 2,
    d28_path_local_moves: int = 2,
    d28_relink_mode: str = "bidirectional_internal",
    d28_path_max_steps: int = 20,
    d28_acceptance_mode: str = "always",
    macro_perturb_policy: str = "official",
    macro_controller: object | None = None,
    trace_callback: Callable[[OfficialBLSStateTrace], None] | None = None,
    resume_state: OfficialBLSStateTrace | None = None,
    resume_action: tuple[int, int] | None = None,
    resume_skip_descent: bool = False,
    wall_clock_includes_initialization: bool = False,
) -> OfficialBLSResult:
    """Run official BLS with optional wall, outer-iteration, or descent-decision budgets."""
    function_started = time.perf_counter()
    flow, distance = _integer_matrices(instance)
    if method not in {"full", "random", "random_certified", "random_one_rescue", "learned"}:
        raise ValueError("method must be full, random, random_certified, random_one_rescue, or learned")
    if method == "learned" and selector is None and pair_score_callback is None:
        raise ValueError("selector or pair_score_callback is required for learned official BLS")
    if int(learned_score_interval) <= 0:
        raise ValueError("learned_score_interval must be positive")
    if learned_score_schedule not in {"interval", "outer_first"}:
        raise ValueError("learned_score_schedule must be interval or outer_first")
    if learned_variant not in {"plain", "random_rescue", "delta_rank_blend", "rank_sample"}:
        raise ValueError("unsupported learned_variant")
    if int(learned_random_rescue) < 0:
        raise ValueError("learned_random_rescue must be nonnegative")
    if max_descent_decisions is not None and int(max_descent_decisions) <= 0:
        raise ValueError("max_descent_decisions must be positive or None")
    if delta_update_mode not in {"rebuild", "incremental"}:
        raise ValueError("delta_update_mode must be rebuild or incremental")
    if elite_relink_interval is not None and int(elite_relink_interval) <= 0:
        raise ValueError("elite_relink_interval must be positive or None")
    if int(elite_relink_steps) <= 0 or int(elite_pool_size) < 2:
        raise ValueError("elite_relink_steps must be positive and elite_pool_size must be at least 2")
    if d28_relink_interval is not None and int(d28_relink_interval) <= 0:
        raise ValueError("d28_relink_interval must be positive or None")
    if elite_relink_interval is not None and d28_relink_interval is not None:
        raise ValueError("D26 and D28 elite relinking cannot be enabled together")
    if int(d28_elite_pool_size) < 2:
        raise ValueError("d28_elite_pool_size must be at least 2")
    if not 0.0 < float(d28_min_distance_fraction) <= 1.0:
        raise ValueError("d28_min_distance_fraction must be in (0, 1]")
    if int(d28_path_local_stride) <= 0 or int(d28_path_local_moves) <= 0:
        raise ValueError("D28 local stride and moves must be positive")
    if d28_relink_mode not in {"bidirectional_internal", "sparse_unidirectional"}:
        raise ValueError("unsupported D28 relink mode")
    if int(d28_path_max_steps) <= 0:
        raise ValueError("d28_path_max_steps must be positive")
    if d28_acceptance_mode not in {"always", "improve_current", "improve_incumbent"}:
        raise ValueError("unsupported D28 acceptance mode")
    if macro_perturb_policy not in {"official", "tabu", "controller"}:
        raise ValueError("macro_perturb_policy must be official, tabu, or controller")
    if macro_perturb_policy == "controller" and macro_controller is None:
        raise ValueError("macro_controller is required for controller policy")
    if resume_state is not None and initial_perm is not None:
        raise ValueError("initial_perm and resume_state are mutually exclusive")
    if resume_state is None and (resume_action is not None or resume_skip_descent):
        raise ValueError("resume_action and resume_skip_descent require resume_state")
    if resume_state is not None and (elite_relink_interval is not None or d28_relink_interval is not None):
        raise ValueError("resuming elite relinking requires an elite-pool snapshot")
    n = flow.shape[0]
    rng = CStdRand(int(seed))
    candidate_rng = np.random.default_rng(int(seed))
    if resume_state is not None:
        perm = np.asarray(resume_state.perm, dtype=np.int64).copy()
        if sorted(perm.tolist()) != list(range(n)):
            raise ValueError("resume_state.perm must be a permutation of 0..n-1")
    elif initial_perm is None:
        perm = _official_random_perm(n, rng)
    else:
        perm = np.asarray(initial_perm, dtype=np.int64).copy()
        if sorted(perm.tolist()) != list(range(n)):
            raise ValueError("initial_perm must be a permutation of 0..n-1")
    computed_current_cost = int(np.sum(flow.astype(object) * distance[np.ix_(perm, perm)].astype(object)))
    if resume_state is not None and computed_current_cost != int(resume_state.current_cost):
        raise ValueError("resume_state.current_cost does not match its permutation")
    current_cost = computed_current_cost
    initial_cost = current_cost
    if resume_state is None:
        best_perm = perm.copy()
        best_cost = current_cost
    else:
        if resume_state.best_perm is None:
            if int(resume_state.best_cost) != current_cost:
                raise ValueError("resume_state.best_perm is required when best_cost differs from current_cost")
            best_perm = perm.copy()
        else:
            best_perm = np.asarray(resume_state.best_perm, dtype=np.int64).copy()
            if sorted(best_perm.tolist()) != list(range(n)):
                raise ValueError("resume_state.best_perm must be a permutation of 0..n-1")
        best_cost = int(resume_state.best_cost)
        computed_best_cost = int(np.sum(flow.astype(object) * distance[np.ix_(best_perm, best_perm)].astype(object)))
        if computed_best_cost != best_cost:
            raise ValueError("resume_state.best_cost does not match best_perm")
    delta = _initialize_delta(perm, flow, distance)
    feature_cache = build_swap_feature_cache(instance) if method == "learned" else None
    last_swapped = (
        np.zeros((n, n), dtype=np.int64)
        if resume_state is None
        else np.asarray(resume_state.last_swapped, dtype=np.int64).copy()
    )
    if last_swapped.shape != (n, n):
        raise ValueError("resume_state.last_swapped must have shape [n, n]")
    iteration = 0 if resume_state is None else int(resume_state.iteration)
    iter_without_improvement = 0 if resume_state is None else int(resume_state.iter_without_improvement)
    perturb_strength = (
        math.ceil(init_perturb_strength * n)
        if resume_state is None
        else float(resume_state.perturb_strength)
    )
    n_outer = n_moves = n_descent_decisions = n_descent = n_perturb = n_delta = n_score = 0
    n_certification_checks = n_certification_moves = 0
    n_relink_calls = n_relink_moves = 0
    n_d28_relink_calls = n_d28_path_moves = n_d28_internal_moves = n_d28_descent_moves = 0
    n_d28_pool_rejections = n_d28_relink_rejections = 0
    n_directed = n_recency = n_random = n_noop = 0
    n_macro_official = n_macro_tabu = 0
    macro_controller_overhead = 0.0
    state_source = "initial" if resume_state is None else str(resume_state.state_source)
    elite_pool: list[tuple[int, np.ndarray]] = []
    d28_elite_pool = []
    recent_descent_gains: deque[float] = deque(maxlen=10)
    learned_decisions = 0 if resume_state is None else int(resume_state.learned_decisions)
    resume_pending = resume_state is not None
    started = function_started if wall_clock_includes_initialization else time.perf_counter()
    deadline = None if max_time_sec is None else started + float(max_time_sec)
    incumbent_integral = 0.0
    incumbent_integral_time = started
    incumbent_integral_best = int(best_cost)
    incumbent_scale = max(abs(float(initial_cost)), 1.0)

    def record_incumbent(now: float | None = None) -> float:
        nonlocal incumbent_integral, incumbent_integral_time, incumbent_integral_best
        observed_at = time.perf_counter() if now is None else float(now)
        duration = max(observed_at - incumbent_integral_time, 0.0)
        incumbent_integral += (
            (float(initial_cost) - float(incumbent_integral_best)) / incumbent_scale
        ) * duration
        incumbent_integral_time = observed_at
        incumbent_integral_best = int(best_cost)
        return observed_at

    if macro_perturb_policy == "controller" and hasattr(macro_controller, "start_run"):
        controller_started = time.perf_counter()
        macro_controller.start_run(
            initial_cost=float(initial_cost),
            best_cost=float(best_cost),
            started_at=float(started),
            max_time_sec=max_time_sec,
        )
        macro_controller_overhead += time.perf_counter() - controller_started

    def target_reached() -> bool:
        return bool(stop_at_optimum and target_cost is not None and best_cost <= int(target_cost))

    for _ in range(int(max_outer_iterations)):
        if (
            wall_clock_includes_initialization
            and deadline is not None
            and time.perf_counter() >= deadline
        ):
            break
        n_outer += 1
        if resume_pending:
            previous_cost = (
                current_cost
                if resume_state.outer_previous_cost is None
                else int(resume_state.outer_previous_cost)
            )
            descent_num = int(resume_state.outer_descent_num)
            descent_step = int(resume_state.descent_step)
            rescue_used = bool(resume_state.rescue_used)
            if resume_action is not None:
                action_i, action_j = sorted((int(resume_action[0]), int(resume_action[1])))
                if action_i < 0 or action_j >= n or action_i == action_j:
                    raise ValueError("resume_action must contain two distinct indices in [0, n)")
                if int(delta[action_i, action_j]) >= 0:
                    raise ValueError("resume_action must be an improving swap")
                current_cost, best_cost, iter_without_improvement, iteration, moved = _apply_move(
                    perm, delta, current_cost, best_cost, best_perm, last_swapped, flow, distance,
                    iter_without_improvement, iteration, action_i, action_j, delta_update_mode,
                )
                n_descent_decisions += 1
                n_delta += 1
                n_moves += int(moved)
                n_descent += int(moved)
                descent_num += int(moved)
                descent_step += int(moved)
            skip_descent = bool(resume_skip_descent)
            resume_pending = False
        else:
            previous_cost = current_cost
            descent_num = 0
            descent_step = 0
            rescue_used = False
            skip_descent = False
        while not skip_descent:
            if (
                wall_clock_includes_initialization
                and deadline is not None
                and time.perf_counter() >= deadline
            ):
                break
            if max_descent_decisions is not None and n_descent_decisions >= int(max_descent_decisions):
                break
            if trace_callback is not None:
                trace_callback(
                    OfficialBLSStateTrace(
                        outer_iteration=int(n_outer),
                        descent_step=int(descent_step),
                        state_source=state_source,
                        perm=perm.copy(),
                        current_cost=int(current_cost),
                        best_cost=int(best_cost),
                        iteration=int(iteration),
                        iter_without_improvement=int(iter_without_improvement),
                        perturb_strength=float(perturb_strength),
                        last_swapped=last_swapped.copy(),
                        best_perm=best_perm.copy(),
                        outer_previous_cost=int(previous_cost),
                        outer_descent_num=int(descent_num),
                        learned_decisions=int(learned_decisions),
                        rescue_used=bool(rescue_used),
                    )
                )
            score_learned = (
                descent_step == 0
                if learned_score_schedule == "outer_first"
                else learned_decisions % int(learned_score_interval) == 0
            )
            (
                moved, current_cost, best_cost, iter_without_improvement, iteration,
                delta_evals, score_evals, certification_checked, certification_move,
            ) = _best_improvement_move(
                instance, perm, delta, current_cost, best_cost, best_perm, last_swapped, flow, distance,
                iter_without_improvement, iteration, method, candidate_rng, candidate_swaps,
                score_pool_swaps, selector, pair_score_callback, feature_cache, score_learned, delta_update_mode,
                certification_allowed=not rescue_used,
                learned_variant=learned_variant,
                learned_random_rescue=learned_random_rescue,
                learned_delta_blend=learned_delta_blend,
                learned_rank_bias=learned_rank_bias,
            )
            rescue_used = rescue_used or certification_move
            learned_decisions += int(method == "learned")
            n_descent_decisions += 1
            n_delta += int(delta_evals)
            n_score += int(score_evals)
            n_certification_checks += int(certification_checked)
            n_certification_moves += int(certification_move)
            if not moved:
                break
            if int(best_cost) != incumbent_integral_best:
                record_incumbent()
            n_moves += 1
            n_descent += 1
            descent_num += 1
            descent_step += 1

        if (
            wall_clock_includes_initialization
            and deadline is not None
            and time.perf_counter() >= deadline
        ):
            break
        if max_descent_decisions is not None and n_descent_decisions >= int(max_descent_decisions):
            break

        _update_elite_pool(elite_pool, perm, current_cost, int(elite_pool_size))
        if elite_relink_interval is not None and n_outer % int(elite_relink_interval) == 0:
            guides = [
                (cost, elite_perm)
                for cost, elite_perm in elite_pool
                if not np.array_equal(perm, elite_perm)
            ]
            if guides:
                _, guide = min(
                    guides,
                    key=lambda item: (item[0], -int(np.sum(perm != item[1]))),
                )
                n_relink_calls += 1
                current_cost, best_cost, iter_without_improvement, iteration, relink_moves = _elite_relink_perturb(
                    perm, guide, delta, current_cost, best_cost, best_perm, last_swapped,
                    flow, distance, iter_without_improvement, iteration,
                    int(elite_relink_steps), delta_update_mode, deadline,
                )
                if relink_moves:
                    n_relink_moves += int(relink_moves)
                    n_perturb += int(relink_moves)
                    n_moves += int(relink_moves)
                    state_source = "relink"
                    if target_reached():
                        break
                    if deadline is not None and time.perf_counter() >= deadline:
                        break
                    continue

        if d28_relink_interval is not None:
            from .d28_elite_relink import (
                choose_diverse_elite_guide,
                run_d28_bidirectional_relink,
                run_d28_sparse_unidirectional_relink,
                update_diverse_elite_pool,
            )

            minimum_distance = max(int(math.ceil(float(d28_min_distance_fraction) * n)), 2)
            accepted = update_diverse_elite_pool(
                d28_elite_pool,
                perm,
                current_cost,
                max_size=int(d28_elite_pool_size),
                min_distance=minimum_distance,
            )
            n_d28_pool_rejections += int(not accepted)
            if n_outer % int(d28_relink_interval) == 0:
                guide = choose_diverse_elite_guide(
                    d28_elite_pool,
                    perm,
                    min_distance=minimum_distance,
                )
                if guide is not None:
                    source_current_cost = int(current_cost)
                    source_best_cost = int(best_cost)
                    trace = OfficialBLSStateTrace(
                        outer_iteration=int(n_outer),
                        descent_step=int(descent_step),
                        state_source=state_source,
                        perm=perm.copy(),
                        current_cost=int(current_cost),
                        best_cost=int(best_cost),
                        iteration=int(iteration),
                        iter_without_improvement=int(iter_without_improvement),
                        perturb_strength=float(perturb_strength),
                        last_swapped=last_swapped.copy(),
                        best_perm=best_perm.copy(),
                        outer_previous_cost=int(previous_cost),
                        outer_descent_num=int(descent_num),
                        learned_decisions=int(learned_decisions),
                        rescue_used=bool(rescue_used),
                    )
                    if d28_relink_mode == "sparse_unidirectional":
                        relink = run_d28_sparse_unidirectional_relink(
                            instance,
                            trace,
                            guide,
                            max_steps=int(d28_path_max_steps),
                            deadline=deadline,
                        )
                    else:
                        relink = run_d28_bidirectional_relink(
                            instance,
                            trace,
                            guide,
                            local_stride=int(d28_path_local_stride),
                            local_moves=int(d28_path_local_moves),
                            deadline=deadline,
                        )
                    n_d28_relink_calls += 1
                    n_d28_path_moves += int(relink.path_moves)
                    n_d28_internal_moves += int(relink.internal_moves)
                    n_d28_descent_moves += int(relink.descent_moves)
                    n_delta += int(relink.delta_evals)
                    n_perturb += int(relink.path_moves + relink.internal_moves)
                    n_descent += int(relink.descent_moves)
                    n_moves += int(relink.path_moves + relink.internal_moves + relink.descent_moves)
                    accept_relink = (
                        d28_acceptance_mode == "always"
                        or (
                            d28_acceptance_mode == "improve_current"
                            and int(relink.workspace.current_cost) < source_current_cost
                        )
                        or int(relink.workspace.best_cost) < source_best_cost
                    )
                    if accept_relink:
                        workspace = relink.workspace
                        perm[:] = workspace.perm
                        delta[:, :] = workspace.delta
                        current_cost = int(workspace.current_cost)
                        best_cost = int(workspace.best_cost)
                        best_perm[:] = workspace.best_perm
                        last_swapped[:, :] = workspace.last_swapped
                        iteration = int(workspace.iteration)
                        iter_without_improvement = int(workspace.iter_without_improvement)
                        state_source = "d28_relink"
                        if target_reached():
                            break
                        if deadline is not None and time.perf_counter() >= deadline:
                            break
                        continue
                    n_d28_relink_rejections += 1
                    if deadline is not None and time.perf_counter() >= deadline:
                        break

        if iter_without_improvement > stagnation_threshold:
            iter_without_improvement = 0
            perturb_strength = n * (0.4 + rng.rand() % 20 / 100.0)
        elif descent_num != 0 and previous_cost != current_cost:
            iter_without_improvement += 1
            perturb_strength = math.ceil(init_perturb_strength * n)
            if perturb_strength < 5:
                perturb_strength = 5
        elif previous_cost == current_cost:
            perturb_strength += 1

        recent_descent_gains.append(
            max(float(previous_cost - current_cost), 0.0) / max(abs(float(initial_cost)), 1.0)
        )

        selected_macro = macro_perturb_policy
        if selected_macro == "controller":
            controller_started = time.perf_counter()
            if hasattr(macro_controller, "select_action"):
                observed_at = record_incumbent(controller_started)
                elapsed_fraction = (
                    None
                    if max_time_sec is None
                    else (observed_at - started) / max(float(max_time_sec), EPS)
                )
                selected_macro = str(
                    macro_controller.select_action(
                        instance=instance,
                        perm=perm,
                        current_cost=current_cost,
                        best_cost=best_cost,
                        now=observed_at,
                        elapsed_fraction=elapsed_fraction,
                        state_features={
                            **hamming_distance_features(
                                perm,
                                [elite_perm for _, elite_perm in elite_pool],
                                incumbent_perm=best_perm,
                            ),
                            **basin_depth_features(recent_descent_gains),
                        },
                    )
                )
            else:
                selected_macro = str(
                    macro_controller.predict_action(instance, perm, current_cost=current_cost)
                )
            macro_controller_overhead += time.perf_counter() - controller_started
            if selected_macro == "official_perturb":
                selected_macro = "official"
            elif selected_macro == "tabu_burst":
                selected_macro = "tabu"
            else:
                raise ValueError(f"macro controller returned unsupported action: {selected_macro}")
        if selected_macro == "tabu":
            n_macro_tabu += 1
            current_cost, best_cost, iter_without_improvement, iteration, moved = _tabu_burst_perturb(
                perm, delta, current_cost, best_cost, best_perm, last_swapped, flow, distance,
                iteration, iter_without_improvement, candidate_rng, n, delta_update_mode, deadline,
            )
            n_perturb += int(moved)
            n_moves += int(moved)
            state_source = "tabu"
        else:
            n_macro_official += 1
            init_cost = previous_cost
            d = float(iter_without_improvement) / float(stagnation_threshold)
            probability = math.exp(-d)
            if probability < p0:
                probability = p0
            directed = probability > rng.unit_101()
            perturb_kinds: list[str] = []
            for _ in range(int(math.ceil(perturb_strength))):
                if directed:
                    perturb_kinds.append("directed")
                    current_cost, best_cost, iter_without_improvement, iteration, moved, count = _directed_perturb(
                        perm, delta, current_cost, best_cost, best_perm, last_swapped, flow, distance,
                        init_cost, iteration, iter_without_improvement, rng, r1, r2, delta_update_mode
                    )
                    n_directed += count
                elif q > rng.unit_101():
                    perturb_kinds.append("recency")
                    current_cost, best_cost, iter_without_improvement, iteration, moved = _recency_perturb(
                        perm, delta, current_cost, best_cost, best_perm, last_swapped, flow, distance,
                        iteration, iter_without_improvement, delta_update_mode
                    )
                    n_recency += int(moved)
                else:
                    perturb_kinds.append("random")
                    current_cost, best_cost, iter_without_improvement, iteration, moved = _random_perturb(
                        perm, delta, current_cost, best_cost, best_perm, last_swapped, flow, distance,
                        init_cost, iteration, iter_without_improvement, rng, delta_update_mode
                    )
                    n_random += int(moved)
                n_perturb += 1
                n_moves += int(moved)
                n_noop += int(not moved)
            unique_kinds = set(perturb_kinds)
            state_source = next(iter(unique_kinds)) if len(unique_kinds) == 1 else "mixed"
        if target_reached():
            break
        # The official source checks the stopping condition after perturbation.
        if deadline is not None and time.perf_counter() >= deadline:
            break

    finished_at = record_incumbent()
    if macro_perturb_policy == "controller" and hasattr(macro_controller, "finish_run"):
        controller_started = time.perf_counter()
        macro_controller.finish_run(
            best_cost=float(best_cost),
            now=float(finished_at),
        )
        macro_controller_overhead += time.perf_counter() - controller_started
    runtime = finished_at - started
    controller_updates = int(getattr(macro_controller, "n_updates", 0)) if macro_controller is not None else 0
    controller_switches = int(getattr(macro_controller, "n_switches", 0)) if macro_controller is not None else 0
    return OfficialBLSResult(
        perm=best_perm.copy(),
        cost=int(best_cost),
        initial_cost=int(initial_cost),
        runtime_sec=float(runtime),
        n_outer_iters=int(n_outer),
        n_moves=int(n_moves),
        n_descent_decisions=int(n_descent_decisions),
        n_descent_moves=int(n_descent),
        n_perturb_moves=int(n_perturb),
        n_delta_evals=int(n_delta),
        n_score_evals=int(n_score),
        n_certification_checks=int(n_certification_checks),
        n_certification_moves=int(n_certification_moves),
        n_relink_calls=int(n_relink_calls),
        n_relink_moves=int(n_relink_moves),
        n_d28_relink_calls=int(n_d28_relink_calls),
        n_d28_path_moves=int(n_d28_path_moves),
        n_d28_internal_moves=int(n_d28_internal_moves),
        n_d28_descent_moves=int(n_d28_descent_moves),
        n_d28_pool_rejections=int(n_d28_pool_rejections),
        n_d28_relink_rejections=int(n_d28_relink_rejections),
        n_directed_moves=int(n_directed),
        n_recency_moves=int(n_recency),
        n_random_moves=int(n_random),
        n_noop_moves=int(n_noop),
        n_macro_official_actions=int(n_macro_official),
        n_macro_tabu_actions=int(n_macro_tabu),
        mean_relative_incumbent_gain=float(incumbent_integral / max(runtime, EPS)),
        macro_controller_overhead_sec=float(macro_controller_overhead),
        n_macro_controller_updates=controller_updates,
        n_macro_controller_switches=controller_switches,
        hit_target=target_reached(),
        seed=int(seed),
        method=method,
        delta_update_mode=delta_update_mode,
    )


__all__ = ["CStdRand", "OfficialBLSResult", "OfficialBLSStateTrace", "official_bls"]
