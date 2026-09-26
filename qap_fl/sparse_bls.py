"""Candidate-restricted BLS without a dense swap-delta matrix.

This is an experimental sparse counterpart to :mod:`official_bls`.  It keeps
accepted move costs exact, but computes them only for the current candidate
set.  The perturbation phase is candidate-restricted as well, so no operation
requires initializing or updating all O(n^2) swap deltas.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import math
import time

import numpy as np
import torch

from .features import EPS, FEATURE_FIELDS, FEATURE_NAMES, normalized_row_stats
from .local_search import SelectorBundle
from .official_bls import CStdRand, _compute_deltas_for_pairs, _integer_matrices, _official_random_perm
from .qap import QAPInstance


@dataclass(frozen=True)
class SparseSwapFeatureCache:
    F: np.ndarray
    D: np.ndarray
    f_sum: np.ndarray
    f_mean: np.ndarray
    f_std: np.ndarray
    d_sum: np.ndarray
    d_mean: np.ndarray
    d_std: np.ndarray
    qap_scale: float
    denom: float
    f_norm: np.ndarray
    d_norm: np.ndarray
    f_abs_scale: float
    d_abs_scale: float
    degree_order: np.ndarray


@dataclass
class SparseSwapFeatureState:
    cache: SparseSwapFeatureCache
    node_contrib: np.ndarray

    @classmethod
    def create(cls, instance: QAPInstance, perm: np.ndarray) -> "SparseSwapFeatureState":
        cache = build_sparse_feature_cache(instance)
        return cls(cache=cache, node_contrib=_node_current_contributions(cache, perm))

    def apply_swap(self, perm: np.ndarray, r: int, s: int) -> None:
        """Update all state-dependent features in O(n), then mutate ``perm``."""
        cache = self.cache
        F = cache.F
        D = cache.D
        old_r = int(perm[r])
        old_s = int(perm[s])
        locations = perm.copy()
        mask = np.ones(len(perm), dtype=bool)
        mask[[r, s]] = False
        k = np.flatnonzero(mask)
        p_k = locations[k]
        self.node_contrib[k] += (
            (F[k, r] - F[k, s]) * (D[p_k, old_s] - D[p_k, old_r])
            + (F[r, k] - F[s, k]) * (D[old_s, p_k] - D[old_r, p_k])
        ) / cache.qap_scale
        perm[r], perm[s] = perm[s], perm[r]
        self.node_contrib[r] = _single_node_contribution(cache, perm, r)
        self.node_contrib[s] = _single_node_contribution(cache, perm, s)


@dataclass(frozen=True)
class SparseBLSResult:
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
    n_directed_moves: int
    n_recency_moves: int
    n_random_moves: int
    n_noop_moves: int
    seed: int
    method: str
    dense_delta_initializations: int = 0
    dense_delta_updates: int = 0
    sparse_delta_cache_updates: int = 0


@dataclass
class SparseDeltaPool:
    """An exact persistent delta table for a candidate subset."""

    pairs: np.ndarray
    values: np.ndarray

    @classmethod
    def create(
        cls,
        perm: np.ndarray,
        flow: np.ndarray,
        distance: np.ndarray,
        pairs: np.ndarray,
    ) -> "SparseDeltaPool":
        pairs = np.asarray(pairs, dtype=np.int64).copy()
        return cls(
            pairs=pairs,
            values=_compute_deltas_for_pairs(perm, flow, distance, pairs),
        )

    def update_after_swap(
        self,
        perm: np.ndarray,
        flow: np.ndarray,
        distance: np.ndarray,
        r: int,
        s: int,
    ) -> tuple[int, int]:
        """Update cached deltas after ``perm`` has already swapped r and s."""
        i = self.pairs[:, 0]
        j = self.pairs[:, 1]
        incident = (i == r) | (j == r) | (i == s) | (j == s)
        regular = ~incident
        if np.any(regular):
            ii = i[regular]
            jj = j[regular]
            p_i = perm[ii]
            p_j = perm[jj]
            p_r = int(perm[r])
            p_s = int(perm[s])
            correction = (
                (flow[r, ii] - flow[r, jj] + flow[s, jj] - flow[s, ii])
                * (
                    distance[p_s, p_i]
                    - distance[p_s, p_j]
                    + distance[p_r, p_j]
                    - distance[p_r, p_i]
                )
                + (flow[ii, r] - flow[jj, r] + flow[jj, s] - flow[ii, s])
                * (
                    distance[p_i, p_s]
                    - distance[p_j, p_s]
                    + distance[p_j, p_r]
                    - distance[p_i, p_r]
                )
            )
            self.values[regular] += correction
        incident_count = int(np.sum(incident))
        if incident_count:
            self.values[incident] = _compute_deltas_for_pairs(
                perm, flow, distance, self.pairs[incident]
            )
        return incident_count, int(np.sum(regular))


def build_sparse_feature_cache(instance: QAPInstance) -> SparseSwapFeatureCache:
    """Build only O(n^2) instance data; pair features remain on demand."""
    F = np.asarray(instance.F, dtype=np.float64)
    D = np.asarray(instance.D, dtype=np.float64)
    f_sum, f_mean, f_std = normalized_row_stats(F)
    d_sum, d_mean, d_std = normalized_row_stats(D)
    return SparseSwapFeatureCache(
        F=F,
        D=D,
        f_sum=f_sum,
        f_mean=f_mean,
        f_std=f_std,
        d_sum=d_sum,
        d_mean=d_mean,
        d_std=d_std,
        qap_scale=(float(np.mean(np.abs(F))) + EPS)
        * (float(np.mean(np.abs(D))) + EPS)
        * max(instance.n, 1),
        denom=float(max(instance.n - 1, 1)),
        f_norm=np.linalg.norm(F, axis=1),
        d_norm=np.linalg.norm(D, axis=1),
        f_abs_scale=float(np.mean(np.abs(F))) + EPS,
        d_abs_scale=float(np.mean(np.abs(D))) + EPS,
        degree_order=np.argsort(-np.sum(np.abs(F), axis=1)),
    )


def _node_current_contributions(cache: SparseSwapFeatureCache, perm: np.ndarray) -> np.ndarray:
    assigned = cache.D[np.ix_(perm, perm)]
    products = cache.F * assigned
    diagonal = np.diag(cache.F) * np.diag(assigned)
    return (products.sum(axis=1) + products.sum(axis=0) - diagonal) / cache.qap_scale


def _single_node_contribution(cache: SparseSwapFeatureCache, perm: np.ndarray, i: int) -> float:
    p_i = int(perm[i])
    locations = perm
    outgoing = np.sum(cache.F[i, :] * cache.D[p_i, locations])
    incoming = np.sum(cache.F[:, i] * cache.D[locations, p_i])
    diagonal = cache.F[i, i] * cache.D[p_i, p_i]
    return float((outgoing + incoming - diagonal) / cache.qap_scale)


def _anchors_for_pairs(cache: SparseSwapFeatureCache, pairs: np.ndarray, max_anchors: int = 5) -> np.ndarray:
    anchors = np.full((len(pairs), max_anchors), -1, dtype=np.int64)
    for row, (i_raw, j_raw) in enumerate(pairs):
        i = int(i_raw)
        j = int(j_raw)
        selected = [int(k) for k in cache.degree_order if int(k) != i and int(k) != j][:max_anchors]
        anchors[row, : len(selected)] = selected
    return anchors


def build_sparse_swap_features_for_fields(
    instance: QAPInstance,
    perm: np.ndarray,
    pairs: np.ndarray,
    feature_fields: list[str],
    state: SparseSwapFeatureState,
) -> np.ndarray:
    """Reproduce the old 40 base features without all-pair feature matrices."""
    pairs = np.asarray(pairs, dtype=np.int64)
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError("pairs must have shape [num_pairs, 2]")
    if len(pairs) == 0:
        return np.zeros((0, len(feature_fields)), dtype=np.float32)
    unsupported = [field for field in feature_fields if field not in FEATURE_FIELDS]
    if unsupported:
        raise ValueError(f"sparse feature builder supports base fields only: {unsupported[:3]}")

    cache = state.cache
    F = cache.F
    D = cache.D
    i = pairs[:, 0]
    j = pairs[:, 1]
    p_i = perm[i]
    p_j = perm[j]
    pair_old = (
        F[i, i] * D[p_i, p_i]
        + F[j, j] * D[p_j, p_j]
        + F[i, j] * D[p_i, p_j]
        + F[j, i] * D[p_j, p_i]
    )
    pair_swap = (
        F[i, i] * D[p_j, p_j]
        + F[j, j] * D[p_i, p_i]
        + F[i, j] * D[p_j, p_i]
        + F[j, i] * D[p_i, p_j]
    )
    facility_cosine = np.einsum("ij,ij->i", F[i], F[j]) / (cache.f_norm[i] * cache.f_norm[j] + EPS)
    location_cosine = np.einsum("ij,ij->i", D[p_i], D[p_j]) / (
        cache.d_norm[p_i] * cache.d_norm[p_j] + EPS
    )
    facility_l1 = np.mean(np.abs(F[i] - F[j]), axis=1) / cache.f_abs_scale
    location_l1 = np.mean(np.abs(D[p_i] - D[p_j]), axis=1) / cache.d_abs_scale

    anchors = _anchors_for_pairs(cache, pairs)
    anchor_mask = anchors >= 0
    safe_anchors = np.where(anchor_mask, anchors, 0)
    anchor_locations = perm[safe_anchors]
    ii = i[:, None]
    jj = j[:, None]
    pii = p_i[:, None]
    pjj = p_j[:, None]
    old_terms = (
        F[ii, safe_anchors] * D[pii, anchor_locations]
        + F[safe_anchors, ii] * D[anchor_locations, pii]
        + F[jj, safe_anchors] * D[pjj, anchor_locations]
        + F[safe_anchors, jj] * D[anchor_locations, pjj]
    )
    swap_terms = (
        F[ii, safe_anchors] * D[pjj, anchor_locations]
        + F[safe_anchors, ii] * D[anchor_locations, pjj]
        + F[jj, safe_anchors] * D[pii, anchor_locations]
        + F[safe_anchors, jj] * D[anchor_locations, pii]
    )
    old_terms = np.where(anchor_mask, old_terms, 0.0)
    swap_terms = np.where(anchor_mask, swap_terms, 0.0)
    delta_terms = swap_terms - old_terms
    counts = anchor_mask.sum(axis=1)
    expansion = np.zeros(len(pairs), dtype=np.float64)
    active = counts > 0
    expansion[active] = max(len(perm) - 2, 1) / counts[active]
    scaled_delta = np.where(anchor_mask, delta_terms * expansion[:, None] / cache.qap_scale, 0.0)
    mean_delta = np.zeros(len(pairs), dtype=np.float64)
    mean_delta[active] = scaled_delta.sum(axis=1)[active] / counts[active]
    centered = np.where(anchor_mask, scaled_delta - mean_delta[:, None], 0.0)
    std_delta = np.zeros(len(pairs), dtype=np.float64)
    std_delta[active] = np.sqrt((centered * centered).sum(axis=1)[active] / counts[active])
    min_delta = np.zeros(len(pairs), dtype=np.float64)
    max_delta = np.zeros(len(pairs), dtype=np.float64)
    min_delta[active] = np.where(anchor_mask, scaled_delta, np.inf).min(axis=1)[active]
    max_delta[active] = np.where(anchor_mask, scaled_delta, -np.inf).max(axis=1)[active]

    features = np.zeros((len(pairs), len(FEATURE_NAMES)), dtype=np.float32)
    features[:, 0] = cache.f_sum[i]
    features[:, 1] = cache.f_sum[j]
    features[:, 2] = cache.f_mean[i]
    features[:, 3] = cache.f_mean[j]
    features[:, 4] = cache.f_std[i]
    features[:, 5] = cache.f_std[j]
    features[:, 6] = cache.d_sum[p_i]
    features[:, 7] = cache.d_sum[p_j]
    features[:, 8] = cache.d_mean[p_i]
    features[:, 9] = cache.d_mean[p_j]
    features[:, 10] = cache.d_std[p_i]
    features[:, 11] = cache.d_std[p_j]
    features[:, 12] = np.abs(cache.f_sum[i] - cache.f_sum[j])
    features[:, 13] = np.abs(cache.f_std[i] - cache.f_std[j])
    features[:, 14] = np.abs(cache.d_sum[p_i] - cache.d_sum[p_j])
    features[:, 15] = np.abs(cache.d_std[p_i] - cache.d_std[p_j])
    features[:, 16] = np.abs(i - j) / cache.denom
    features[:, 17] = np.abs(p_i - p_j) / cache.denom
    features[:, 18] = pair_old / cache.qap_scale
    features[:, 19] = pair_swap / cache.qap_scale
    features[:, 20] = (pair_swap - pair_old) / cache.qap_scale
    features[:, 21] = facility_cosine
    features[:, 22] = facility_l1
    features[:, 23] = location_cosine
    features[:, 24] = location_l1
    features[:, 25] = old_terms.sum(axis=1) * expansion / cache.qap_scale
    features[:, 26] = swap_terms.sum(axis=1) * expansion / cache.qap_scale
    features[:, 27] = delta_terms.sum(axis=1) * expansion / cache.qap_scale
    features[:, 28] = mean_delta
    features[:, 29] = std_delta
    features[:, 30] = min_delta
    features[:, 31] = max_delta
    features[:, 32] = p_i / cache.denom
    features[:, 33] = p_j / cache.denom
    features[:, 34] = np.abs(p_i - p_j) / cache.denom
    features[:, 35] = state.node_contrib[i]
    features[:, 36] = state.node_contrib[j]
    features[:, 37] = np.abs(state.node_contrib[i] - state.node_contrib[j])
    features[:, 38] = state.node_contrib[i] + state.node_contrib[j]
    features[:, 39] = np.log(float(instance.n))
    indices = [FEATURE_FIELDS.index(field) for field in feature_fields]
    return features[:, indices]


def _score_pairs(
    selector: SelectorBundle,
    instance: QAPInstance,
    perm: np.ndarray,
    pairs: np.ndarray,
    feature_state: SparseSwapFeatureState,
) -> np.ndarray:
    features = build_sparse_swap_features_for_fields(
        instance, perm, pairs, selector.feature_fields, feature_state
    )
    if selector.numpy_model is not None:
        return selector.numpy_model(features)
    normalized = (features - selector.feature_mean) / selector.feature_std
    selector.model.eval()
    device = next(selector.model.parameters()).device
    with torch.no_grad():
        return selector.model(
            torch.from_numpy(normalized).to(device=device, dtype=torch.float32)
        ).detach().cpu().numpy().astype(np.float64)


def _sample_pairs(all_pairs: np.ndarray, count: int | None, rng: np.random.Generator) -> np.ndarray:
    if count is None or int(count) >= len(all_pairs):
        return all_pairs
    indices = rng.choice(len(all_pairs), size=int(count), replace=False)
    return all_pairs[indices]


def sparse_candidate_bls(
    instance: QAPInstance,
    *,
    seed: int = 0,
    initial_perm: np.ndarray | None = None,
    max_time_sec: float | None = None,
    max_outer_iterations: int = 2_000_000_000,
    max_descent_decisions: int | None = None,
    stop_at_optimum: bool = False,
    method: str = "random",
    candidate_swaps: int = 32,
    score_pool_swaps: int = 64,
    perturb_pool_swaps: int = 64,
    perturbation_mode: str = "resample",
    perturb_candidate_multiplier: int = 8,
    unseen_pairs_admissible: bool = False,
    selector: SelectorBundle | None = None,
    candidate_pool_builder: Callable[
        [QAPInstance, np.ndarray, np.random.Generator, SparseSwapFeatureState], np.ndarray
    ]
    | None = None,
    learned_score_interval: int = 4,
    r1: float = 0.7,
    r2: float = 0.2,
    init_perturb_strength: float = 0.15,
    stagnation_threshold: int = 2500,
    p0: float = 0.75,
    q: float = 0.3,
) -> SparseBLSResult:
    """Run candidate-restricted BLS with exact on-demand swap deltas."""
    started = time.perf_counter()
    if method not in {"random", "learned"}:
        raise ValueError("sparse method must be random or learned")
    if method == "learned" and selector is None:
        raise ValueError("selector is required for learned sparse BLS")
    if candidate_swaps <= 0 or score_pool_swaps <= 0 or perturb_pool_swaps <= 0:
        raise ValueError("candidate and pool sizes must be positive")
    if perturbation_mode not in {"resample", "candidate_list"}:
        raise ValueError("perturbation_mode must be resample or candidate_list")
    if perturb_candidate_multiplier <= 0:
        raise ValueError("perturb_candidate_multiplier must be positive")
    if learned_score_interval <= 0:
        raise ValueError("learned_score_interval must be positive")

    flow, distance = _integer_matrices(instance)
    n = instance.n
    official_rng = CStdRand(int(seed))
    candidate_rng = np.random.default_rng(int(seed))
    perm = (
        _official_random_perm(n, official_rng)
        if initial_perm is None
        else np.asarray(initial_perm, dtype=np.int64).copy()
    )
    if sorted(perm.tolist()) != list(range(n)):
        raise ValueError("initial_perm must be a permutation of 0..n-1")
    current_cost = int(np.sum(flow.astype(object) * distance[np.ix_(perm, perm)].astype(object)))
    initial_cost = current_cost
    best_cost = current_cost
    best_perm = perm.copy()
    feature_state = SparseSwapFeatureState.create(instance, perm) if method == "learned" else None
    all_pairs = np.column_stack(np.triu_indices(n, k=1)).astype(np.int64)
    if unseen_pairs_admissible:
        unseen_age = math.ceil(n * r1) + max(int(n * r2), 1) + 1
        last_swapped = np.full((n, n), -unseen_age, dtype=np.int64)
        np.fill_diagonal(last_swapped, 0)
    else:
        last_swapped = np.zeros((n, n), dtype=np.int64)
    deadline = None if max_time_sec is None else started + float(max_time_sec)

    iteration = 0
    iter_without_improvement = 0
    perturb_strength = math.ceil(init_perturb_strength * n)
    learned_decisions = 0
    n_outer = n_moves = n_descent_decisions = n_descent = n_perturb = 0
    n_delta = n_score = n_directed = n_recency = n_random = n_noop = 0
    n_sparse_cache_updates = 0
    active_delta_pool: SparseDeltaPool | None = None

    def expired() -> bool:
        return deadline is not None and time.perf_counter() >= deadline

    def target_reached() -> bool:
        return bool(stop_at_optimum and instance.optimum is not None and best_cost <= int(instance.optimum))

    def apply_move(i: int, j: int, value: int, phase: str) -> None:
        nonlocal current_cost, best_cost, best_perm, iteration, iter_without_improvement
        nonlocal n_moves, n_descent, n_perturb, n_delta, n_sparse_cache_updates
        last_swapped[i, j] = last_swapped[j, i] = iteration
        if feature_state is None:
            perm[i], perm[j] = perm[j], perm[i]
        else:
            feature_state.apply_swap(perm, i, j)
        if active_delta_pool is not None:
            exact_count, update_count = active_delta_pool.update_after_swap(
                perm, flow, distance, i, j
            )
            n_delta += exact_count
            n_sparse_cache_updates += update_count
        current_cost += int(value)
        iteration += 1
        n_moves += 1
        if phase == "descent":
            n_descent += 1
        else:
            n_perturb += 1
        if current_cost < best_cost:
            best_cost = current_cost
            best_perm = perm.copy()
            iter_without_improvement = 0

    for _ in range(int(max_outer_iterations)):
        if expired() or target_reached():
            break
        n_outer += 1
        previous_cost = current_cost
        descent_num = 0
        while not expired():
            if max_descent_decisions is not None and n_descent_decisions >= int(max_descent_decisions):
                break
            score_now = method == "learned" and learned_decisions % int(learned_score_interval) == 0
            if score_now and candidate_pool_builder is not None:
                assert feature_state is not None
                pool = np.asarray(
                    candidate_pool_builder(instance, perm, candidate_rng, feature_state),
                    dtype=np.int64,
                )
                if pool.ndim != 2 or pool.shape[1] != 2 or len(pool) == 0:
                    raise ValueError("candidate_pool_builder must return nonempty [num_pairs, 2]")
                if np.any(pool[:, 0] < 0) or np.any(pool[:, 1] >= n) or np.any(pool[:, 0] >= pool[:, 1]):
                    raise ValueError("candidate_pool_builder returned invalid canonical pairs")
                if len(np.unique(pool, axis=0)) != len(pool):
                    raise ValueError("candidate_pool_builder returned duplicate pairs")
            else:
                pool = _sample_pairs(all_pairs, score_pool_swaps, candidate_rng)
            k = min(int(candidate_swaps), len(pool))
            if score_now:
                assert selector is not None and feature_state is not None
                scores = _score_pairs(selector, instance, perm, pool, feature_state)
                chosen = np.argpartition(-scores, kth=k - 1)[:k]
                chosen = chosen[np.argsort(-scores[chosen])]
                candidates = pool[chosen]
                n_score += len(pool)
            else:
                chosen = candidate_rng.choice(len(pool), size=k, replace=False)
                candidates = pool[chosen]
            learned_decisions += int(method == "learned")
            values = _compute_deltas_for_pairs(perm, flow, distance, candidates)
            n_delta += len(candidates)
            n_descent_decisions += 1
            selected = int(np.argmin(values))
            if int(values[selected]) >= 0:
                break
            i, j = int(candidates[selected, 0]), int(candidates[selected, 1])
            apply_move(i, j, int(values[selected]), "descent")
            descent_num += 1

        if expired() or (
            max_descent_decisions is not None and n_descent_decisions >= int(max_descent_decisions)
        ):
            break
        if iter_without_improvement > stagnation_threshold:
            iter_without_improvement = 0
            perturb_strength = n * (0.4 + official_rng.rand() % 20 / 100.0)
        elif descent_num != 0 and previous_cost != current_cost:
            iter_without_improvement += 1
            perturb_strength = max(math.ceil(init_perturb_strength * n), 5)
        elif previous_cost == current_cost:
            perturb_strength += 1

        d = float(iter_without_improvement) / float(stagnation_threshold)
        probability = max(math.exp(-d), p0)
        directed = probability > official_rng.unit_101()
        init_cost = previous_cost
        if perturbation_mode == "candidate_list":
            list_size = min(
                len(all_pairs),
                max(int(perturb_pool_swaps), int(perturb_candidate_multiplier) * n),
            )
            persistent_pairs = _sample_pairs(all_pairs, list_size, candidate_rng)
            active_delta_pool = SparseDeltaPool.create(perm, flow, distance, persistent_pairs)
            n_delta += len(persistent_pairs)
        for _ in range(int(math.ceil(perturb_strength))):
            if expired():
                break
            if active_delta_pool is None:
                pool = _sample_pairs(all_pairs, perturb_pool_swaps, candidate_rng)
                cached_values = None
            else:
                pool = active_delta_pool.pairs
                cached_values = active_delta_pool.values
            moved = False
            if directed:
                if cached_values is None:
                    values = _compute_deltas_for_pairs(perm, flow, distance, pool)
                    n_delta += len(pool)
                else:
                    values = cached_values
                tenure_noise = max(int(n * r2), 1)
                next_costs = current_cost + values
                admissible = (next_costs != init_cost) & (
                    (last_swapped[pool[:, 0], pool[:, 1]] + n * r1 + official_rng.rand() % tenure_noise < iteration)
                    | (next_costs < best_cost)
                )
                indices = np.flatnonzero(admissible)
                if len(indices):
                    selected = int(indices[np.argmin(values[indices])])
                    i, j = int(pool[selected, 0]), int(pool[selected, 1])
                    apply_move(i, j, int(values[selected]), "perturb")
                    n_directed += 1
                    moved = True
            elif q > official_rng.unit_101():
                ages = last_swapped[pool[:, 0], pool[:, 1]]
                selected = int(np.argmin(ages))
                pair = pool[selected : selected + 1]
                if cached_values is None:
                    value = int(_compute_deltas_for_pairs(perm, flow, distance, pair)[0])
                    n_delta += 1
                else:
                    value = int(cached_values[selected])
                i, j = int(pair[0, 0]), int(pair[0, 1])
                apply_move(i, j, value, "perturb")
                n_recency += 1
                moved = True
            else:
                if cached_values is None:
                    values = _compute_deltas_for_pairs(perm, flow, distance, pool)
                    n_delta += len(pool)
                else:
                    values = cached_values
                valid = np.flatnonzero(current_cost + values != init_cost)
                if len(valid):
                    selected = int(valid[official_rng.rand() % len(valid)])
                    i, j = int(pool[selected, 0]), int(pool[selected, 1])
                    apply_move(i, j, int(values[selected]), "perturb")
                    n_random += 1
                    moved = True
            n_noop += int(not moved)
        active_delta_pool = None
        if target_reached():
            break

    runtime = time.perf_counter() - started
    return SparseBLSResult(
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
        n_directed_moves=int(n_directed),
        n_recency_moves=int(n_recency),
        n_random_moves=int(n_random),
        n_noop_moves=int(n_noop),
        seed=int(seed),
        method=str(method),
        sparse_delta_cache_updates=int(n_sparse_cache_updates),
    )
