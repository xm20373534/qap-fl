from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np

from .features import FEATURE_FIELDS, SwapFeatureCache, all_swap_pairs, build_swap_feature_cache, build_swap_features, sample_swap_pairs
from .local_search import SelectorBundle
from .qap import QAPInstance, compute_cost, random_perm


ROTS_EXTRA_FEATURE_FIELDS = [
    "feature_rots_delta",
    "feature_rots_delta_abs",
    "feature_rots_delta_rank",
    "feature_rots_delta_percentile",
    "feature_rots_delta_negative",
    "feature_rots_authorized",
    "feature_rots_aspired",
    "feature_rots_aspired_by_cost",
    "feature_rots_tabu_remaining_i",
    "feature_rots_tabu_remaining_j",
    "feature_rots_tabu_remaining_min",
    "feature_rots_tabu_remaining_max",
    "feature_rots_current_cost",
    "feature_rots_best_cost",
    "feature_rots_current_minus_best",
    "feature_rots_iteration",
]
ROTS_FEATURE_FIELDS = FEATURE_FIELDS + ROTS_EXTRA_FEATURE_FIELDS
BASE_DELTA_LEAKAGE_FEATURE_FIELDS = [
    "feature_qap_pair_old",
    "feature_qap_pair_swap",
    "feature_qap_pair_delta",
    "feature_anchor_external_old",
    "feature_anchor_external_swap",
    "feature_anchor_external_delta_estimate",
    "feature_anchor_delta_mean",
    "feature_anchor_delta_std",
    "feature_anchor_delta_min",
    "feature_anchor_delta_max",
]
ROTS_DIRECT_DELTA_LEAKAGE_FEATURE_FIELDS = [
    "feature_rots_delta",
    "feature_rots_delta_abs",
    "feature_rots_delta_rank",
    "feature_rots_delta_percentile",
    "feature_rots_delta_negative",
    "feature_rots_aspired",
    "feature_rots_aspired_by_cost",
]
ROTS_DELTA_LEAKAGE_FEATURE_FIELDS = ROTS_DIRECT_DELTA_LEAKAGE_FEATURE_FIELDS
ROTS_STRICT_DELTA_LEAKAGE_FEATURE_FIELDS = BASE_DELTA_LEAKAGE_FEATURE_FIELDS + ROTS_DIRECT_DELTA_LEAKAGE_FEATURE_FIELDS
ROTS_NO_DELTA_FEATURE_FIELDS = [
    field for field in ROTS_FEATURE_FIELDS if field not in set(ROTS_DELTA_LEAKAGE_FEATURE_FIELDS)
]
ROTS_STRICT_NO_DELTA_FEATURE_FIELDS = [
    field for field in ROTS_FEATURE_FIELDS if field not in set(ROTS_STRICT_DELTA_LEAKAGE_FEATURE_FIELDS)
]

@dataclass(frozen=True)
class ROTSResult:
    perm: np.ndarray
    cost: float
    n_iters: int
    n_delta_evals: int
    n_score_evals: int
    method: str


def compute_delta_taillard(perm: np.ndarray, F: np.ndarray, D: np.ndarray, i: int, j: int) -> float:
    p = np.asarray(perm, dtype=np.int64)
    pi = int(p[i])
    pj = int(p[j])
    delta = (F[i, i] - F[j, j]) * (D[pj, pj] - D[pi, pi]) + (F[i, j] - F[j, i]) * (D[pj, pi] - D[pi, pj])
    n = len(p)
    for k in range(n):
        if k == i or k == j:
            continue
        pk = int(p[k])
        delta += (F[k, i] - F[k, j]) * (D[pk, pj] - D[pk, pi])
        delta += (F[i, k] - F[j, k]) * (D[pj, pk] - D[pi, pk])
    return float(delta)


def compute_delta_part_taillard(
    perm_after: np.ndarray,
    F: np.ndarray,
    D: np.ndarray,
    delta_before: np.ndarray,
    i: int,
    j: int,
    r: int,
    s: int,
) -> float:
    p = np.asarray(perm_after, dtype=np.int64)
    return float(
        delta_before[i, j]
        + (F[r, i] - F[r, j] + F[s, j] - F[s, i])
        * (D[p[s], p[i]] - D[p[s], p[j]] + D[p[r], p[j]] - D[p[r], p[i]])
        + (F[i, r] - F[j, r] + F[j, s] - F[i, s])
        * (D[p[i], p[s]] - D[p[j], p[s]] + D[p[j], p[r]] - D[p[i], p[r]])
    )


def initialize_delta_matrix(perm: np.ndarray, F: np.ndarray, D: np.ndarray) -> np.ndarray:
    n = len(perm)
    delta = np.zeros((n, n), dtype=np.float64)
    for i in range(n - 1):
        for j in range(i + 1, n):
            value = compute_delta_taillard(perm, F, D, i, j)
            delta[i, j] = value
            delta[j, i] = value
    np.fill_diagonal(delta, np.inf)
    return delta


def update_delta_matrix_after_swap(
    delta: np.ndarray,
    perm_after: np.ndarray,
    F: np.ndarray,
    D: np.ndarray,
    r: int,
    s: int,
) -> np.ndarray:
    before = delta.copy()
    n = len(perm_after)
    for i in range(n - 1):
        for j in range(i + 1, n):
            if i != r and i != s and j != r and j != s:
                value = compute_delta_part_taillard(perm_after, F, D, before, i, j, r, s)
            else:
                value = compute_delta_taillard(perm_after, F, D, i, j)
            delta[i, j] = value
            delta[j, i] = value
    np.fill_diagonal(delta, np.inf)
    return delta


def rots_move_flags(
    perm: np.ndarray,
    delta: np.ndarray,
    tabu_list: np.ndarray,
    current_iteration: int,
    current_cost: float,
    best_cost: float,
    aspiration: int,
    pairs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    i = pairs[:, 0].astype(np.int64)
    j = pairs[:, 1].astype(np.int64)
    values = delta[i, j].astype(np.float64)
    tabu_i = tabu_list[i, perm[j]].astype(np.int64)
    tabu_j = tabu_list[j, perm[i]].astype(np.int64)
    authorized = (tabu_i < current_iteration) | (tabu_j < current_iteration)
    aspired_by_age = (tabu_i < current_iteration - aspiration) | (tabu_j < current_iteration - aspiration)
    aspired_by_cost = current_cost + values < best_cost
    aspired = aspired_by_age | aspired_by_cost
    return values, authorized, aspired, aspired_by_cost, np.stack([tabu_i, tabu_j], axis=1)


def build_rots_feature_matrix(
    instance: QAPInstance,
    perm: np.ndarray,
    pairs: np.ndarray,
    delta: np.ndarray,
    tabu_list: np.ndarray,
    current_iteration: int,
    current_cost: float,
    best_cost: float,
    aspiration: int,
    tabu_duration: int,
    feature_fields: list[str] | None = None,
    feature_cache: SwapFeatureCache | None = None,
) -> np.ndarray:
    requested = ROTS_FEATURE_FIELDS if feature_fields is None else list(feature_fields)
    cache = feature_cache if feature_cache is not None else build_swap_feature_cache(instance)
    base_features = None
    base_index = {name: idx for idx, name in enumerate(FEATURE_FIELDS)}

    values, authorized, aspired, aspired_by_cost, tabu_until = rots_move_flags(
        perm=perm,
        delta=delta,
        tabu_list=tabu_list,
        current_iteration=current_iteration,
        current_cost=current_cost,
        best_cost=best_cost,
        aspiration=aspiration,
        pairs=pairs,
    )
    order = np.argsort(values)
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    rank_denom = float(max(len(values) - 1, 1))
    delta_scale = float(cache.qap_scale) + 1e-9
    cost_scale = delta_scale * max(instance.n, 1)
    remaining = np.maximum(tabu_until - int(current_iteration), 0).astype(np.float64) / float(max(int(tabu_duration), 1))
    extra = {
        "feature_rots_delta": values / delta_scale,
        "feature_rots_delta_abs": np.abs(values) / delta_scale,
        "feature_rots_delta_rank": ranks / rank_denom,
        "feature_rots_delta_percentile": ranks / rank_denom,
        "feature_rots_delta_negative": (values < 0).astype(np.float64),
        "feature_rots_authorized": authorized.astype(np.float64),
        "feature_rots_aspired": aspired.astype(np.float64),
        "feature_rots_aspired_by_cost": aspired_by_cost.astype(np.float64),
        "feature_rots_tabu_remaining_i": remaining[:, 0],
        "feature_rots_tabu_remaining_j": remaining[:, 1],
        "feature_rots_tabu_remaining_min": np.minimum(remaining[:, 0], remaining[:, 1]),
        "feature_rots_tabu_remaining_max": np.maximum(remaining[:, 0], remaining[:, 1]),
        "feature_rots_current_cost": np.full(len(pairs), float(current_cost) / cost_scale, dtype=np.float64),
        "feature_rots_best_cost": np.full(len(pairs), float(best_cost) / cost_scale, dtype=np.float64),
        "feature_rots_current_minus_best": np.full(len(pairs), (float(current_cost) - float(best_cost)) / cost_scale, dtype=np.float64),
        "feature_rots_iteration": np.full(len(pairs), float(current_iteration) / float(max(5 * instance.n * instance.n, 1)), dtype=np.float64),
    }

    columns = []
    for field in requested:
        if field in base_index:
            if base_features is None:
                base_features = build_swap_features(instance, perm, pairs, feature_cache=cache)
            columns.append(base_features[:, base_index[field]].astype(np.float64))
        elif field in extra:
            columns.append(extra[field])
        else:
            raise ValueError(f"unsupported Ro-TS feature field: {field}")
    return np.stack(columns, axis=1).astype(np.float32)


def build_rots_sparse_feature_matrix(
    instance: QAPInstance,
    perm: np.ndarray,
    pairs: np.ndarray,
    values: np.ndarray,
    tabu_list: np.ndarray,
    current_iteration: int,
    current_cost: float,
    best_cost: float,
    aspiration: int,
    tabu_duration: int,
    feature_fields: list[str],
    feature_cache: SwapFeatureCache | None = None,
) -> np.ndarray:
    """Build selector features from candidate-only exact deltas.

    The sparse Gate 2 path intentionally accepts only the values for ``pairs``;
    it never allocates or updates a dense n x n delta matrix.
    """
    requested = list(feature_fields)
    cache = feature_cache if feature_cache is not None else build_swap_feature_cache(instance)
    values = np.asarray(values, dtype=np.float64)
    i = pairs[:, 0].astype(np.int64)
    j = pairs[:, 1].astype(np.int64)
    tabu_i = tabu_list[i, perm[j]].astype(np.int64)
    tabu_j = tabu_list[j, perm[i]].astype(np.int64)
    authorized = (tabu_i < current_iteration) | (tabu_j < current_iteration)
    aspired_by_age = (tabu_i < current_iteration - aspiration) | (tabu_j < current_iteration - aspiration)
    delta_scale = float(cache.qap_scale) + 1e-9
    cost_scale = delta_scale * max(instance.n, 1)
    remaining = np.maximum(np.stack([tabu_i, tabu_j], axis=1) - int(current_iteration), 0).astype(np.float64) / float(max(int(tabu_duration), 1))
    extra = {
        "feature_rots_authorized": authorized.astype(np.float64),
        "feature_rots_aspired": (aspired_by_age | (current_cost + values < best_cost)).astype(np.float64),
        "feature_rots_tabu_remaining_i": remaining[:, 0],
        "feature_rots_tabu_remaining_j": remaining[:, 1],
        "feature_rots_tabu_remaining_min": np.minimum(remaining[:, 0], remaining[:, 1]),
        "feature_rots_tabu_remaining_max": np.maximum(remaining[:, 0], remaining[:, 1]),
        "feature_rots_current_cost": np.full(len(pairs), float(current_cost) / cost_scale, dtype=np.float64),
        "feature_rots_best_cost": np.full(len(pairs), float(best_cost) / cost_scale, dtype=np.float64),
        "feature_rots_current_minus_best": np.full(len(pairs), (float(current_cost) - float(best_cost)) / cost_scale, dtype=np.float64),
        "feature_rots_iteration": np.full(len(pairs), float(current_iteration) / float(max(5 * instance.n * instance.n, 1)), dtype=np.float64),
    }
    # Sparse Gate 2 forbids all direct-delta and delta-estimate fields.
    forbidden = set(ROTS_STRICT_DELTA_LEAKAGE_FEATURE_FIELDS)
    if forbidden.intersection(requested):
        raise ValueError("sparse Ro-TS selector cannot use delta leakage features")
    base_features = None
    base_index = {name: idx for idx, name in enumerate(FEATURE_FIELDS)}
    columns = []
    for field in requested:
        if field in base_index:
            if base_features is None:
                base_features = build_swap_features(instance, perm, pairs, feature_cache=cache)
            columns.append(base_features[:, base_index[field]].astype(np.float64))
        elif field in extra:
            columns.append(extra[field])
        else:
            raise ValueError(f"unsupported sparse Ro-TS feature field: {field}")
    return np.stack(columns, axis=1).astype(np.float32)


def _score_pairs(
    selector: SelectorBundle,
    instance: QAPInstance,
    perm: np.ndarray,
    pairs: np.ndarray,
    delta: np.ndarray,
    tabu_list: np.ndarray,
    current_iteration: int,
    current_cost: float,
    best_cost: float,
    aspiration: int,
    tabu_duration: int,
    feature_cache: SwapFeatureCache,
) -> np.ndarray:
    features = build_rots_feature_matrix(
        instance=instance,
        perm=perm,
        pairs=pairs,
        delta=delta,
        tabu_list=tabu_list,
        current_iteration=current_iteration,
        current_cost=current_cost,
        best_cost=best_cost,
        aspiration=aspiration,
        tabu_duration=tabu_duration,
        feature_fields=selector.feature_fields,
        feature_cache=feature_cache,
    )
    features = (features - selector.feature_mean) / selector.feature_std
    import torch

    device = next(selector.model.parameters()).device
    x = torch.from_numpy(features).to(device=device, dtype=torch.float32)
    selector.model.eval()
    with torch.no_grad():
        return selector.model(x).detach().cpu().numpy().astype(np.float64)


def _select_candidate_pairs(
    method: str,
    instance: QAPInstance,
    perm: np.ndarray,
    delta: np.ndarray,
    tabu_list: np.ndarray,
    current_iteration: int,
    current_cost: float,
    best_cost: float,
    aspiration: int,
    tabu_duration: int,
    rng: np.random.Generator,
    candidate_swaps: int | None,
    score_pool_swaps: int | None,
    selector: SelectorBundle | None,
    feature_cache: SwapFeatureCache | None,
) -> tuple[np.ndarray, int]:
    if method == "full":
        return all_swap_pairs(instance.n), 0
    pool = sample_swap_pairs(instance.n, score_pool_swaps, rng)
    k = len(pool) if candidate_swaps is None else min(int(candidate_swaps), len(pool))
    if method == "random":
        if k >= len(pool):
            return pool[rng.permutation(len(pool))], 0
        return pool[rng.choice(len(pool), size=k, replace=False)], 0
    if method == "learned":
        if selector is None:
            raise ValueError("learned Ro-TS requires selector bundle.")
        if feature_cache is None:
            raise ValueError("learned Ro-TS requires feature cache.")
        scores = _score_pairs(
            selector=selector,
            instance=instance,
            perm=perm,
            pairs=pool,
            delta=delta,
            tabu_list=tabu_list,
            current_iteration=current_iteration,
            current_cost=current_cost,
            best_cost=best_cost,
            aspiration=aspiration,
            tabu_duration=tabu_duration,
            feature_cache=feature_cache,
        )
        if k >= len(pool):
            idx = np.argsort(-scores)
        else:
            idx = np.argpartition(-scores, kth=k - 1)[:k]
            idx = idx[np.argsort(-scores[idx])]
        return pool[idx], len(pool)
    if method == "delta_oracle":
        return pool, len(pool)
    raise ValueError(f"unknown method: {method}")


def _choose_rots_move(
    perm: np.ndarray,
    delta: np.ndarray,
    tabu_list: np.ndarray,
    current_iteration: int,
    current_cost: float,
    best_cost: float,
    aspiration: int,
    pairs: np.ndarray,
    method: str,
    candidate_swaps: int | None,
) -> tuple[int, int, float] | None:
    chosen_i = -1
    chosen_j = -1
    min_delta = float("inf")
    already_aspired = False
    selected_pairs = pairs
    if method == "delta_oracle":
        values, authorized, aspired, _, _ = rots_move_flags(
            perm=perm,
            delta=delta,
            tabu_list=tabu_list,
            current_iteration=current_iteration,
            current_cost=current_cost,
            best_cost=best_cost,
            aspiration=aspiration,
            pairs=pairs,
        )
        allowed_mask = authorized | aspired
        if not bool(np.any(allowed_mask)):
            return None
        allowed_pairs = pairs[allowed_mask]
        allowed_values = values[allowed_mask]
        k = len(allowed_pairs) if candidate_swaps is None else min(int(candidate_swaps), len(allowed_pairs))
        selected_pairs = allowed_pairs[np.argsort(allowed_values)[:k]]

    for i_raw, j_raw in selected_pairs:
        i = int(i_raw)
        j = int(j_raw)
        value = float(delta[i, j])
        authorized = (tabu_list[i, perm[j]] < current_iteration) or (tabu_list[j, perm[i]] < current_iteration)
        aspired = (
            (tabu_list[i, perm[j]] < current_iteration - aspiration)
            or (tabu_list[j, perm[i]] < current_iteration - aspiration)
            or (current_cost + value < best_cost)
        )
        if (
            (aspired and not already_aspired)
            or (aspired and already_aspired and value < min_delta)
            or ((not aspired) and (not already_aspired) and value < min_delta and authorized)
        ):
            chosen_i = i
            chosen_j = j
            min_delta = value
            if aspired:
                already_aspired = True
    if chosen_i < 0:
        return None
    return chosen_i, chosen_j, min_delta


def rots_search(
    instance: QAPInstance,
    initial_perm: np.ndarray,
    method: str,
    n_iterations: int,
    seed: int,
    candidate_swaps: int | None = None,
    score_pool_swaps: int | None = None,
    selector: SelectorBundle | None = None,
    tabu_duration: int | None = None,
    aspiration: int | None = None,
    stop_at_optimum: bool = True,
    tenure_rng=None,
    max_seconds: float | None = None,
) -> ROTSResult:
    started = time.perf_counter()
    n = instance.n
    F = np.asarray(instance.F, dtype=np.float64)
    D = np.asarray(instance.D, dtype=np.float64)
    tabu_duration = int(8 * n if tabu_duration is None else tabu_duration)
    aspiration = int(5 * n * n if aspiration is None else aspiration)
    rng = np.random.default_rng(seed)
    tenure_rng = rng if tenure_rng is None else tenure_rng

    perm = np.asarray(initial_perm, dtype=np.int64).copy()
    current_cost = compute_cost(perm, F, D)
    best_cost = current_cost
    best_perm = perm.copy()
    delta = initialize_delta_matrix(perm, F, D)
    tabu_list = np.zeros((n, n), dtype=np.int64)
    for i in range(n):
        for j in range(n):
            tabu_list[i, j] = -(n * i + j)
    feature_cache = build_swap_feature_cache(instance) if method == "learned" else None

    n_delta = 0
    n_score = 0
    completed = 0
    for current_iteration in range(1, int(n_iterations) + 1):
        if max_seconds is not None and time.perf_counter() - started >= max_seconds:
            break
        if stop_at_optimum and instance.optimum is not None and best_cost <= float(instance.optimum):
            break
        completed = current_iteration
        pairs, score_evals = _select_candidate_pairs(
            method=method,
            instance=instance,
            perm=perm,
            delta=delta,
            tabu_list=tabu_list,
            current_iteration=current_iteration,
            current_cost=current_cost,
            best_cost=best_cost,
            aspiration=aspiration,
            tabu_duration=tabu_duration,
            rng=rng,
            candidate_swaps=candidate_swaps,
            score_pool_swaps=score_pool_swaps,
            selector=selector,
            feature_cache=feature_cache,
        )
        n_score += int(score_evals)
        move = _choose_rots_move(
            perm=perm,
            delta=delta,
            tabu_list=tabu_list,
            current_iteration=current_iteration,
            current_cost=current_cost,
            best_cost=best_cost,
            aspiration=aspiration,
            pairs=pairs,
            method=method,
            candidate_swaps=candidate_swaps,
        )
        if move is None:
            break
        i, j, move_delta = move
        n_delta += len(pairs) if method != "delta_oracle" else min(int(candidate_swaps or len(pairs)), len(pairs))

        perm[i], perm[j] = perm[j], perm[i]
        current_cost += move_delta
        tabu_list[i, perm[j]] = current_iteration + int((tenure_rng.random() ** 3) * tabu_duration)
        tabu_list[j, perm[i]] = current_iteration + int((tenure_rng.random() ** 3) * tabu_duration)
        if current_cost < best_cost:
            best_cost = current_cost
            best_perm = perm.copy()
        update_delta_matrix_after_swap(delta, perm, F, D, i, j)

    return ROTSResult(
        perm=best_perm,
        cost=float(best_cost),
        n_iters=completed,
        n_delta_evals=n_delta,
        n_score_evals=n_score,
        method=method,
    )


def random_multistart_rots(
    instance: QAPInstance,
    method: str,
    n_starts: int,
    n_iterations: int,
    seed: int,
    candidate_swaps: int | None = None,
    score_pool_swaps: int | None = None,
    selector: SelectorBundle | None = None,
    tabu_duration: int | None = None,
    aspiration: int | None = None,
    stop_at_optimum: bool = True,
) -> ROTSResult:
    rng = np.random.default_rng(seed)
    results: list[ROTSResult] = []
    for _ in range(int(n_starts)):
        start = random_perm(instance.n, rng)
        search_seed = int(rng.integers(0, 2**32 - 1))
        results.append(
            rots_search(
                instance=instance,
                initial_perm=start,
                method=method,
                n_iterations=n_iterations,
                seed=search_seed,
                candidate_swaps=candidate_swaps,
                score_pool_swaps=score_pool_swaps,
                selector=selector if method == "learned" else None,
                tabu_duration=tabu_duration,
                aspiration=aspiration,
                stop_at_optimum=stop_at_optimum,
            )
        )
    if not results:
        raise ValueError("n_starts must be positive.")
    best = min(results, key=lambda item: item.cost)
    return ROTSResult(
        perm=best.perm,
        cost=best.cost,
        n_iters=sum(item.n_iters for item in results),
        n_delta_evals=sum(item.n_delta_evals for item in results),
        n_score_evals=sum(item.n_score_evals for item in results),
        method=method,
    )


def sparse_rots_search(
    instance: QAPInstance,
    initial_perm: np.ndarray,
    method: str,
    n_iterations: int,
    seed: int,
    candidate_swaps: int = 32,
    score_pool_swaps: int = 64,
    selector: SelectorBundle | None = None,
    tabu_duration: int | None = None,
    aspiration: int | None = None,
    stop_at_optimum: bool = True,
    max_seconds: float | None = None,
) -> ROTSResult:
    """Candidate-filtered Ro-TS without a dense delta matrix."""
    if method not in {"random", "learned", "delta_oracle"}:
        raise ValueError("sparse Ro-TS supports random, learned and delta_oracle")
    n = instance.n
    tabu_duration = int(8 * n if tabu_duration is None else tabu_duration)
    aspiration = int(5 * n * n if aspiration is None else aspiration)
    rng = np.random.default_rng(seed)
    perm = np.asarray(initial_perm, dtype=np.int64).copy()
    F = np.asarray(instance.F, dtype=np.float64)
    D = np.asarray(instance.D, dtype=np.float64)
    current_cost = compute_cost(perm, F, D)
    best_cost = current_cost
    best_perm = perm.copy()
    tabu_list = np.zeros((n, n), dtype=np.int64)
    for i in range(n):
        for j in range(n):
            tabu_list[i, j] = -(n * i + j)
    feature_cache = build_swap_feature_cache(instance)
    n_delta = 0
    n_score = 0
    completed = 0
    time_start = __import__("time").perf_counter()
    all_pairs = all_swap_pairs(n)
    for current_iteration in range(1, int(n_iterations) + 1):
        if max_seconds is not None and __import__("time").perf_counter() - time_start >= float(max_seconds):
            break
        if stop_at_optimum and instance.optimum is not None and best_cost <= float(instance.optimum):
            break
        pool_size = min(int(score_pool_swaps), len(all_pairs))
        pool = all_pairs[rng.choice(len(all_pairs), size=pool_size, replace=False)]
        k = min(int(candidate_swaps), len(pool))
        if method == "random":
            selected_pairs = pool[rng.choice(len(pool), size=k, replace=False)]
        elif method == "delta_oracle":
            pool_values = np.asarray([compute_delta_taillard(perm, F, D, int(i), int(j)) for i, j in pool], dtype=np.float64)
            n_delta += len(pool)
            selected_pairs = pool[np.argsort(pool_values)[:k]]
        else:
            if selector is None:
                raise ValueError("sparse learned Ro-TS requires selector")
            features = build_rots_sparse_feature_matrix(
                instance, perm, pool, np.zeros(len(pool), dtype=np.float64), tabu_list, current_iteration,
                current_cost, best_cost, aspiration, tabu_duration,
                selector.feature_fields, feature_cache,
            )
            features = (features - selector.feature_mean) / selector.feature_std
            import torch
            device = next(selector.model.parameters()).device
            x = torch.from_numpy(features).to(device=device, dtype=torch.float32)
            selector.model.eval()
            with torch.no_grad():
                scores = selector.model(x).detach().cpu().numpy().astype(np.float64)
            n_score += len(pool)
            selected_pairs = pool[np.argsort(-scores)[:k]]
        values = np.asarray([compute_delta_taillard(perm, F, D, int(i), int(j)) for i, j in selected_pairs], dtype=np.float64)
        n_delta += len(selected_pairs)
        i_arr = selected_pairs[:, 0].astype(np.int64)
        j_arr = selected_pairs[:, 1].astype(np.int64)
        tabu_i = tabu_list[i_arr, perm[j_arr]]
        tabu_j = tabu_list[j_arr, perm[i_arr]]
        authorized = (tabu_i < current_iteration) | (tabu_j < current_iteration)
        aspired = (tabu_i < current_iteration - aspiration) | (tabu_j < current_iteration - aspiration) | (current_cost + values < best_cost)
        allowed = authorized | aspired
        if not np.any(allowed):
            break
        allowed_idx = np.flatnonzero(allowed)
        asp_idx = allowed_idx[aspired[allowed_idx]]
        choose_idx = int(asp_idx[np.argmin(values[asp_idx])] if len(asp_idx) else allowed_idx[np.argmin(values[allowed_idx])])
        i, j = int(i_arr[choose_idx]), int(j_arr[choose_idx])
        move_delta = float(values[choose_idx])
        completed = current_iteration
        old_i, old_j = int(perm[i]), int(perm[j])
        perm[i], perm[j] = perm[j], perm[i]
        current_cost += move_delta
        tabu_list[i, old_j] = current_iteration + int((rng.random() ** 3) * tabu_duration)
        tabu_list[j, old_i] = current_iteration + int((rng.random() ** 3) * tabu_duration)
        if current_cost < best_cost:
            best_cost = current_cost
            best_perm = perm.copy()
    return ROTSResult(best_perm, float(best_cost), completed, n_delta, n_score, method)


def sparse_random_multistart_rots(
    instance: QAPInstance, method: str, n_starts: int, n_iterations: int, seed: int,
    candidate_swaps: int = 32, score_pool_swaps: int = 64, selector: SelectorBundle | None = None,
    tabu_duration: int | None = None, aspiration: int | None = None, stop_at_optimum: bool = True,
    max_seconds: float | None = None,
) -> ROTSResult:
    rng = np.random.default_rng(seed)
    results = []
    for _ in range(int(n_starts)):
        start = random_perm(instance.n, rng)
        search_seed = int(rng.integers(0, 2**32 - 1))
        results.append(sparse_rots_search(instance, start, method, n_iterations, search_seed,
                                          candidate_swaps, score_pool_swaps, selector,
                                          tabu_duration, aspiration, stop_at_optimum, max_seconds))
        if max_seconds is not None:
            break
    best = min(results, key=lambda item: item.cost)
    return ROTSResult(best.perm, best.cost, sum(x.n_iters for x in results),
                      sum(x.n_delta_evals for x in results), sum(x.n_score_evals for x in results), method)


def check_delta_update(instance: QAPInstance, trials: int = 20, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    perm = random_perm(instance.n, rng)
    delta = initialize_delta_matrix(perm, instance.F, instance.D)
    for _ in range(int(trials)):
        i, j = rng.choice(instance.n, 2, replace=False)
        i = int(i)
        j = int(j)
        if i > j:
            i, j = j, i
        old_cost = compute_cost(perm, instance.F, instance.D)
        move_delta = float(delta[i, j])
        perm[i], perm[j] = perm[j], perm[i]
        new_cost = compute_cost(perm, instance.F, instance.D)
        if not np.isclose(old_cost + move_delta, new_cost, rtol=1e-8, atol=1e-8):
            raise AssertionError(f"cost delta mismatch: {old_cost} + {move_delta} != {new_cost}")
        update_delta_matrix_after_swap(delta, perm, instance.F, instance.D, i, j)
        exact = initialize_delta_matrix(perm, instance.F, instance.D)
        if not np.allclose(delta, exact, rtol=1e-8, atol=1e-8):
            raise AssertionError("delta matrix update mismatch")


