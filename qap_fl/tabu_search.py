from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .features import SwapFeatureCache, build_swap_feature_cache, sample_swap_pairs
from .local_search import SelectorBundle, select_pairs
from .qap import QAPInstance, compute_cost, random_perm, swap_delta_cost


@dataclass(frozen=True)
class TabuSearchResult:
    perm: np.ndarray
    cost: float
    n_iters: int
    n_delta_evals: int
    n_score_evals: int
    method: str


def _tenure(tabu_duration: int, rng: np.random.Generator) -> int:
    return 1 + int((rng.random() ** 3) * max(int(tabu_duration), 1))


def _is_allowed(
    perm: np.ndarray,
    tabu_list: np.ndarray,
    iteration: int,
    current_cost: float,
    best_cost: float,
    delta: float,
    i: int,
    j: int,
) -> bool:
    new_i_location = int(perm[j])
    new_j_location = int(perm[i])
    authorized = (tabu_list[i, new_i_location] <= iteration) or (tabu_list[j, new_j_location] <= iteration)
    aspired = current_cost + delta < best_cost
    return bool(authorized or aspired)


def _select_delta_oracle_allowed(
    instance: QAPInstance,
    perm: np.ndarray,
    tabu_list: np.ndarray,
    iteration: int,
    current_cost: float,
    best_cost: float,
    rng: np.random.Generator,
    candidate_swaps: int | None,
    score_pool_swaps: int | None,
) -> tuple[np.ndarray, np.ndarray, int]:
    pool = sample_swap_pairs(instance.n, score_pool_swaps, rng)
    allowed_pairs: list[tuple[int, int]] = []
    allowed_deltas: list[float] = []
    for i_raw, j_raw in pool:
        i = int(i_raw)
        j = int(j_raw)
        delta = swap_delta_cost(perm, instance.F, instance.D, i, j)
        if _is_allowed(perm, tabu_list, iteration, current_cost, best_cost, delta, i, j):
            allowed_pairs.append((i, j))
            allowed_deltas.append(float(delta))
    if not allowed_pairs:
        return np.empty((0, 2), dtype=np.int64), np.empty(0, dtype=np.float64), len(pool)
    pairs = np.asarray(allowed_pairs, dtype=np.int64)
    deltas = np.asarray(allowed_deltas, dtype=np.float64)
    k = len(pairs) if candidate_swaps is None else min(int(candidate_swaps), len(pairs))
    order = np.argsort(deltas)[:k]
    return pairs[order], deltas[order], len(pool)


def tabu_search(
    instance: QAPInstance,
    initial_perm: np.ndarray,
    method: str,
    n_iterations: int,
    seed: int,
    candidate_swaps: int | None,
    score_pool_swaps: int | None,
    selector: SelectorBundle | None = None,
    tabu_duration: int | None = None,
) -> TabuSearchResult:
    n = instance.n
    tabu_duration = int(8 * n if tabu_duration is None else tabu_duration)
    seed_rng = np.random.default_rng(seed)
    candidate_rng = np.random.default_rng(int(seed_rng.integers(0, 2**32 - 1)))
    tenure_rng = np.random.default_rng(int(seed_rng.integers(0, 2**32 - 1)))

    perm = np.asarray(initial_perm, dtype=np.int64).copy()
    current_cost = compute_cost(perm, instance.F, instance.D)
    best_cost = current_cost
    best_perm = perm.copy()
    tabu_list = np.zeros((n, n), dtype=np.int64)
    feature_cache: SwapFeatureCache | None = build_swap_feature_cache(instance) if method == "learned" else None

    n_delta = 0
    n_score = 0
    completed = 0
    for iteration in range(1, int(n_iterations) + 1):
        completed = iteration
        if method == "delta_oracle":
            pairs, selected_deltas, score_evals = _select_delta_oracle_allowed(
                instance=instance,
                perm=perm,
                tabu_list=tabu_list,
                iteration=iteration,
                current_cost=current_cost,
                best_cost=best_cost,
                rng=candidate_rng,
                candidate_swaps=candidate_swaps,
                score_pool_swaps=score_pool_swaps,
            )
            n_score += int(score_evals)
            if len(pairs) == 0:
                break
            n_delta += len(pairs)
            best_idx = int(np.argmin(selected_deltas))
            i = int(pairs[best_idx, 0])
            j = int(pairs[best_idx, 1])
            best_delta = float(selected_deltas[best_idx])
        else:
            pairs, score_evals = select_pairs(
                method=method,
                instance=instance,
                perm=perm,
                rng=candidate_rng,
                candidate_swaps=candidate_swaps,
                score_pool_swaps=score_pool_swaps,
                selector=selector if method == "learned" else None,
                feature_cache=feature_cache,
            )
            n_score += int(score_evals)

            best_move: tuple[int, int] | None = None
            best_delta = float("inf")
            for i_raw, j_raw in pairs:
                ii = int(i_raw)
                jj = int(j_raw)
                delta = swap_delta_cost(perm, instance.F, instance.D, ii, jj)
                n_delta += 1
                if _is_allowed(perm, tabu_list, iteration, current_cost, best_cost, delta, ii, jj) and delta < best_delta:
                    best_delta = float(delta)
                    best_move = (ii, jj)

            if best_move is None:
                break
            i, j = best_move

        old_i_location = int(perm[i])
        old_j_location = int(perm[j])
        perm[i], perm[j] = perm[j], perm[i]
        current_cost += best_delta

        tabu_list[i, old_i_location] = iteration + _tenure(tabu_duration, tenure_rng)
        tabu_list[j, old_j_location] = iteration + _tenure(tabu_duration, tenure_rng)

        if current_cost < best_cost:
            best_cost = current_cost
            best_perm = perm.copy()

    return TabuSearchResult(
        perm=best_perm,
        cost=float(best_cost),
        n_iters=completed,
        n_delta_evals=n_delta,
        n_score_evals=n_score,
        method=method,
    )


def random_multistart_tabu_search(
    instance: QAPInstance,
    method: str,
    n_starts: int,
    n_iterations: int,
    candidate_swaps: int | None,
    score_pool_swaps: int | None,
    seed: int,
    selector: SelectorBundle | None = None,
    tabu_duration: int | None = None,
) -> TabuSearchResult:
    rng = np.random.default_rng(seed)
    starts = [random_perm(instance.n, rng) for _ in range(int(n_starts))]
    search_seeds = [int(rng.integers(0, 2**32 - 1)) for _ in range(int(n_starts))]

    results = []
    for start, search_seed in zip(starts, search_seeds):
        results.append(
            tabu_search(
                instance=instance,
                initial_perm=start,
                method=method,
                n_iterations=n_iterations,
                seed=search_seed,
                candidate_swaps=candidate_swaps,
                score_pool_swaps=score_pool_swaps,
                selector=selector if method == "learned" else None,
                tabu_duration=tabu_duration,
            )
        )

    if not results:
        raise ValueError("n_starts must be positive.")
    best = min(results, key=lambda item: item.cost)
    return TabuSearchResult(
        perm=best.perm,
        cost=best.cost,
        n_iters=sum(item.n_iters for item in results),
        n_delta_evals=sum(item.n_delta_evals for item in results),
        n_score_evals=sum(item.n_score_evals for item in results),
        method=method,
    )
