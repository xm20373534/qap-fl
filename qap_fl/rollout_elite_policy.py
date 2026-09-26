from __future__ import annotations

import numpy as np

from .features import SwapFeatureCache, build_swap_feature_cache, sample_swap_pairs
from .local_search import LocalSearchResult, SelectorBundle
from .qap import QAPInstance, compute_cost, random_perm, swap_delta_cost, swap_perm
from .rollout_policy_v2 import _score_pairs, _top_k_by_score


def _choose_elite_rollout_move(
    scores: np.ndarray,
    deltas: np.ndarray,
    elite_m: int,
) -> int | None:
    improving = np.flatnonzero(deltas < 0.0)
    if len(improving) == 0:
        return None
    elite_count = min(int(elite_m), len(improving))
    delta_order = improving[np.argsort(deltas[improving], kind="stable")[:elite_count]]
    return int(delta_order[np.argmax(scores[delta_order])])


def two_opt_rollout_elite_policy(
    perm: np.ndarray,
    instance: QAPInstance,
    max_iters: int,
    candidate_swaps: int | None,
    score_pool_swaps: int | None,
    elite_m: int,
    rng: np.random.Generator,
    selector: SelectorBundle,
    feature_cache: SwapFeatureCache | None = None,
) -> LocalSearchResult:
    current = np.asarray(perm, dtype=np.int64).copy()
    current_cost = compute_cost(current, instance.F, instance.D)
    start_cost = current_cost
    cache = build_swap_feature_cache(instance) if feature_cache is None else feature_cache
    n_delta = 0
    n_score = 0
    completed = 0

    for iteration in range(int(max_iters)):
        completed = iteration + 1
        pool = sample_swap_pairs(instance.n, score_pool_swaps, rng)
        scores = _score_pairs(selector, instance, current, pool, cache)
        k = len(pool) if candidate_swaps is None else int(candidate_swaps)
        selected_pairs, selected_scores = _top_k_by_score(pool, scores, k)
        deltas = np.asarray(
            [
                swap_delta_cost(current, instance.F, instance.D, int(i), int(j))
                for i, j in selected_pairs
            ],
            dtype=np.float64,
        )
        n_score += len(pool)
        n_delta += len(selected_pairs)
        chosen_index = _choose_elite_rollout_move(selected_scores, deltas, elite_m)
        if chosen_index is None:
            break
        best_pair = selected_pairs[chosen_index]
        current = swap_perm(current, int(best_pair[0]), int(best_pair[1]))
        current_cost += float(deltas[chosen_index])

    return LocalSearchResult(
        perm=current,
        cost=float(current_cost),
        n_iters=completed,
        n_delta_evals=n_delta,
        n_score_evals=n_score,
        improved=bool(current_cost < start_cost),
        method=f"learned_rollout_elite_m{int(elite_m)}",
    )


def random_multistart_rollout_elite_policy(
    instance: QAPInstance,
    n_starts: int,
    max_iters: int,
    candidate_swaps: int | None,
    score_pool_swaps: int | None,
    elite_m: int,
    seed: int,
    selector: SelectorBundle,
) -> LocalSearchResult:
    rng = np.random.default_rng(seed)
    feature_cache = build_swap_feature_cache(instance)
    results = []
    for _ in range(int(n_starts)):
        results.append(
            two_opt_rollout_elite_policy(
                perm=random_perm(instance.n, rng),
                instance=instance,
                max_iters=max_iters,
                candidate_swaps=candidate_swaps,
                score_pool_swaps=score_pool_swaps,
                elite_m=elite_m,
                rng=rng,
                selector=selector,
                feature_cache=feature_cache,
            )
        )
    best = min(results, key=lambda item: item.cost)
    return LocalSearchResult(
        perm=best.perm,
        cost=best.cost,
        n_iters=sum(item.n_iters for item in results),
        n_delta_evals=sum(item.n_delta_evals for item in results),
        n_score_evals=sum(item.n_score_evals for item in results),
        improved=any(item.improved for item in results),
        method=f"learned_rollout_elite_m{int(elite_m)}",
    )
