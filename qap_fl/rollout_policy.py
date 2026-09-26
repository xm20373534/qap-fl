from __future__ import annotations

import numpy as np
import torch

from .features import SwapFeatureCache, all_swap_pairs, build_swap_feature_cache, sample_swap_pairs
from .global_features import build_swap_features_for_fields
from .local_search import LocalSearchResult, SelectorBundle
from .qap import QAPInstance, compute_cost, random_perm, swap_delta_cost, swap_perm


def _top_k_by_score(pairs: np.ndarray, scores: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    k = min(int(k), len(pairs))
    if k >= len(pairs):
        order = np.argsort(-scores, kind="stable")
    else:
        order = np.argpartition(-scores, kth=k - 1)[:k]
        order = order[np.argsort(-scores[order], kind="stable")]
    return pairs[order], scores[order]


def _score_pairs(
    bundle: SelectorBundle,
    instance: QAPInstance,
    perm: np.ndarray,
    pairs: np.ndarray,
    feature_cache: SwapFeatureCache,
) -> np.ndarray:
    features = build_swap_features_for_fields(
        instance=instance,
        perm=perm,
        pairs=pairs,
        feature_fields=bundle.feature_fields,
        feature_cache=feature_cache,
    )
    features = (features - bundle.feature_mean) / bundle.feature_std
    device = next(bundle.model.parameters()).device
    x = torch.from_numpy(features).to(device=device, dtype=torch.float32)
    bundle.model.eval()
    with torch.no_grad():
        return bundle.model(x).detach().cpu().numpy().astype(np.float64)


def _select_rollout_move(
    pairs: np.ndarray,
    scores: np.ndarray,
    deltas: np.ndarray,
) -> tuple[int, int] | None:
    improving = deltas < 0.0
    if np.any(improving):
        candidate_indices = np.flatnonzero(improving)
        best_index = int(candidate_indices[np.argmax(scores[candidate_indices])])
        return int(pairs[best_index, 0]), int(pairs[best_index, 1])

    best_index = int(np.argmin(deltas))
    if float(deltas[best_index]) < 0.0:
        return int(pairs[best_index, 0]), int(pairs[best_index, 1])
    return None


def two_opt_rollout_policy(
    perm: np.ndarray,
    instance: QAPInstance,
    max_iters: int,
    candidate_swaps: int | None,
    score_pool_swaps: int | None,
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
        selected_pairs, selected_scores = _top_k_by_score(pool, scores, candidate_swaps or len(pool))
        deltas = np.asarray(
            [
                swap_delta_cost(current, instance.F, instance.D, int(i), int(j))
                for i, j in selected_pairs
            ],
            dtype=np.float64,
        )
        n_score += len(pool)
        n_delta += len(selected_pairs)
        best_pair = _select_rollout_move(selected_pairs, selected_scores, deltas)
        if best_pair is None:
            break
        best_delta = swap_delta_cost(
            current,
            instance.F,
            instance.D,
            best_pair[0],
            best_pair[1],
        )
        current = swap_perm(current, best_pair[0], best_pair[1])
        current_cost += float(best_delta)

    return LocalSearchResult(
        perm=current,
        cost=float(current_cost),
        n_iters=completed,
        n_delta_evals=n_delta,
        n_score_evals=n_score,
        improved=bool(current_cost < start_cost),
        method="learned_rollout_policy",
    )


def random_multistart_rollout_policy(
    instance: QAPInstance,
    n_starts: int,
    max_iters: int,
    candidate_swaps: int | None,
    score_pool_swaps: int | None,
    seed: int,
    selector: SelectorBundle,
) -> LocalSearchResult:
    rng = np.random.default_rng(seed)
    feature_cache = build_swap_feature_cache(instance)
    results = []
    for _ in range(int(n_starts)):
        results.append(
            two_opt_rollout_policy(
                perm=random_perm(instance.n, rng),
                instance=instance,
                max_iters=max_iters,
                candidate_swaps=candidate_swaps,
                score_pool_swaps=score_pool_swaps,
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
        method="learned_rollout_policy",
    )
