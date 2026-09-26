from __future__ import annotations

import numpy as np
import torch

from .features import SwapFeatureCache, build_swap_feature_cache, sample_swap_pairs
from .global_features import build_swap_features_for_fields
from .local_search import LocalSearchResult, SelectorBundle
from .qap import QAPInstance, compute_cost, random_perm, swap_delta_cost, swap_perm


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


def _normalized_delta_rank(deltas: np.ndarray) -> np.ndarray:
    order = np.argsort(deltas, kind="stable")
    ranks = np.empty(len(deltas), dtype=np.float64)
    ranks[order] = np.arange(len(deltas), dtype=np.float64)
    return ranks / float(max(len(deltas) - 1, 1))


def _top_k_indices(values: np.ndarray, k: int) -> np.ndarray:
    k = max(1, min(int(k), len(values)))
    if k >= len(values):
        return np.argsort(-values, kind="stable")
    idx = np.argpartition(-values, kth=k - 1)[:k]
    return idx[np.argsort(-values[idx], kind="stable")]


def two_opt_hybrid_policy(
    perm: np.ndarray,
    instance: QAPInstance,
    max_iters: int,
    candidate_swaps: int | None,
    score_pool_swaps: int | None,
    delta_weight: float,
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
        deltas = np.asarray(
            [
                swap_delta_cost(current, instance.F, instance.D, int(i), int(j))
                for i, j in pool
            ],
            dtype=np.float64,
        )
        n_score += len(pool)
        n_delta += len(pool)

        delta_rank = _normalized_delta_rank(deltas)
        combined = scores - float(delta_weight) * delta_rank
        k = len(pool) if candidate_swaps is None else int(candidate_swaps)
        selected = _top_k_indices(combined, k)
        improving = selected[deltas[selected] < -1e-12]
        if len(improving) == 0:
            break
        chosen = int(improving[np.argmax(combined[improving])])
        i = int(pool[chosen, 0])
        j = int(pool[chosen, 1])
        current = swap_perm(current, i, j)
        current_cost += float(deltas[chosen])

    return LocalSearchResult(
        perm=current,
        cost=float(current_cost),
        n_iters=completed,
        n_delta_evals=n_delta,
        n_score_evals=n_score,
        improved=bool(current_cost < start_cost),
        method=f"learned_hybrid_w{float(delta_weight):g}",
    )


def random_multistart_hybrid_policy(
    instance: QAPInstance,
    n_starts: int,
    max_iters: int,
    candidate_swaps: int | None,
    score_pool_swaps: int | None,
    delta_weight: float,
    seed: int,
    selector: SelectorBundle,
) -> LocalSearchResult:
    rng = np.random.default_rng(seed)
    feature_cache = build_swap_feature_cache(instance)
    results = []
    for _ in range(int(n_starts)):
        results.append(
            two_opt_hybrid_policy(
                perm=random_perm(instance.n, rng),
                instance=instance,
                max_iters=max_iters,
                candidate_swaps=candidate_swaps,
                score_pool_swaps=score_pool_swaps,
                delta_weight=float(delta_weight),
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
        method=f"learned_hybrid_w{float(delta_weight):g}",
    )
