from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .features import SwapFeatureCache, all_swap_pairs, build_swap_features, sample_swap_pairs
from .model import SwapMLP
from .numpy_inference import NumpyMLP
from .qap import QAPInstance, compute_cost, random_perm, swap_delta_cost, swap_perm


@dataclass(frozen=True)
class LocalSearchResult:
    perm: np.ndarray
    cost: float
    n_iters: int
    n_delta_evals: int
    n_score_evals: int
    improved: bool
    method: str


@dataclass(frozen=True)
class SelectorBundle:
    model: SwapMLP
    feature_fields: list[str]
    feature_mean: np.ndarray
    feature_std: np.ndarray
    numpy_model: NumpyMLP | None = None


def _top_k_largest(pairs: np.ndarray, scores: np.ndarray, k: int) -> np.ndarray:
    k = min(int(k), len(pairs))
    if k >= len(pairs):
        return pairs[np.argsort(-scores)]
    idx = np.argpartition(-scores, kth=k - 1)[:k]
    idx = idx[np.argsort(-scores[idx])]
    return pairs[idx]


def _score_pairs(
    bundle: SelectorBundle,
    instance: QAPInstance,
    perm: np.ndarray,
    pairs: np.ndarray,
    feature_cache: SwapFeatureCache | None,
) -> np.ndarray:
    features = build_swap_features(instance, perm, pairs, feature_cache=feature_cache)
    if bundle.numpy_model is not None:
        return bundle.numpy_model(features)
    features = (features - bundle.feature_mean) / bundle.feature_std
    device = next(bundle.model.parameters()).device
    x = torch.from_numpy(features).to(device=device, dtype=torch.float32)
    bundle.model.eval()
    with torch.no_grad():
        return bundle.model(x).detach().cpu().numpy().astype(np.float64)


def select_pairs(
    method: str,
    instance: QAPInstance,
    perm: np.ndarray,
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
            raise ValueError("learned local search requires selector bundle.")
        if feature_cache is None:
            raise ValueError("learned local search requires a feature cache.")
        scores = _score_pairs(selector, instance, perm, pool, feature_cache)
        return _top_k_largest(pool, scores, k), len(pool)
    if method == "delta_oracle":
        deltas = np.asarray(
            [swap_delta_cost(perm, instance.F, instance.D, int(i), int(j)) for i, j in pool],
            dtype=np.float64,
        )
        return pool[np.argsort(deltas)[:k]], len(pool)
    raise ValueError(f"unknown method: {method}")


def two_opt_local_search(
    perm: np.ndarray,
    instance: QAPInstance,
    method: str,
    max_iters: int,
    candidate_swaps: int | None,
    score_pool_swaps: int | None,
    rng: np.random.Generator,
    selector: SelectorBundle | None = None,
    feature_cache: SwapFeatureCache | None = None,
) -> LocalSearchResult:
    current = np.asarray(perm, dtype=np.int64).copy()
    current_cost = compute_cost(current, instance.F, instance.D)
    start_cost = current_cost
    cache = feature_cache
    if method == "learned" and cache is None:
        from .features import build_swap_feature_cache

        cache = build_swap_feature_cache(instance)
    n_delta = 0
    n_score = 0
    completed = 0
    for iteration in range(int(max_iters)):
        completed = iteration + 1
        pairs, score_evals = select_pairs(
            method=method,
            instance=instance,
            perm=current,
            rng=rng,
            candidate_swaps=candidate_swaps,
            score_pool_swaps=score_pool_swaps,
            selector=selector,
            feature_cache=cache,
        )
        n_score += int(score_evals)
        best_delta = 0.0
        best_pair: tuple[int, int] | None = None
        for i_raw, j_raw in pairs:
            i = int(i_raw)
            j = int(j_raw)
            delta = swap_delta_cost(current, instance.F, instance.D, i, j)
            n_delta += 1
            if delta < best_delta:
                best_delta = float(delta)
                best_pair = (i, j)
        if best_pair is None:
            break
        current = swap_perm(current, best_pair[0], best_pair[1])
        current_cost += best_delta
    return LocalSearchResult(
        perm=current,
        cost=float(current_cost),
        n_iters=completed,
        n_delta_evals=n_delta,
        n_score_evals=n_score,
        improved=bool(current_cost < start_cost),
        method=method,
    )


def random_multistart_two_opt(
    instance: QAPInstance,
    method: str,
    n_starts: int,
    max_iters: int,
    candidate_swaps: int | None,
    score_pool_swaps: int | None,
    seed: int,
    selector: SelectorBundle | None = None,
) -> LocalSearchResult:
    rng = np.random.default_rng(seed)
    feature_cache = None
    if method == "learned":
        from .features import build_swap_feature_cache

        feature_cache = build_swap_feature_cache(instance)
    results = []
    for _ in range(int(n_starts)):
        start = random_perm(instance.n, rng)
        results.append(
            two_opt_local_search(
                perm=start,
                instance=instance,
                method=method,
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
        method=method,
    )


