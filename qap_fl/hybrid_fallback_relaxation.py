from __future__ import annotations

from dataclasses import replace

import numpy as np

from .features import SwapFeatureCache, build_swap_feature_cache, sample_swap_pairs
from .hybrid_fallback_policy import (
    FallbackLocalSearchResult,
    _gate_unsafe_probability,
    _method_name,
    _score_pairs,
    _softmax,
    _top_k_indices,
    _normalized_delta_rank,
)
from .hybrid_gate_relaxation import build_hybrid_gate_relaxation_features
from .local_search import SelectorBundle
from .qap import QAPInstance, compute_cost, random_perm, swap_delta_cost, swap_perm
from .relaxation_features_v2 import build_relaxation_feature_cache


def two_opt_hybrid_fallback_relaxation(
    perm: np.ndarray,
    instance: QAPInstance,
    max_iters: int,
    candidate_swaps: int | None,
    score_pool_swaps: int | None,
    delta_weight: float,
    score_margin: float,
    max_delta_rank: float,
    min_model_prob: float,
    max_model_moves: int | None,
    rng: np.random.Generator,
    selector: SelectorBundle,
    feature_cache: SwapFeatureCache | None = None,
    relaxation_cache=None,
    gate: SelectorBundle | None = None,
    gate_threshold: float = 0.5,
) -> FallbackLocalSearchResult:
    current = np.asarray(perm, dtype=np.int64).copy()
    current_cost = compute_cost(current, instance.F, instance.D)
    start_cost = current_cost
    cache = build_swap_feature_cache(instance) if feature_cache is None else feature_cache
    relax_cache = (
        build_relaxation_feature_cache(instance)
        if relaxation_cache is None
        else relaxation_cache
    )
    n_delta = 0
    n_score = 0
    completed = 0
    n_model_moves = 0
    n_fallback_moves = 0
    n_gate_rejects = 0
    total_pairs = int(instance.n * (instance.n - 1) / 2)

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
        improving = np.flatnonzero(deltas < -1e-12)
        if len(improving) == 0:
            break

        best_delta_idx = int(improving[np.argmin(deltas[improving])])
        delta_rank = _normalized_delta_rank(deltas)
        combined = scores - float(delta_weight) * delta_rank
        k = len(pool) if candidate_swaps is None else int(candidate_swaps)
        selected = _top_k_indices(combined, k)
        selected_improving = selected[deltas[selected] < -1e-12]

        chosen = best_delta_idx
        use_model = False
        can_use_model = (
            max_model_moves is None or n_model_moves < int(max_model_moves)
        )
        if can_use_model and len(selected_improving) > 0:
            selected_combined = combined[selected_improving]
            model_pos = int(np.argmax(selected_combined))
            model_idx = int(selected_improving[model_pos])
            model_prob = float(_softmax(selected_combined)[model_pos])
            margin = float(combined[model_idx] - combined[best_delta_idx])
            if (
                model_idx != best_delta_idx
                and model_prob >= float(min_model_prob)
                and margin >= float(score_margin)
                and float(delta_rank[model_idx]) <= float(max_delta_rank)
            ):
                accept_model = True
                if gate is not None:
                    best_pair = (
                        int(pool[best_delta_idx, 0]),
                        int(pool[best_delta_idx, 1]),
                    )
                    model_pair = (
                        int(pool[model_idx, 0]),
                        int(pool[model_idx, 1]),
                    )
                    gate_features = build_hybrid_gate_relaxation_features(
                        n=instance.n,
                        iteration=iteration,
                        max_iters=max_iters,
                        pool_size=len(pool),
                        total_pairs=total_pairs,
                        n_improving=len(improving),
                        selected_count=len(selected_improving),
                        best_delta=float(deltas[best_delta_idx]),
                        model_delta=float(deltas[model_idx]),
                        best_score=float(scores[best_delta_idx]),
                        model_score=float(scores[model_idx]),
                        best_combined=float(combined[best_delta_idx]),
                        model_combined=float(combined[model_idx]),
                        model_prob=model_prob,
                        model_delta_rank=float(delta_rank[model_idx]),
                        selected_deltas=deltas[selected_improving],
                        selected_scores=scores[selected_improving],
                        selected_combined=combined[selected_improving],
                        instance=instance,
                        perm=current,
                        best_pair=best_pair,
                        model_pair=model_pair,
                        relaxation_cache=relax_cache,
                    )
                    unsafe_prob = _gate_unsafe_probability(gate, gate_features)
                    accept_model = unsafe_prob <= float(gate_threshold)
                    if not accept_model:
                        n_gate_rejects += 1
                if accept_model:
                    chosen = model_idx
                    use_model = True

        if use_model:
            n_model_moves += 1
        else:
            n_fallback_moves += 1
        i = int(pool[chosen, 0])
        j = int(pool[chosen, 1])
        current = swap_perm(current, i, j)
        current_cost += float(deltas[chosen])

    return FallbackLocalSearchResult(
        perm=current,
        cost=float(current_cost),
        n_iters=completed,
        n_delta_evals=n_delta,
        n_score_evals=n_score,
        improved=bool(current_cost < start_cost),
        method=_method_name(
            delta_weight,
            score_margin,
            max_delta_rank,
            min_model_prob,
            max_model_moves,
            gate,
            gate_threshold,
        ).replace("_g", "_relaxg", 1) if gate is not None else _method_name(
            delta_weight,
            score_margin,
            max_delta_rank,
            min_model_prob,
            max_model_moves,
            gate,
            gate_threshold,
        ),
        n_model_moves=n_model_moves,
        n_fallback_moves=n_fallback_moves,
        n_gate_rejects=n_gate_rejects,
    )


def random_multistart_hybrid_fallback_relaxation(
    instance: QAPInstance,
    n_starts: int,
    max_iters: int,
    candidate_swaps: int | None,
    score_pool_swaps: int | None,
    delta_weight: float,
    score_margin: float,
    max_delta_rank: float,
    min_model_prob: float,
    max_model_moves: int | None,
    seed: int,
    selector: SelectorBundle,
    gate: SelectorBundle | None = None,
    gate_threshold: float = 0.5,
) -> FallbackLocalSearchResult:
    rng = np.random.default_rng(seed)
    feature_cache = build_swap_feature_cache(instance)
    relaxation_cache = build_relaxation_feature_cache(instance)
    results = []
    for _ in range(int(n_starts)):
        results.append(
            two_opt_hybrid_fallback_relaxation(
                perm=random_perm(instance.n, rng),
                instance=instance,
                max_iters=max_iters,
                candidate_swaps=candidate_swaps,
                score_pool_swaps=score_pool_swaps,
                delta_weight=float(delta_weight),
                score_margin=float(score_margin),
                max_delta_rank=float(max_delta_rank),
                min_model_prob=float(min_model_prob),
                max_model_moves=max_model_moves,
                rng=rng,
                selector=selector,
                feature_cache=feature_cache,
                relaxation_cache=relaxation_cache,
                gate=gate,
                gate_threshold=float(gate_threshold),
            )
        )
    best = min(results, key=lambda item: item.cost)
    return replace(
        best,
        n_iters=sum(item.n_iters for item in results),
        n_delta_evals=sum(item.n_delta_evals for item in results),
        n_score_evals=sum(item.n_score_evals for item in results),
        improved=any(item.improved for item in results),
        n_model_moves=sum(item.n_model_moves for item in results),
        n_fallback_moves=sum(item.n_fallback_moves for item in results),
        n_gate_rejects=sum(item.n_gate_rejects for item in results),
    )
