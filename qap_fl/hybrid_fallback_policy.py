from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .features import SwapFeatureCache, build_swap_feature_cache, sample_swap_pairs
from .global_features import build_swap_features_for_fields
from .hybrid_gate import build_hybrid_gate_features
from .local_search import LocalSearchResult, SelectorBundle
from .qap import QAPInstance, compute_cost, random_perm, swap_delta_cost, swap_perm


@dataclass(frozen=True)
class FallbackStats:
    n_model_moves: int
    n_fallback_moves: int
    n_gate_rejects: int = 0


@dataclass(frozen=True)
class FallbackLocalSearchResult(LocalSearchResult):
    n_model_moves: int
    n_fallback_moves: int
    n_gate_rejects: int = 0


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


def _gate_unsafe_probability(bundle: SelectorBundle, features: np.ndarray) -> float:
    features = (np.asarray(features, dtype=np.float32) - bundle.feature_mean) / bundle.feature_std
    device = next(bundle.model.parameters()).device
    x = torch.from_numpy(features).to(device=device, dtype=torch.float32).unsqueeze(0)
    bundle.model.eval()
    with torch.no_grad():
        value = torch.sigmoid(bundle.model(x)).detach().cpu().item()
    return float(value)


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


def _softmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    values = values - float(np.max(values))
    exp = np.exp(values)
    total = float(np.sum(exp))
    if not np.isfinite(total) or total <= 0:
        return np.full(len(values), 1.0 / max(len(values), 1), dtype=np.float64)
    return exp / total


def _method_name(
    delta_weight: float,
    score_margin: float,
    max_delta_rank: float,
    min_model_prob: float,
    max_model_moves: int | None,
    gate: SelectorBundle | None,
    gate_threshold: float,
) -> str:
    cap_suffix = "all" if max_model_moves is None else str(int(max_model_moves))
    prob_suffix = f"p{float(min_model_prob):g}"
    gate_suffix = "" if gate is None else f"_g{float(gate_threshold):g}"
    return (
        f"learned_fallback_w{float(delta_weight):g}_"
        f"m{float(score_margin):g}_r{float(max_delta_rank):g}_"
        f"{prob_suffix}{gate_suffix}_cap{cap_suffix}"
    )


def two_opt_hybrid_fallback_policy(
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
    gate: SelectorBundle | None = None,
    gate_threshold: float = 0.5,
) -> FallbackLocalSearchResult:
    current = np.asarray(perm, dtype=np.int64).copy()
    current_cost = compute_cost(current, instance.F, instance.D)
    start_cost = current_cost
    cache = build_swap_feature_cache(instance) if feature_cache is None else feature_cache
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
        can_use_model = max_model_moves is None or n_model_moves < int(max_model_moves)
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
                    gate_features = build_hybrid_gate_features(
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
                        model_prob=float(model_prob),
                        model_delta_rank=float(delta_rank[model_idx]),
                        selected_deltas=deltas[selected_improving],
                        selected_scores=scores[selected_improving],
                        selected_combined=combined[selected_improving],
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
        method=_method_name(delta_weight, score_margin, max_delta_rank, min_model_prob, max_model_moves, gate, gate_threshold),
        n_model_moves=n_model_moves,
        n_fallback_moves=n_fallback_moves,
        n_gate_rejects=n_gate_rejects,
    )


def random_multistart_hybrid_fallback_policy(
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
    results = []
    for _ in range(int(n_starts)):
        results.append(
            two_opt_hybrid_fallback_policy(
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
                gate=gate,
                gate_threshold=float(gate_threshold),
            )
        )
    best = min(results, key=lambda item: item.cost)
    return FallbackLocalSearchResult(
        perm=best.perm,
        cost=best.cost,
        n_iters=sum(item.n_iters for item in results),
        n_delta_evals=sum(item.n_delta_evals for item in results),
        n_score_evals=sum(item.n_score_evals for item in results),
        improved=any(item.improved for item in results),
        method=_method_name(delta_weight, score_margin, max_delta_rank, min_model_prob, max_model_moves, gate, gate_threshold),
        n_model_moves=sum(item.n_model_moves for item in results),
        n_fallback_moves=sum(item.n_fallback_moves for item in results),
        n_gate_rejects=sum(item.n_gate_rejects for item in results),
    )
