from __future__ import annotations

from dataclasses import dataclass, field
import math
import time

import numpy as np
import torch

from .features import SwapFeatureCache, all_swap_pairs, build_swap_feature_cache, sample_swap_pairs
from .global_features import build_swap_features_for_fields
from .local_search import SelectorBundle
from .qap import QAPInstance, compute_cost, random_perm
from .rots import compute_delta_taillard, initialize_delta_matrix, update_delta_matrix_after_swap
from .vdss import find_vdss_improvement


LEARNED_BLS_METHODS = {
    "learned",
    "learned_random",
    "learned_adaptive_k",
    "random_learned_replace",
    "random_learned_add",
    "random_learned_tiebreak",
    "official_coupled_learned",
    "learned_conflict",
}


@dataclass(frozen=True)
class BLSTiming:
    wall_total_sec: float = 0.0
    search_loop_sec: float = 0.0
    delta_init_sec: float = 0.0
    delta_update_sec: float = 0.0
    feature_cache_sec: float = 0.0
    feature_build_sec: float = 0.0
    model_forward_sec: float = 0.0
    candidate_pool_sec: float = 0.0
    candidate_select_sec: float = 0.0
    delta_scan_sec: float = 0.0
    perturb_sec: float = 0.0
    vdss_sec: float = 0.0


@dataclass(frozen=True)
class BLSResult:
    perm: np.ndarray
    cost: float
    n_outer_iters: int
    n_moves: int
    n_descent_moves: int
    n_perturb_moves: int
    n_delta_evals: int
    n_score_evals: int
    method: str
    hit_optimum: bool
    timing: BLSTiming | None = None
    n_vdss_calls: int = 0
    n_vdss_accepted: int = 0
    n_vdss_attempts: int = 0


@dataclass
class _BLSState:
    perm: np.ndarray
    delta: np.ndarray
    last_swapped: np.ndarray
    current_cost: float
    best_cost: float
    best_perm: np.ndarray
    iteration: int
    iter_without_improvement: int
    perturb_strength: float
    n_moves: int = 0
    n_descent_moves: int = 0
    n_perturb_moves: int = 0
    n_delta_evals: int = 0
    n_score_evals: int = 0
    profile_timing: bool = False
    timings: dict[str, float] = field(default_factory=dict)
    lazy_delta: bool = False
    delta_dirty: bool = False
    delta_refresh_estimate_sec: float = 0.0
    n_vdss_calls: int = 0
    n_vdss_accepted: int = 0
    n_vdss_attempts: int = 0


def _add_timing(state: _BLSState, key: str, start: float) -> None:
    if state.profile_timing:
        state.timings[key] = state.timings.get(key, 0.0) + (time.perf_counter() - start)


def _timing_from_dict(values: dict[str, float]) -> BLSTiming:
    return BLSTiming(**{name: float(values.get(name, 0.0)) for name in BLSTiming.__dataclass_fields__})


def _sum_timings(values: list[BLSTiming | None]) -> BLSTiming | None:
    active = [value for value in values if value is not None]
    if not active:
        return None
    return BLSTiming(
        **{
            name: sum(float(getattr(value, name)) for value in active)
            for name in BLSTiming.__dataclass_fields__
        }
    )


def _refresh_delta_matrix(state: _BLSState, instance: QAPInstance) -> None:
    start = time.perf_counter()
    state.delta[:, :] = initialize_delta_matrix(state.perm, instance.F, instance.D)
    state.delta_dirty = False
    elapsed = time.perf_counter() - start
    state.delta_refresh_estimate_sec = max(state.delta_refresh_estimate_sec, elapsed)
    _add_timing(state, "delta_update_sec", start)


def _top_k_largest(pairs: np.ndarray, scores: np.ndarray, k: int) -> np.ndarray:
    k = min(int(k), len(pairs))
    if k >= len(pairs):
        return pairs[np.argsort(-scores)]
    idx = np.argpartition(-scores, kth=k - 1)[:k]
    idx = idx[np.argsort(-scores[idx])]
    return pairs[idx]


def _conflict_pool(
    instance: QAPInstance,
    perm: np.ndarray,
    rng: np.random.Generator,
    pool_size: int | None,
    conflict_fraction: float,
    conflict_mode: str = "pair",
) -> np.ndarray:
    """Build a fixed-size pool mixing high-conflict and random swaps."""
    pairs = all_swap_pairs(instance.n)
    if pool_size is None or int(pool_size) >= len(pairs):
        return pairs[rng.permutation(len(pairs))]
    size = int(pool_size)
    if size <= 0:
        raise ValueError("pool_size must be positive or None.")
    fraction = float(conflict_fraction)
    if not (0.0 <= fraction <= 1.0):
        raise ValueError("conflict_fraction must be in [0, 1].")
    n_conflict = min(size, max(0, int(round(size * fraction))))
    i = pairs[:, 0]
    j = pairs[:, 1]
    if conflict_mode == "pair":
        conflict = (
            np.abs(instance.F[i, j] * instance.D[perm[i], perm[j]])
            + np.abs(instance.F[j, i] * instance.D[perm[j], perm[i]])
        )
    elif conflict_mode == "incident":
        assigned = np.asarray(perm, dtype=np.int64)
        current_interaction = np.abs(instance.F * instance.D[np.ix_(assigned, assigned)])
        node_activity = (
            current_interaction.sum(axis=0)
            + current_interaction.sum(axis=1)
            - np.diag(current_interaction)
        )
        direct = current_interaction[i, j] + current_interaction[j, i]
        conflict = node_activity[i] + node_activity[j] - direct
    else:
        raise ValueError("conflict_mode must be pair or incident.")
    conflict_order = np.argsort(-conflict, kind="stable")
    selected = conflict_order[:n_conflict]
    mask = np.ones(len(pairs), dtype=bool)
    mask[selected] = False
    remaining = np.flatnonzero(mask)
    n_random = size - n_conflict
    if n_random > 0:
        random_idx = (
            remaining[rng.permutation(len(remaining))]
            if n_random >= len(remaining)
            else rng.choice(remaining, size=n_random, replace=False)
        )
        selected = np.concatenate([selected, random_idx])
    return pairs[selected[rng.permutation(len(selected))]]


def _score_pairs(
    selector: SelectorBundle,
    instance: QAPInstance,
    perm: np.ndarray,
    pairs: np.ndarray,
    feature_cache: SwapFeatureCache,
    state: _BLSState | None = None,
) -> np.ndarray:
    start = time.perf_counter() if state is not None and state.profile_timing else 0.0
    features = build_swap_features_for_fields(
        instance=instance,
        perm=perm,
        pairs=pairs,
        feature_fields=selector.feature_fields,
        feature_cache=feature_cache,
    )
    features = (features - selector.feature_mean) / selector.feature_std
    if state is not None:
        _add_timing(state, "feature_build_sec", start)
    device = next(selector.model.parameters()).device
    start = time.perf_counter() if state is not None and state.profile_timing else 0.0
    x = torch.from_numpy(features).to(device=device, dtype=torch.float32)
    selector.model.eval()
    with torch.no_grad():
        scores = selector.model(x).detach().cpu().numpy().astype(np.float64)
    if state is not None:
        _add_timing(state, "model_forward_sec", start)
    return scores


def _select_descent_pairs(
    method: str,
    instance: QAPInstance,
    state: _BLSState,
    rng: np.random.Generator,
    candidate_swaps: int | None,
    score_pool_swaps: int | None,
    selector: SelectorBundle | None,
    feature_cache: SwapFeatureCache | None,
    learned_random_fraction: float = 0.5,
    adaptive_candidate_swaps: int | None = None,
    adaptive_score_margin: float = 0.25,
    conflict_fraction: float = 0.5,
    conflict_mode: str = "pair",
) -> tuple[np.ndarray, int]:
    if method == "full":
        start = time.perf_counter() if state.profile_timing else 0.0
        pairs = all_swap_pairs(instance.n)
        _add_timing(state, "candidate_pool_sec", start)
        return pairs, 0

    if method == "official_coupled_learned":
        if selector is None:
            raise ValueError("official_coupled_learned requires selector bundle.")
        if feature_cache is None:
            raise ValueError("official_coupled_learned requires a feature cache.")
        start = time.perf_counter() if state.profile_timing else 0.0
        pool = all_swap_pairs(instance.n)
        _add_timing(state, "candidate_pool_sec", start)
        k = len(pool) if candidate_swaps is None else min(int(candidate_swaps), len(pool))
        if k <= 0:
            raise ValueError("candidate_swaps must be positive or None.")
        scores = _score_pairs(selector, instance, state.perm, pool, feature_cache, state=state)
        return _top_k_largest(pool, scores, k), len(pool)

    start = time.perf_counter() if state.profile_timing else 0.0
    pool = (
        _conflict_pool(instance, state.perm, rng, score_pool_swaps, conflict_fraction, conflict_mode)
        if method == "learned_conflict"
        else sample_swap_pairs(instance.n, score_pool_swaps, rng)
    )
    _add_timing(state, "candidate_pool_sec", start)
    k = len(pool) if candidate_swaps is None else min(int(candidate_swaps), len(pool))
    if k <= 0:
        raise ValueError("candidate_swaps must be positive or None.")

    if method == "random":
        if k >= len(pool):
            return pool[rng.permutation(len(pool))], 0
        return pool[rng.choice(len(pool), size=k, replace=False)], 0

    if method in LEARNED_BLS_METHODS:
        if selector is None:
            raise ValueError("learned BLS local search requires selector bundle.")
        if feature_cache is None:
            raise ValueError("learned BLS local search requires a feature cache.")
        if method == "random_learned_tiebreak":
            if k >= len(pool):
                random_idx = rng.permutation(len(pool))
            else:
                random_idx = rng.choice(len(pool), size=k, replace=False)
            selected_pairs = pool[random_idx]
            tie_scores = _score_pairs(selector, instance, state.perm, selected_pairs, feature_cache, state=state)
            return selected_pairs[np.argsort(-tie_scores, kind="stable")], len(selected_pairs)

        scores = _score_pairs(selector, instance, state.perm, pool, feature_cache, state=state)
        if method in {"random_learned_replace", "random_learned_add"}:
            if k >= len(pool):
                random_idx = rng.permutation(len(pool))
            else:
                random_idx = rng.choice(len(pool), size=k, replace=False)

            order = np.argsort(-scores, kind="stable")
            score_std = float(np.std(scores))
            if len(order) < 2 or score_std <= 1e-12:
                confidence = 0.0
            else:
                confidence = float(scores[order[0]] - scores[order[1]]) / score_std
            if confidence < float(adaptive_score_margin):
                return pool[random_idx], len(pool)

            learned_fraction = float(learned_random_fraction)
            if not (0.0 < learned_fraction <= 1.0):
                raise ValueError("learned_random_fraction must be in (0, 1].")
            n_learned = int(round(k * learned_fraction))
            n_learned = max(1, min(k, n_learned))
            learned_idx = order[: min(n_learned, len(order))]
            if method == "random_learned_replace":
                selected = np.zeros(len(pool), dtype=bool)
                selected[learned_idx] = True
                remaining_idx = np.flatnonzero(~selected)
                n_random = k - len(learned_idx)
                if n_random <= 0 or len(remaining_idx) == 0:
                    return pool[learned_idx], len(pool)
                if n_random >= len(remaining_idx):
                    fill_idx = remaining_idx[rng.permutation(len(remaining_idx))]
                else:
                    fill_idx = rng.choice(remaining_idx, size=n_random, replace=False)
                return pool[np.concatenate([learned_idx, fill_idx])], len(pool)

            selected = np.zeros(len(pool), dtype=bool)
            selected[random_idx] = True
            learned_extra = learned_idx[~selected[learned_idx]]
            if len(learned_extra) == 0:
                return pool[random_idx], len(pool)
            return pool[np.concatenate([random_idx, learned_extra])], len(pool)

        if method == "learned_random":
            learned_fraction = float(learned_random_fraction)
            if not (0.0 < learned_fraction <= 1.0):
                raise ValueError("learned_random_fraction must be in (0, 1].")
            n_learned = int(round(k * learned_fraction))
            n_learned = max(1, min(k, n_learned))
            if n_learned >= len(pool):
                learned_idx = np.argsort(-scores)
            else:
                learned_idx = np.argpartition(-scores, kth=n_learned - 1)[:n_learned]
                learned_idx = learned_idx[np.argsort(-scores[learned_idx])]
            selected = np.zeros(len(pool), dtype=bool)
            selected[learned_idx] = True
            remaining_idx = np.flatnonzero(~selected)
            n_random = k - len(learned_idx)
            if n_random <= 0 or len(remaining_idx) == 0:
                return pool[learned_idx], len(pool)
            if n_random >= len(remaining_idx):
                fill_idx = remaining_idx[rng.permutation(len(remaining_idx))]
            else:
                fill_idx = rng.choice(remaining_idx, size=n_random, replace=False)
            return pool[np.concatenate([learned_idx, fill_idx])], len(pool)
        if method == "learned_adaptive_k":
            max_k = len(pool) if adaptive_candidate_swaps is None else min(int(adaptive_candidate_swaps), len(pool))
            max_k = max(k, max_k)
            if k >= len(pool) or max_k <= k:
                return _top_k_largest(pool, scores, k), len(pool)
            order = np.argsort(-scores, kind="stable")
            score_std = float(np.std(scores))
            if score_std <= 1e-12:
                use_k = max_k
            else:
                boundary_margin = float(scores[order[k - 1]] - scores[order[max_k - 1]]) / score_std
                use_k = max_k if boundary_margin < float(adaptive_score_margin) else k
            return pool[order[:use_k]], len(pool)
        return _top_k_largest(pool, scores, k), len(pool)

    if method == "delta_oracle":
        values = state.delta[pool[:, 0], pool[:, 1]]
        order = np.argsort(values, kind="stable")[:k]
        return pool[order], len(pool)

    raise ValueError(f"unknown BLS local search method: {method}")


def _apply_move(
    state: _BLSState,
    instance: QAPInstance,
    i: int | None,
    j: int | None,
    phase: str,
) -> None:
    if i is not None and j is not None:
        value = float(state.delta[i, j])
        old_i = int(i)
        old_j = int(j)
        state.last_swapped[old_i, old_j] = state.iteration
        state.last_swapped[old_j, old_i] = state.iteration
        state.perm[old_i], state.perm[old_j] = state.perm[old_j], state.perm[old_i]
        state.current_cost += value
        if state.lazy_delta and phase == "descent":
            state.delta_dirty = True
        else:
            start = time.perf_counter() if state.profile_timing else 0.0
            update_delta_matrix_after_swap(state.delta, state.perm, instance.F, instance.D, old_i, old_j)
            _add_timing(state, "delta_update_sec", start)
        state.n_moves += 1
        if phase == "descent":
            state.n_descent_moves += 1
        elif phase == "perturb":
            state.n_perturb_moves += 1
        if state.current_cost < state.best_cost:
            state.best_cost = float(state.current_cost)
            state.best_perm = state.perm.copy()
            state.iter_without_improvement = 0
    state.iteration += 1


def _best_improvement_move(
    state: _BLSState,
    instance: QAPInstance,
    method: str,
    rng: np.random.Generator,
    candidate_swaps: int | None,
    score_pool_swaps: int | None,
    selector: SelectorBundle | None,
    feature_cache: SwapFeatureCache | None,
    learned_random_fraction: float = 0.5,
    adaptive_candidate_swaps: int | None = None,
    adaptive_score_margin: float = 0.25,
    conflict_fraction: float = 0.5,
    conflict_mode: str = "pair",
) -> bool:
    start = time.perf_counter() if state.profile_timing else 0.0
    pairs, score_evals = _select_descent_pairs(
        method=method,
        instance=instance,
        state=state,
        rng=rng,
        candidate_swaps=candidate_swaps,
        score_pool_swaps=score_pool_swaps,
        selector=selector,
        feature_cache=feature_cache,
        learned_random_fraction=learned_random_fraction,
        adaptive_candidate_swaps=adaptive_candidate_swaps,
        adaptive_score_margin=adaptive_score_margin,
        conflict_fraction=conflict_fraction,
        conflict_mode=conflict_mode,
    )
    _add_timing(state, "candidate_select_sec", start)
    state.n_score_evals += int(score_evals)
    if len(pairs) == 0:
        return False
    if state.lazy_delta:
        start = time.perf_counter() if state.profile_timing else 0.0
        values = np.asarray(
            [compute_delta_taillard(state.perm, instance.F, instance.D, int(i), int(j)) for i, j in pairs],
            dtype=np.float64,
        )
    else:
        start = time.perf_counter() if state.profile_timing else 0.0
        values = state.delta[pairs[:, 0], pairs[:, 1]]
    state.n_delta_evals += int(len(pairs))
    best_idx = int(np.argmin(values))
    best_delta = float(values[best_idx])
    _add_timing(state, "delta_scan_sec", start)
    if best_delta < -1e-12:
        i = int(pairs[best_idx, 0])
        j = int(pairs[best_idx, 1])
        state.delta[i, j] = best_delta
        state.delta[j, i] = best_delta
        _apply_move(state, instance, i, j, phase="descent")
        return True
    return False


def _tabu_search_perturb(
    state: _BLSState,
    instance: QAPInstance,
    rng: np.random.Generator,
    init_cost: float,
    r1: float,
    r2: float,
) -> None:
    n = instance.n
    best_pair: tuple[int, int] | None = None
    min_delta = float("inf")
    jitter_bound = max(int(n * float(r2)), 1)
    for i in range(n - 1):
        for j in range(i + 1, n):
            value = float(state.delta[i, j])
            state.n_delta_evals += 1
            next_cost = state.current_cost + value
            if next_cost == init_cost:
                continue
            tenure_clear = state.last_swapped[i, j] + n * float(r1) + int(rng.integers(0, jitter_bound)) < state.iteration
            aspired = next_cost < state.best_cost
            if value < min_delta and (tenure_clear or aspired):
                best_pair = (i, j)
                min_delta = value
    if best_pair is None:
        _apply_move(state, instance, None, None, phase="perturb")
    else:
        _apply_move(state, instance, best_pair[0], best_pair[1], phase="perturb")


def _recency_based_perturb(state: _BLSState, instance: QAPInstance) -> None:
    n = instance.n
    best_pair = (0, 1)
    best_age = state.last_swapped[0, 1]
    for i in range(n - 1):
        for j in range(i + 1, n):
            state.n_delta_evals += 1
            if state.last_swapped[i, j] < best_age:
                best_pair = (i, j)
                best_age = state.last_swapped[i, j]
    _apply_move(state, instance, best_pair[0], best_pair[1], phase="perturb")


def _random_perturb(
    state: _BLSState,
    instance: QAPInstance,
    rng: np.random.Generator,
    init_cost: float,
) -> None:
    n = instance.n
    max_trials = max(n * n * 2, 1)
    for _ in range(max_trials):
        i, j = rng.choice(n, 2, replace=False)
        if i > j:
            i, j = j, i
        i = int(i)
        j = int(j)
        state.n_delta_evals += 1
        if state.current_cost + float(state.delta[i, j]) != init_cost:
            _apply_move(state, instance, i, j, phase="perturb")
            return
    _apply_move(state, instance, None, None, phase="perturb")


def _apply_perturbation_move(
    state: _BLSState,
    instance: QAPInstance,
    rng: np.random.Generator,
    init_cost: float,
    kind: str,
    r1: float,
    r2: float,
) -> None:
    if kind == "directed":
        _tabu_search_perturb(state, instance, rng, init_cost, r1, r2)
    elif kind == "recency":
        _recency_based_perturb(state, instance)
    elif kind == "random":
        _random_perturb(state, instance, rng, init_cost)
    else:
        raise ValueError(f"Unknown perturbation kind: {kind}.")


def _perturb_with_action(
    state: _BLSState,
    instance: QAPInstance,
    rng: np.random.Generator,
    init_cost: float,
    kind: str,
    n_moves: int,
    r1: float,
    r2: float,
    deadline: float | None = None,
) -> None:
    """Apply one fixed perturbation macro-action for counterfactual evaluation."""
    for _ in range(max(int(n_moves), 1)):
        if deadline is not None and time.perf_counter() >= deadline:
            break
        _apply_perturbation_move(state, instance, rng, init_cost, kind, r1, r2)


def _perturb(
    state: _BLSState,
    instance: QAPInstance,
    rng: np.random.Generator,
    init_cost: float,
    r1: float,
    r2: float,
    p0: float,
    q: float,
    stagnation_scale: int,
    deadline: float | None = None,
) -> None:
    d = float(state.iter_without_improvement) / float(max(stagnation_scale, 1))
    directed_prob = math.exp(-d)
    if directed_prob < float(p0):
        directed_prob = float(p0)
    use_directed = directed_prob > float(rng.integers(0, 101)) / 100.0
    n_moves = max(int(math.ceil(state.perturb_strength)), 1)
    for _ in range(n_moves):
        if deadline is not None and time.perf_counter() >= deadline:
            break
        if use_directed:
            kind = "directed"
        elif float(q) > float(rng.integers(0, 101)) / 100.0:
            kind = "recency"
        else:
            kind = "random"
        _apply_perturbation_move(state, instance, rng, init_cost, kind, r1, r2)


def _determine_jump_magnitude(
    state: _BLSState,
    descent_num: int,
    previous_cost: float,
    n: int,
    rng: np.random.Generator,
    init_perturb_strength: float,
    stagnation_scale: int,
) -> None:
    if state.iter_without_improvement > int(stagnation_scale):
        state.iter_without_improvement = 0
        state.perturb_strength = n * (0.4 + float(rng.integers(0, 20)) / 100.0)
    elif descent_num != 0 and previous_cost != state.current_cost:
        state.iter_without_improvement += 1
        state.perturb_strength = math.ceil(n * float(init_perturb_strength))
        if state.perturb_strength < 5:
            state.perturb_strength = 5
    elif previous_cost == state.current_cost:
        state.perturb_strength += 1


def _resolve_stop_target(instance: QAPInstance, target_cost: float | None) -> float | None:
    if target_cost is not None:
        return float(target_cost)
    if instance.optimum is not None:
        return float(instance.optimum)
    return None


def _target_reached(state: _BLSState, stop_target: float | None) -> bool:
    return bool(stop_target is not None and state.best_cost <= float(stop_target))


def breakout_local_search(
    instance: QAPInstance,
    initial_perm: np.ndarray,
    method: str,
    max_outer_iterations: int,
    seed: int,
    candidate_swaps: int | None = None,
    score_pool_swaps: int | None = None,
    selector: SelectorBundle | None = None,
    max_time_sec: float | None = None,
    stop_at_optimum: bool = True,
    target_cost: float | None = None,
    r1: float = 0.7,
    r2: float = 0.2,
    init_perturb_strength: float = 0.15,
    stagnation_scale: int = 2500,
    p0: float = 0.75,
    q: float = 0.3,
    learned_random_fraction: float = 0.5,
    adaptive_candidate_swaps: int | None = None,
    adaptive_score_margin: float = 0.25,
    conflict_fraction: float = 0.5,
    conflict_mode: str = "pair",
    profile_timing: bool = False,
    lazy_delta: bool = False,
    forced_perturb_kind: str | None = None,
    forced_perturb_strength_fraction: float | None = None,
    vdss_depths: tuple[int, ...] | None = None,
    vdss_frequency: int = 5,
    vdss_max_attempts_per_start: int = 256,
) -> BLSResult:
    solver_start = time.perf_counter()
    wall_start = solver_start if profile_timing else 0.0
    deadline = None if max_time_sec is None else solver_start + float(max_time_sec)
    if (forced_perturb_kind is None) != (forced_perturb_strength_fraction is None):
        raise ValueError("forced perturbation kind and strength must be provided together.")
    if forced_perturb_kind is not None and forced_perturb_kind not in {"directed", "recency", "random"}:
        raise ValueError(f"Unknown forced perturbation kind: {forced_perturb_kind}.")
    if forced_perturb_strength_fraction is not None and float(forced_perturb_strength_fraction) <= 0.0:
        raise ValueError("forced perturbation strength must be positive.")
    if vdss_depths is not None:
        if not vdss_depths or any(int(depth) < 2 for depth in vdss_depths):
            raise ValueError("vdss_depths must contain integers >= 2.")
        if any(int(right) <= int(left) for left, right in zip(vdss_depths, vdss_depths[1:])):
            raise ValueError("vdss_depths must be strictly increasing.")
        if int(vdss_frequency) <= 0 or int(vdss_max_attempts_per_start) <= 0:
            raise ValueError("VDSS frequency and attempt limit must be positive.")
    rng = np.random.default_rng(seed)
    perm = np.asarray(initial_perm, dtype=np.int64).copy()
    current_cost = compute_cost(perm, instance.F, instance.D)
    delta_start = time.perf_counter() if profile_timing else 0.0
    if lazy_delta and method not in {"full", "delta_oracle"}:
        delta = np.full((instance.n, instance.n), np.inf, dtype=np.float64)
        np.fill_diagonal(delta, np.inf)
    else:
        delta = initialize_delta_matrix(perm, instance.F, instance.D)
    timings = {}
    if profile_timing:
        timings["delta_init_sec"] = time.perf_counter() - delta_start
    state = _BLSState(
        perm=perm,
        delta=delta,
        last_swapped=np.zeros((instance.n, instance.n), dtype=np.int64),
        current_cost=float(current_cost),
        best_cost=float(current_cost),
        best_perm=perm.copy(),
        iteration=0,
        iter_without_improvement=0,
        perturb_strength=math.ceil(float(init_perturb_strength) * instance.n),
        profile_timing=bool(profile_timing),
        timings=timings,
        lazy_delta=bool(lazy_delta and method not in {"full", "delta_oracle"}),
        delta_dirty=bool(lazy_delta and method not in {"full", "delta_oracle"}),
    )
    cache_start = time.perf_counter() if profile_timing else 0.0
    feature_cache = build_swap_feature_cache(instance) if method in LEARNED_BLS_METHODS else None
    _add_timing(state, "feature_cache_sec", cache_start)
    previous_cost = float(state.current_cost)
    start_time = time.perf_counter()
    completed_outer = 0
    stop_target = _resolve_stop_target(instance, target_cost) if stop_at_optimum else None

    for outer in range(1, int(max_outer_iterations) + 1):
        if deadline is not None and time.perf_counter() >= deadline:
            break
        completed_outer = outer
        descent_num = 0
        while True:
            if deadline is not None and time.perf_counter() >= deadline:
                break
            moved = _best_improvement_move(
                state=state,
                instance=instance,
                method=method,
                rng=rng,
                candidate_swaps=candidate_swaps,
                score_pool_swaps=score_pool_swaps,
                selector=selector if method in LEARNED_BLS_METHODS else None,
                feature_cache=feature_cache,
                learned_random_fraction=learned_random_fraction,
                adaptive_candidate_swaps=adaptive_candidate_swaps,
                adaptive_score_margin=adaptive_score_margin,
                conflict_fraction=conflict_fraction,
                conflict_mode=conflict_mode,
            )
            if not moved:
                break
            descent_num += 1
            if _target_reached(state, stop_target):
                break
            if deadline is not None and time.perf_counter() >= deadline:
                break

        _determine_jump_magnitude(
            state=state,
            descent_num=descent_num,
            previous_cost=previous_cost,
            n=instance.n,
            rng=rng,
            init_perturb_strength=init_perturb_strength,
            stagnation_scale=stagnation_scale,
        )
        previous_cost = float(state.current_cost)

        if _target_reached(state, stop_target):
            break
        if deadline is not None and time.perf_counter() >= deadline:
            break

        if state.lazy_delta and state.delta_dirty:
            if (
                deadline is not None
                and state.delta_refresh_estimate_sec > 0.0
                and time.perf_counter() + 1.2 * state.delta_refresh_estimate_sec >= deadline
            ):
                break
            _refresh_delta_matrix(state, instance)
        if deadline is not None and time.perf_counter() >= deadline:
            break
        perturb_start = time.perf_counter() if profile_timing else 0.0
        vdss_accepted = False
        if vdss_depths is not None and outer % int(vdss_frequency) == 0:
            vdss_start = time.perf_counter() if profile_timing else 0.0
            state.n_vdss_calls += 1
            vdss_result = find_vdss_improvement(
                instance=instance,
                perm=state.perm,
                rng=rng,
                depths=tuple(int(depth) for depth in vdss_depths),
                max_attempts_per_start=int(vdss_max_attempts_per_start),
                deadline=deadline,
            )
            state.n_vdss_attempts += int(vdss_result.attempts)
            if vdss_result.accepted and vdss_result.cost < state.current_cost - 1e-12:
                state.perm = vdss_result.perm.copy()
                state.current_cost = float(vdss_result.cost)
                state.n_vdss_accepted += 1
                state.n_moves += int(vdss_result.depth)
                state.iteration += 1
                if state.current_cost < state.best_cost:
                    state.best_cost = state.current_cost
                    state.best_perm = state.perm.copy()
                    state.iter_without_improvement = 0
                else:
                    state.iter_without_improvement += 1
                state.delta[:, :] = initialize_delta_matrix(state.perm, instance.F, instance.D)
                state.delta_dirty = False
                previous_cost = float(state.current_cost)
                vdss_accepted = True
            _add_timing(state, "vdss_sec", vdss_start)

        if vdss_accepted:
            continue
        if forced_perturb_kind is None:
            _perturb(
                state=state,
                instance=instance,
                rng=rng,
                init_cost=previous_cost,
                r1=r1,
                r2=r2,
                p0=p0,
                q=q,
                stagnation_scale=stagnation_scale,
                deadline=deadline,
            )
        else:
            _perturb_with_action(
                state=state,
                instance=instance,
                rng=rng,
                init_cost=previous_cost,
                kind=forced_perturb_kind,
                n_moves=max(int(math.ceil(float(forced_perturb_strength_fraction) * instance.n)), 1),
                r1=r1,
                r2=r2,
                deadline=deadline,
            )
        _add_timing(state, "perturb_sec", perturb_start)

        if _target_reached(state, stop_target):
            break
        if deadline is not None and time.perf_counter() >= deadline:
            break

    hit_optimum = _target_reached(state, _resolve_stop_target(instance, target_cost))
    if profile_timing:
        state.timings["search_loop_sec"] = time.perf_counter() - start_time
        state.timings["wall_total_sec"] = time.perf_counter() - wall_start
    return BLSResult(
        perm=state.best_perm.copy(),
        cost=float(state.best_cost),
        n_outer_iters=completed_outer,
        n_moves=state.n_moves,
        n_descent_moves=state.n_descent_moves,
        n_perturb_moves=state.n_perturb_moves,
        n_delta_evals=state.n_delta_evals,
        n_score_evals=state.n_score_evals,
        method=method,
        hit_optimum=hit_optimum,
        timing=_timing_from_dict(state.timings) if profile_timing else None,
        n_vdss_calls=state.n_vdss_calls,
        n_vdss_accepted=state.n_vdss_accepted,
        n_vdss_attempts=state.n_vdss_attempts,
    )


def random_multistart_bls(
    instance: QAPInstance,
    method: str,
    n_starts: int,
    max_outer_iterations: int,
    candidate_swaps: int | None,
    score_pool_swaps: int | None,
    seed: int,
    selector: SelectorBundle | None = None,
    max_time_sec: float | None = None,
    stop_at_optimum: bool = True,
    target_cost: float | None = None,
    learned_random_fraction: float = 0.5,
    adaptive_candidate_swaps: int | None = None,
    adaptive_score_margin: float = 0.25,
    conflict_fraction: float = 0.5,
    conflict_mode: str = "pair",
    profile_timing: bool = False,
    lazy_delta: bool = False,
    forced_perturb_kind: str | None = None,
    forced_perturb_strength_fraction: float | None = None,
    vdss_depths: tuple[int, ...] | None = None,
    vdss_frequency: int = 5,
    vdss_max_attempts_per_start: int = 256,
) -> BLSResult:
    rng = np.random.default_rng(seed)
    starts = [random_perm(instance.n, rng) for _ in range(int(n_starts))]
    search_seeds = [int(rng.integers(0, 2**32 - 1)) for _ in range(int(n_starts))]
    per_start_time = None if max_time_sec is None else float(max_time_sec) / max(int(n_starts), 1)
    results: list[BLSResult] = []
    for start, search_seed in zip(starts, search_seeds):
        result = breakout_local_search(
            instance=instance,
            initial_perm=start,
            method=method,
            max_outer_iterations=max_outer_iterations,
            seed=search_seed,
            candidate_swaps=candidate_swaps,
            score_pool_swaps=score_pool_swaps,
            selector=selector if method in LEARNED_BLS_METHODS else None,
            max_time_sec=per_start_time,
            stop_at_optimum=stop_at_optimum,
            target_cost=target_cost,
            learned_random_fraction=learned_random_fraction,
            adaptive_candidate_swaps=adaptive_candidate_swaps,
            adaptive_score_margin=adaptive_score_margin,
            conflict_fraction=conflict_fraction,
            conflict_mode=conflict_mode,
            profile_timing=profile_timing,
            lazy_delta=lazy_delta,
            forced_perturb_kind=forced_perturb_kind,
            forced_perturb_strength_fraction=forced_perturb_strength_fraction,
            vdss_depths=vdss_depths,
            vdss_frequency=vdss_frequency,
            vdss_max_attempts_per_start=vdss_max_attempts_per_start,
        )
        results.append(result)
        if stop_at_optimum and result.hit_optimum:
            break
    best = min(results, key=lambda item: item.cost)
    return BLSResult(
        perm=best.perm,
        cost=best.cost,
        n_outer_iters=sum(item.n_outer_iters for item in results),
        n_moves=sum(item.n_moves for item in results),
        n_descent_moves=sum(item.n_descent_moves for item in results),
        n_perturb_moves=sum(item.n_perturb_moves for item in results),
        n_delta_evals=sum(item.n_delta_evals for item in results),
        n_score_evals=sum(item.n_score_evals for item in results),
        method=method,
        hit_optimum=any(item.hit_optimum for item in results),
        timing=_sum_timings([item.timing for item in results]),
        n_vdss_calls=sum(item.n_vdss_calls for item in results),
        n_vdss_accepted=sum(item.n_vdss_accepted for item in results),
        n_vdss_attempts=sum(item.n_vdss_attempts for item in results),
    )


