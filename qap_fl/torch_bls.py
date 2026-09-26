from __future__ import annotations

from dataclasses import dataclass
import math
import time

import numpy as np
import torch

from .features import all_swap_pairs
from .local_search import SelectorBundle
from .official_bls import CStdRand, INFINITE
from .qap import QAPInstance
from .torch_qap import (
    TorchSwapFeatureCache,
    batched_apply_swaps,
    batched_cost,
    batched_initialize_delta,
    batched_swap_features,
)


@dataclass(frozen=True)
class BatchedBLSResult:
    perms: np.ndarray
    costs: np.ndarray
    initial_costs: np.ndarray
    runtime_sec: float
    n_outer_iters: np.ndarray
    n_moves: np.ndarray
    n_descent_decisions: np.ndarray
    n_delta_evals: np.ndarray
    n_score_evals: np.ndarray
    device: str


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _sample_pool(all_pairs: np.ndarray, count: int, rng: np.random.Generator) -> np.ndarray:
    if count >= len(all_pairs):
        return all_pairs.copy()
    return all_pairs[rng.choice(len(all_pairs), size=count, replace=False)]


def _score_pools(
    selector: SelectorBundle,
    cache: TorchSwapFeatureCache,
    perms: torch.Tensor,
    pools: torch.Tensor,
    feature_mean: torch.Tensor,
    feature_std: torch.Tensor,
) -> torch.Tensor:
    features = batched_swap_features(cache, perms, pools)
    standardized = (features - feature_mean) / feature_std
    with torch.no_grad():
        flat = standardized.reshape(-1, standardized.shape[-1])
        return selector.model(flat).reshape(standardized.shape[:2])


def batched_bls(
    instance: QAPInstance,
    initial_perms: np.ndarray,
    seeds: list[int] | np.ndarray,
    selector: SelectorBundle | None = None,
    *,
    method: str = "learned",
    device: torch.device,
    max_time_sec: float | None = None,
    max_outer_iterations: int = 2_000_000_000,
    max_descent_decisions: int | None = None,
    candidate_swaps: int = 32,
    score_pool_swaps: int = 64,
    learned_score_interval: int = 4,
    r1: float = 0.7,
    r2: float = 0.2,
    init_perturb_strength: float = 0.15,
    stagnation_threshold: int = 2500,
    p0: float = 0.75,
    q: float = 0.3,
    validate_interval: int = 0,
    feature_cache: TorchSwapFeatureCache | None = None,
) -> BatchedBLSResult:
    """Run independent BLS lanes with tensorized QAP kernels.

    Lanes share kernels but retain independent candidate and official perturbation
    random streams. A wall-clock budget applies to the complete batch.
    """
    initial_perms = np.asarray(initial_perms, dtype=np.int64)
    seeds = np.asarray(seeds, dtype=np.int64)
    if initial_perms.ndim != 2 or len(initial_perms) != len(seeds):
        raise ValueError("initial_perms and seeds must describe the same batch")
    batch, n = initial_perms.shape
    if batch == 0 or any(sorted(row.tolist()) != list(range(n)) for row in initial_perms):
        raise ValueError("every initial row must be a valid permutation")
    if method not in {"learned", "random", "full"}:
        raise ValueError(f"unsupported batched BLS method: {method}")
    if method == "learned" and selector is None:
        raise ValueError("learned batched BLS requires a selector")
    if method != "full":
        if int(candidate_swaps) <= 0 or int(score_pool_swaps) <= 0:
            raise ValueError("candidate counts must be positive")
        if int(candidate_swaps) > int(score_pool_swaps):
            raise ValueError("candidate_swaps cannot exceed score_pool_swaps")

    synchronize(device)
    started = time.perf_counter()
    deadline = None if max_time_sec is None else started + float(max_time_sec)
    flow = torch.as_tensor(instance.F, dtype=torch.int64, device=device)
    distance = torch.as_tensor(instance.D, dtype=torch.int64, device=device)
    perms = torch.as_tensor(initial_perms, dtype=torch.long, device=device)
    if selector is not None:
        selector.model.to(device)
        selector.model.eval()
    cache = feature_cache
    if method == "learned" and cache is None:
        cache = TorchSwapFeatureCache.from_instance(instance, device)
    feature_mean = (
        torch.as_tensor(selector.feature_mean, dtype=torch.float32, device=device)
        if method == "learned" else None
    )
    feature_std = (
        torch.as_tensor(selector.feature_std, dtype=torch.float32, device=device)
        if method == "learned" else None
    )
    delta = batched_initialize_delta(flow, distance, perms)
    current = batched_cost(flow, distance, perms)
    initial_costs = current.clone()
    best = current.clone()
    best_perms = perms.clone()
    last_swapped = torch.zeros((batch, n, n), dtype=torch.int64, device=device)

    candidate_rngs = [np.random.default_rng(int(seed)) for seed in seeds]
    official_rngs = [CStdRand(int(seed)) for seed in seeds]
    all_pairs_np = all_swap_pairs(n)
    all_pairs_tensor = torch.as_tensor(all_pairs_np, dtype=torch.long, device=device)
    tri_first, tri_second = all_pairs_tensor[:, 0], all_pairs_tensor[:, 1]
    if method == "full":
        pool_count = len(all_pairs_np)
        keep_count = len(all_pairs_np)
    else:
        pool_count = min(int(score_pool_swaps), len(all_pairs_np))
        keep_count = min(int(candidate_swaps), pool_count)

    iterations = torch.zeros(batch, dtype=torch.int64, device=device)
    without_improvement = torch.zeros(batch, dtype=torch.int64, device=device)
    perturb_strength = np.full(batch, math.ceil(init_perturb_strength * n), dtype=np.float64)
    learned_decisions = np.zeros(batch, dtype=np.int64)
    n_outer = np.zeros(batch, dtype=np.int64)
    n_moves = np.zeros(batch, dtype=np.int64)
    n_decisions = np.zeros(batch, dtype=np.int64)
    n_delta = np.zeros(batch, dtype=np.int64)
    n_score = np.zeros(batch, dtype=np.int64)

    def expired() -> bool:
        if deadline is None:
            return False
        # Search decisions already transfer a small mask back to the host. A
        # second explicit CUDA synchronization at every loop boundary roughly
        # doubled the number of global barriers in the original migration.
        return time.perf_counter() >= deadline

    def apply(indices: np.ndarray, first: torch.Tensor, second: torch.Tensor) -> None:
        nonlocal perms, delta, current, best, best_perms
        if len(indices) == 0:
            return
        idx = torch.as_tensor(indices, dtype=torch.long, device=device)
        new_perm, new_delta, move_delta = batched_apply_swaps(
            flow, distance, perms[idx], delta[idx], first, second
        )
        last_swapped[idx, first, second] = iterations[idx]
        last_swapped[idx, second, first] = iterations[idx]
        perms[idx] = new_perm
        delta[idx] = new_delta
        current[idx] += move_delta
        improved = current[idx] < best[idx]
        best[idx] = torch.where(improved, current[idx], best[idx])
        best_perms[idx] = torch.where(improved[:, None], perms[idx], best_perms[idx])
        without_improvement[idx] = torch.where(
            improved, torch.zeros_like(without_improvement[idx]), without_improvement[idx]
        )
        iterations[idx] += 1
        n_moves[indices] += 1

    outer = 0
    while outer < int(max_outer_iterations) and not expired():
        if max_descent_decisions is not None and np.all(n_decisions >= int(max_descent_decisions)):
            break
        outer += 1
        previous = current.clone()
        descent_moves = np.zeros(batch, dtype=np.int64)
        active = np.ones(batch, dtype=bool)
        if max_descent_decisions is not None:
            active &= n_decisions < int(max_descent_decisions)
        n_outer[active] += 1

        while np.any(active) and not expired():
            active_indices = np.flatnonzero(active)
            if method == "full":
                selected = all_pairs_tensor.unsqueeze(0).expand(len(active_indices), -1, -1)
            else:
                pools_np = np.stack([
                    _sample_pool(all_pairs_np, pool_count, candidate_rngs[lane])
                    for lane in active_indices
                ])
                pools = torch.as_tensor(pools_np, dtype=torch.long, device=device)
                # _sample_pool returns a random ordering. Its first k entries
                # are therefore already a uniform k-subset for random/fallback
                # decisions, avoiding one Python loop and many tiny GPU writes.
                selected = pools[:, :keep_count].clone()
                if method == "learned":
                    score_mask = learned_decisions[active_indices] % int(learned_score_interval) == 0
                    if np.any(score_mask):
                        score_rows = np.flatnonzero(score_mask)
                        score_rows_tensor = torch.as_tensor(
                            score_rows, dtype=torch.long, device=device
                        )
                        lane_tensor = torch.as_tensor(
                            active_indices[score_rows], dtype=torch.long, device=device
                        )
                        scores = _score_pools(
                            selector, cache, perms[lane_tensor], pools[score_rows_tensor],
                            feature_mean, feature_std,
                        )
                        top = torch.topk(
                            scores, k=keep_count, dim=1, largest=True, sorted=True
                        ).indices
                        selected[score_rows_tensor] = pools[score_rows_tensor].gather(
                            1, top.unsqueeze(2).expand(-1, -1, 2)
                        )
                        n_score[active_indices[score_rows]] += pool_count

            lane_tensor = torch.as_tensor(active_indices, dtype=torch.long, device=device)
            values = delta[lane_tensor[:, None], selected[:, :, 0], selected[:, :, 1]]
            minimum, position = values.min(dim=1)
            chosen = selected[torch.arange(len(active_indices), device=device), position]
            improving = (minimum < 0).detach().cpu().numpy()
            n_decisions[active_indices] += 1
            n_delta[active_indices] += keep_count
            if method == "learned":
                learned_decisions[active_indices] += 1
            moved_indices = active_indices[improving]
            if len(moved_indices):
                chosen_moved = chosen[torch.as_tensor(np.flatnonzero(improving), device=device)]
                apply(moved_indices, chosen_moved[:, 0], chosen_moved[:, 1])
                descent_moves[moved_indices] += 1
            active[active_indices[~improving]] = False
            if max_descent_decisions is not None:
                active &= n_decisions < int(max_descent_decisions)

            if validate_interval and int(n_moves.sum()) % int(validate_interval) < len(moved_indices):
                rebuilt = batched_initialize_delta(flow, distance, perms)
                if not torch.equal(delta, rebuilt) or not torch.equal(current, batched_cost(flow, distance, perms)):
                    raise AssertionError("batched BLS state validation failed")

        if expired() or (max_descent_decisions is not None and np.all(n_decisions >= int(max_descent_decisions))):
            break

        state_snapshot = torch.stack((previous, current, without_improvement)).detach().cpu().numpy()
        previous_np, current_np, without_improvement_np = state_snapshot
        for lane in range(batch):
            if without_improvement_np[lane] > int(stagnation_threshold):
                without_improvement_np[lane] = 0
                perturb_strength[lane] = n * (0.4 + official_rngs[lane].rand() % 20 / 100.0)
            elif descent_moves[lane] != 0 and previous_np[lane] != current_np[lane]:
                without_improvement_np[lane] += 1
                perturb_strength[lane] = max(math.ceil(init_perturb_strength * n), 5)
            elif previous_np[lane] == current_np[lane]:
                perturb_strength[lane] += 1
        without_improvement.copy_(
            torch.as_tensor(without_improvement_np, dtype=torch.int64, device=device)
        )

        perturb_left = np.ceil(perturb_strength).astype(np.int64)
        if max_descent_decisions is not None:
            perturb_left[n_decisions >= int(max_descent_decisions)] = 0
        init_cost = previous.clone()
        directed = np.asarray([
            max(math.exp(-without_improvement_np[lane] / float(stagnation_threshold)), p0)
            > official_rngs[lane].unit_101()
            for lane in range(batch)
        ])
        while np.any(perturb_left > 0) and not expired():
            active_indices = np.flatnonzero(perturb_left > 0)
            chosen_first = []
            chosen_second = []
            valid_indices = []
            directed_indices = active_indices[directed[active_indices]]
            age_indices = []
            random_indices = []
            for lane in active_indices[~directed[active_indices]]:
                if q > official_rngs[lane].unit_101():
                    age_indices.append(int(lane))
                else:
                    random_indices.append(int(lane))

            # Directed perturbations are the common branch. Evaluate every
            # active lane in one tensor operation and transfer only the chosen
            # position/value pair, instead of synchronizing once per lane.
            if len(directed_indices):
                directed_tensor = torch.as_tensor(
                    directed_indices, dtype=torch.long, device=device
                )
                lane_values = delta[directed_tensor[:, None], tri_first, tri_second]
                tenure_noise = max(int(n * r2), 1)
                noise = torch.as_tensor(
                    np.asarray([
                        [official_rngs[lane].rand() % tenure_noise for _ in range(len(all_pairs_np))]
                        for lane in directed_indices
                    ], dtype=np.int64),
                    dtype=torch.int64, device=device,
                )
                next_cost = current[directed_tensor, None] + lane_values
                allowed = (next_cost != init_cost[directed_tensor, None]) & (
                    (
                        last_swapped[directed_tensor[:, None], tri_first, tri_second]
                        + n * r1 + noise < iterations[directed_tensor, None]
                    )
                    | (next_cost < best[directed_tensor, None])
                )
                masked = torch.where(
                    allowed, lane_values, torch.full_like(lane_values, INFINITE)
                )
                directed_values, directed_positions = masked.min(dim=1)
                directed_choices = torch.stack(
                    (directed_positions, directed_values), dim=1
                ).detach().cpu().numpy()
                invalid_directed = []
                for lane, (position, value) in zip(directed_indices, directed_choices):
                    if int(value) >= INFINITE:
                        perturb_left[lane] = 0
                        invalid_directed.append(int(lane))
                        continue
                    i, j = all_pairs_np[int(position)]
                    valid_indices.append(int(lane))
                    chosen_first.append(int(i))
                    chosen_second.append(int(j))
                if invalid_directed:
                    invalid_tensor = torch.as_tensor(
                        invalid_directed, dtype=torch.long, device=device
                    )
                    iterations[invalid_tensor] += 1

            if age_indices:
                age_tensor = torch.as_tensor(age_indices, dtype=torch.long, device=device)
                age_positions = last_swapped[
                    age_tensor[:, None], tri_first, tri_second
                ].argmin(dim=1).detach().cpu().numpy()
                for lane, position in zip(age_indices, age_positions):
                    i, j = all_pairs_np[int(position)]
                    valid_indices.append(lane)
                    chosen_first.append(int(i))
                    chosen_second.append(int(j))

            # The random branch is uncommon and may need rejection checks
            # against the current delta, so retain its exact per-lane stream.
            for lane in random_indices:
                i = official_rngs[lane].uniform_int(0, n - 1)
                j = official_rngs[lane].uniform_int(0, n - 1)
                if i > j:
                    i, j = j, i
                attempts = 0
                while (
                    i == j or int((current[lane] + delta[lane, i, j]).item()) == int(init_cost[lane].item())
                ) and attempts < max(n * n * 4, 16):
                    j = official_rngs[lane].uniform_int(0, n - 1)
                    if i > j:
                        i, j = j, i
                    attempts += 1
                if i == j:
                    perturb_left[lane] = 0
                    iterations[lane] += 1
                    continue
                valid_indices.append(lane)
                chosen_first.append(i)
                chosen_second.append(j)
            if valid_indices:
                apply(
                    np.asarray(valid_indices, dtype=np.int64),
                    torch.as_tensor(chosen_first, dtype=torch.long, device=device),
                    torch.as_tensor(chosen_second, dtype=torch.long, device=device),
                )
            perturb_left[active_indices] -= 1

    synchronize(device)
    runtime = time.perf_counter() - started
    return BatchedBLSResult(
        perms=best_perms.detach().cpu().numpy(),
        costs=best.detach().cpu().numpy(),
        initial_costs=initial_costs.detach().cpu().numpy(),
        runtime_sec=float(runtime), n_outer_iters=n_outer, n_moves=n_moves,
        n_descent_decisions=n_decisions, n_delta_evals=n_delta, n_score_evals=n_score,
        device=str(device),
    )


def batched_learned_bls(
    instance: QAPInstance,
    initial_perms: np.ndarray,
    seeds: list[int] | np.ndarray,
    selector: SelectorBundle,
    **kwargs,
) -> BatchedBLSResult:
    """Backward-compatible learned-selector entry point used by D214."""
    return batched_bls(
        instance, initial_perms, seeds, selector, method="learned", **kwargs
    )


__all__ = ["BatchedBLSResult", "batched_bls", "batched_learned_bls", "synchronize"]
