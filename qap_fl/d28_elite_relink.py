from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np

from .macro_actions import _Workspace, _descent, _workspace
from .official_bls import (
    OfficialBLSStateTrace,
    _apply_move,
    _compute_deltas_for_pairs,
    _integer_matrices,
)
from .qap import QAPInstance


@dataclass(frozen=True)
class D28EliteEntry:
    cost: int
    perm: np.ndarray


@dataclass(frozen=True)
class D28RelinkResult:
    workspace: _Workspace
    path_moves: int
    internal_moves: int
    descent_moves: int
    delta_evals: int


def _hamming(first: np.ndarray, second: np.ndarray) -> int:
    return int(np.count_nonzero(np.asarray(first) != np.asarray(second)))


def update_diverse_elite_pool(
    pool: list[D28EliteEntry],
    perm: np.ndarray,
    cost: int,
    *,
    max_size: int,
    min_distance: int,
) -> bool:
    """Insert a solution while preserving quality and assignment diversity."""
    candidate = D28EliteEntry(int(cost), np.asarray(perm, dtype=np.int64).copy())
    close = [entry for entry in pool if _hamming(entry.perm, candidate.perm) < int(min_distance)]
    if close:
        if candidate.cost >= min(entry.cost for entry in close):
            return False
        close_ids = {id(entry) for entry in close}
        pool[:] = [entry for entry in pool if id(entry) not in close_ids]
    pool.append(candidate)
    if len(pool) <= int(max_size):
        pool.sort(key=lambda entry: entry.cost)
        return True

    remaining = sorted(pool, key=lambda entry: entry.cost)
    selected = [remaining.pop(0)]
    while remaining and len(selected) < int(max_size):
        chosen = max(
            remaining,
            key=lambda entry: (
                min(_hamming(entry.perm, kept.perm) for kept in selected),
                -entry.cost,
            ),
        )
        selected.append(chosen)
        remaining[:] = [entry for entry in remaining if id(entry) != id(chosen)]
    pool[:] = sorted(selected, key=lambda entry: entry.cost)
    return any(np.array_equal(candidate.perm, entry.perm) for entry in pool)


def choose_diverse_elite_guide(
    pool: list[D28EliteEntry],
    perm: np.ndarray,
    *,
    min_distance: int,
) -> np.ndarray | None:
    candidates = [entry for entry in pool if not np.array_equal(entry.perm, perm)]
    if not candidates:
        return None
    diverse = [entry for entry in candidates if _hamming(entry.perm, perm) >= int(min_distance)]
    distance_eligible = diverse if diverse else candidates
    costs = np.asarray([entry.cost for entry in distance_eligible], dtype=np.float64)
    quality_limit = float(np.quantile(costs, 0.5))
    eligible = [entry for entry in distance_eligible if entry.cost <= quality_limit]
    chosen = max(eligible, key=lambda entry: (_hamming(entry.perm, perm), -entry.cost))
    return chosen.perm.copy()


def _copy_workspace(workspace: _Workspace) -> _Workspace:
    return _Workspace(
        perm=workspace.perm.copy(),
        delta=workspace.delta.copy(),
        current_cost=int(workspace.current_cost),
        best_cost=int(workspace.best_cost),
        best_perm=workspace.best_perm.copy(),
        last_swapped=workspace.last_swapped.copy(),
        iteration=int(workspace.iteration),
        iter_without_improvement=int(workspace.iter_without_improvement),
    )


def _restricted_pairs(n: int, active: np.ndarray) -> np.ndarray:
    pairs = {
        tuple(sorted((int(facility), other)))
        for facility in np.asarray(active, dtype=np.int64)
        for other in range(int(n))
        if int(facility) != other
    }
    return np.asarray(sorted(pairs), dtype=np.int64)


def _active_facilities(
    instance: QAPInstance,
    perm: np.ndarray,
    guide: np.ndarray,
    path_pair: tuple[int, int],
    hotspot_count: int = 2,
) -> np.ndarray:
    flow = np.asarray(instance.F, dtype=np.float64)
    distance = np.asarray(instance.D, dtype=np.float64)
    assigned = distance[np.ix_(perm, perm)]
    interaction = np.abs(flow * assigned)
    activity = interaction.sum(axis=0) + interaction.sum(axis=1) - np.diag(interaction)
    mismatched = np.flatnonzero(perm != guide)
    hotspots = mismatched[np.argsort(-activity[mismatched], kind="stable")[: int(hotspot_count)]]
    return np.unique(np.concatenate((np.asarray(path_pair, dtype=np.int64), hotspots)))


def _restricted_local_candidate(
    instance: QAPInstance,
    workspace: _Workspace,
    active: np.ndarray,
    *,
    max_moves: int,
    deadline: float | None,
) -> tuple[_Workspace, int, int]:
    flow, distance = _integer_matrices(instance)
    candidate = _copy_workspace(workspace)
    pairs = _restricted_pairs(instance.n, active)
    moves = 0
    delta_evals = 0
    for _ in range(int(max_moves)):
        if deadline is not None and time.perf_counter() >= deadline:
            break
        values = candidate.delta[pairs[:, 0], pairs[:, 1]]
        delta_evals += len(pairs)
        selected = int(np.argmin(values))
        if int(values[selected]) >= 0:
            break
        i, j = int(pairs[selected, 0]), int(pairs[selected, 1])
        (
            candidate.current_cost,
            candidate.best_cost,
            candidate.iter_without_improvement,
            candidate.iteration,
            moved,
        ) = _apply_move(
            candidate.perm,
            candidate.delta,
            candidate.current_cost,
            candidate.best_cost,
            candidate.best_perm,
            candidate.last_swapped,
            flow,
            distance,
            candidate.iter_without_improvement,
            candidate.iteration,
            i,
            j,
            "incremental",
        )
        moves += int(moved)
    return candidate, moves, delta_evals


def _relink_branch_with_internal_search(
    instance: QAPInstance,
    trace: OfficialBLSStateTrace,
    start: np.ndarray,
    guide: np.ndarray,
    *,
    local_stride: int,
    local_moves: int,
    deadline: float | None,
) -> tuple[_Workspace, int, int, int, int]:
    flow, distance = _integer_matrices(instance)
    workspace = _workspace(instance, trace, perm=start)
    candidate_best_cost = int(workspace.current_cost)
    candidate_best_perm = workspace.perm.copy()
    path_moves = 0
    internal_moves = 0
    delta_evals = 0
    while not np.array_equal(workspace.perm, guide):
        if deadline is not None and time.perf_counter() >= deadline:
            break
        mismatched = np.flatnonzero(workspace.perm != guide)
        if len(mismatched) < 2:
            break
        location_to_facility = np.empty(instance.n, dtype=np.int64)
        location_to_facility[workspace.perm] = np.arange(instance.n, dtype=np.int64)
        pairs = np.asarray(
            [
                sorted((int(i), int(location_to_facility[int(guide[i])])))
                for i in mismatched
            ],
            dtype=np.int64,
        )
        pairs = np.unique(pairs, axis=0)
        values = workspace.delta[pairs[:, 0], pairs[:, 1]]
        delta_evals += len(pairs)
        selected = int(np.argmin(values))
        i, j = int(pairs[selected, 0]), int(pairs[selected, 1])
        (
            workspace.current_cost,
            workspace.best_cost,
            workspace.iter_without_improvement,
            workspace.iteration,
            moved,
        ) = _apply_move(
            workspace.perm,
            workspace.delta,
            workspace.current_cost,
            workspace.best_cost,
            workspace.best_perm,
            workspace.last_swapped,
            flow,
            distance,
            workspace.iter_without_improvement,
            workspace.iteration,
            i,
            j,
            "incremental",
        )
        path_moves += int(moved)
        if workspace.current_cost < candidate_best_cost:
            candidate_best_cost = int(workspace.current_cost)
            candidate_best_perm = workspace.perm.copy()
        if path_moves % int(local_stride) == 0:
            active = _active_facilities(instance, workspace.perm, guide, (i, j))
            candidate, moves, evaluated = _restricted_local_candidate(
                instance,
                workspace,
                active,
                max_moves=int(local_moves),
                deadline=deadline,
            )
            internal_moves += int(moves)
            delta_evals += int(evaluated)
            if candidate.current_cost < candidate_best_cost:
                candidate_best_cost = int(candidate.current_cost)
                candidate_best_perm = candidate.perm.copy()

    workspace = _workspace(instance, trace, perm=candidate_best_perm)
    descent_moves, descent_evals = _descent(instance, workspace)
    delta_evals += int(descent_evals)
    return workspace, path_moves, internal_moves, descent_moves, delta_evals


def run_d28_bidirectional_relink(
    instance: QAPInstance,
    trace: OfficialBLSStateTrace,
    guide: np.ndarray,
    *,
    local_stride: int = 2,
    local_moves: int = 2,
    deadline: float | None = None,
) -> D28RelinkResult:
    source = np.asarray(trace.perm, dtype=np.int64)
    target = np.asarray(guide, dtype=np.int64)
    branches = [
        _relink_branch_with_internal_search(
            instance,
            trace,
            source,
            target,
            local_stride=int(local_stride),
            local_moves=int(local_moves),
            deadline=deadline,
        ),
        _relink_branch_with_internal_search(
            instance,
            trace,
            target,
            source,
            local_stride=int(local_stride),
            local_moves=int(local_moves),
            deadline=deadline,
        ),
    ]
    chosen = min(branches, key=lambda branch: branch[0].current_cost)
    best_branch = min(branches, key=lambda branch: branch[0].best_cost)
    chosen[0].best_cost = int(best_branch[0].best_cost)
    chosen[0].best_perm[:] = best_branch[0].best_perm
    return D28RelinkResult(
        workspace=chosen[0],
        path_moves=sum(branch[1] for branch in branches),
        internal_moves=sum(branch[2] for branch in branches),
        descent_moves=sum(branch[3] for branch in branches),
        delta_evals=sum(branch[4] for branch in branches),
    )


def run_d28_sparse_unidirectional_relink(
    instance: QAPInstance,
    trace: OfficialBLSStateTrace,
    guide: np.ndarray,
    *,
    max_steps: int = 20,
    deadline: float | None = None,
) -> D28RelinkResult:
    """Follow one sparse path, then close only its best intermediate solution."""
    flow, distance = _integer_matrices(instance)
    path_perm = np.asarray(trace.perm, dtype=np.int64).copy()
    target = np.asarray(guide, dtype=np.int64)
    current_cost = int(trace.current_cost)
    path_iteration = int(trace.iteration)
    path_last_swapped = np.asarray(trace.last_swapped, dtype=np.int64).copy()
    best_path_cost: int | None = None
    best_path_perm: np.ndarray | None = None
    best_path_iteration = path_iteration
    best_path_last_swapped = path_last_swapped.copy()
    path_moves = 0
    delta_evals = 0

    for _ in range(int(max_steps)):
        if deadline is not None and time.perf_counter() >= deadline:
            break
        mismatched = np.flatnonzero(path_perm != target)
        if len(mismatched) < 2:
            break
        location_to_facility = np.empty(instance.n, dtype=np.int64)
        location_to_facility[path_perm] = np.arange(instance.n, dtype=np.int64)
        pairs = np.asarray(
            [
                sorted((int(i), int(location_to_facility[int(target[i])])))
                for i in mismatched
            ],
            dtype=np.int64,
        )
        pairs = np.unique(pairs, axis=0)
        values = _compute_deltas_for_pairs(path_perm, flow, distance, pairs)
        delta_evals += len(pairs)
        selected = int(np.argmin(values))
        i, j = int(pairs[selected, 0]), int(pairs[selected, 1])
        path_last_swapped[i, j] = path_iteration
        path_last_swapped[j, i] = path_iteration
        path_perm[i], path_perm[j] = path_perm[j], path_perm[i]
        current_cost += int(values[selected])
        path_iteration += 1
        path_moves += 1
        if best_path_cost is None or current_cost < best_path_cost:
            best_path_cost = int(current_cost)
            best_path_perm = path_perm.copy()
            best_path_iteration = int(path_iteration)
            best_path_last_swapped = path_last_swapped.copy()

    if best_path_perm is None:
        workspace = _workspace(instance, trace)
        return D28RelinkResult(
            workspace=workspace,
            path_moves=0,
            internal_moves=0,
            descent_moves=0,
            delta_evals=delta_evals,
        )

    workspace = _workspace(instance, trace, perm=best_path_perm)
    if workspace.current_cost != int(best_path_cost):
        raise RuntimeError("sparse D28 path cost drifted from the exact QAP objective")
    workspace.iteration = int(best_path_iteration)
    workspace.last_swapped[:, :] = best_path_last_swapped
    descent_moves, descent_evals = _descent(instance, workspace)
    delta_evals += int(descent_evals)
    return D28RelinkResult(
        workspace=workspace,
        path_moves=path_moves,
        internal_moves=0,
        descent_moves=descent_moves,
        delta_evals=delta_evals,
    )


__all__ = [
    "D28EliteEntry",
    "D28RelinkResult",
    "choose_diverse_elite_guide",
    "run_d28_bidirectional_relink",
    "run_d28_sparse_unidirectional_relink",
    "update_diverse_elite_pool",
]
