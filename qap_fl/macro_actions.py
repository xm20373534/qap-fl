"""Macro actions evaluated from a restored Official-BLS local optimum."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
import math
import time

import numpy as np

from .features import all_swap_pairs
from .official_bls import (
    CStdRand,
    OfficialBLSStateTrace,
    _apply_move,
    _best_improvement_move,
    _directed_perturb,
    _initialize_delta,
    _integer_matrices,
    _random_perturb,
    _recency_perturb,
)
from .qap import QAPInstance


@dataclass
class _Workspace:
    perm: np.ndarray
    delta: np.ndarray
    current_cost: int
    best_cost: int
    best_perm: np.ndarray
    last_swapped: np.ndarray
    iteration: int
    iter_without_improvement: int


@dataclass(frozen=True)
class MacroActionResult:
    action: str
    perm: np.ndarray
    current_cost: int
    best_perm: np.ndarray
    best_cost: int
    incumbent_gain: int
    runtime_sec: float
    action_moves: int
    descent_moves: int
    delta_evals: int
    escape_hamming: int
    objective_evals: int = 0


def _cost(perm: np.ndarray, flow: np.ndarray, distance: np.ndarray) -> int:
    return int(np.sum(flow.astype(object) * distance[np.ix_(perm, perm)].astype(object)))


def _workspace(
    instance: QAPInstance,
    trace: OfficialBLSStateTrace,
    perm: np.ndarray | None = None,
) -> _Workspace:
    flow, distance = _integer_matrices(instance)
    current_perm = np.asarray(trace.perm if perm is None else perm, dtype=np.int64).copy()
    current_cost = _cost(current_perm, flow, distance)
    best_perm = np.asarray(trace.best_perm, dtype=np.int64).copy()
    best_cost = int(trace.best_cost)
    if current_cost < best_cost:
        best_cost = current_cost
        best_perm = current_perm.copy()
    return _Workspace(
        perm=current_perm,
        delta=_initialize_delta(current_perm, flow, distance),
        current_cost=current_cost,
        best_cost=best_cost,
        best_perm=best_perm,
        last_swapped=np.asarray(trace.last_swapped, dtype=np.int64).copy(),
        iteration=int(trace.iteration),
        iter_without_improvement=int(trace.iter_without_improvement),
    )


def _descent(
    instance: QAPInstance,
    workspace: _Workspace,
    *,
    max_moves: int = 10_000,
) -> tuple[int, int]:
    flow, distance = _integer_matrices(instance)
    candidate_rng = np.random.default_rng(0)
    moves = 0
    delta_evals = 0
    while moves < int(max_moves):
        (
            accepted,
            workspace.current_cost,
            workspace.best_cost,
            workspace.iter_without_improvement,
            workspace.iteration,
            evaluated,
            _,
            _,
            _,
        ) = _best_improvement_move(
            instance=instance,
            perm=workspace.perm,
            delta=workspace.delta,
            current_cost=workspace.current_cost,
            best_cost=workspace.best_cost,
            best_perm=workspace.best_perm,
            last_swapped=workspace.last_swapped,
            flow=flow,
            distance=distance,
            iter_without_improvement=workspace.iter_without_improvement,
            iteration=workspace.iteration,
            method="full",
            candidate_rng=candidate_rng,
            candidate_swaps=None,
            score_pool_swaps=None,
            selector=None,
            feature_cache=None,
            delta_update_mode="incremental",
        )
        delta_evals += int(evaluated)
        if not accepted:
            break
        moves += 1
    if moves >= int(max_moves):
        raise RuntimeError("macro-action descent did not reach a local optimum")
    return moves, delta_evals


def _official_perturb(
    instance: QAPInstance,
    trace: OfficialBLSStateTrace,
    workspace: _Workspace,
    seed: int,
) -> tuple[int, float]:
    flow, distance = _integer_matrices(instance)
    rng = CStdRand(int(seed))
    n = instance.n
    previous_cost = int(trace.current_cost if trace.outer_previous_cost is None else trace.outer_previous_cost)
    descent_num = int(trace.outer_descent_num)
    strength = float(trace.perturb_strength)
    if workspace.iter_without_improvement > 2500:
        workspace.iter_without_improvement = 0
        strength = n * (0.4 + rng.rand() % 20 / 100.0)
    elif descent_num != 0 and previous_cost != workspace.current_cost:
        workspace.iter_without_improvement += 1
        strength = max(int(math.ceil(0.15 * n)), 5)
    elif previous_cost == workspace.current_cost:
        strength += 1

    initial_cost = previous_cost
    probability = max(math.exp(-float(workspace.iter_without_improvement) / 2500.0), 0.75)
    directed = probability > rng.unit_101()
    moved_total = 0
    for _ in range(int(math.ceil(strength))):
        if directed:
            (
                workspace.current_cost,
                workspace.best_cost,
                workspace.iter_without_improvement,
                workspace.iteration,
                moved,
                _,
            ) = _directed_perturb(
                workspace.perm,
                workspace.delta,
                workspace.current_cost,
                workspace.best_cost,
                workspace.best_perm,
                workspace.last_swapped,
                flow,
                distance,
                initial_cost,
                workspace.iteration,
                workspace.iter_without_improvement,
                rng,
                0.7,
                0.2,
                "incremental",
            )
        elif 0.3 > rng.unit_101():
            (
                workspace.current_cost,
                workspace.best_cost,
                workspace.iter_without_improvement,
                workspace.iteration,
                moved,
            ) = _recency_perturb(
                workspace.perm,
                workspace.delta,
                workspace.current_cost,
                workspace.best_cost,
                workspace.best_perm,
                workspace.last_swapped,
                flow,
                distance,
                workspace.iteration,
                workspace.iter_without_improvement,
                "incremental",
            )
        else:
            (
                workspace.current_cost,
                workspace.best_cost,
                workspace.iter_without_improvement,
                workspace.iteration,
                moved,
            ) = _random_perturb(
                workspace.perm,
                workspace.delta,
                workspace.current_cost,
                workspace.best_cost,
                workspace.best_perm,
                workspace.last_swapped,
                flow,
                distance,
                initial_cost,
                workspace.iteration,
                workspace.iter_without_improvement,
                rng,
                "incremental",
            )
        moved_total += int(moved)
    return moved_total, float(strength)


def _tabu_burst(
    instance: QAPInstance,
    workspace: _Workspace,
    seed: int,
    steps: int,
) -> int:
    flow, distance = _integer_matrices(instance)
    rng = np.random.default_rng(int(seed))
    pairs = all_swap_pairs(instance.n)
    tabu_until = np.zeros((instance.n, instance.n), dtype=np.int64)
    moved = 0
    base_tenure = max(int(round(0.6 * instance.n)), 2)
    for step in range(int(steps)):
        values = workspace.delta[pairs[:, 0], pairs[:, 1]]
        next_costs = workspace.current_cost + values
        admissible = (tabu_until[pairs[:, 0], pairs[:, 1]] <= step) | (next_costs < workspace.best_cost)
        indices = np.flatnonzero(admissible)
        if len(indices) == 0:
            indices = np.arange(len(pairs), dtype=np.int64)
        best_value = np.min(values[indices])
        tied = indices[values[indices] == best_value]
        selected = int(tied[int(rng.integers(0, len(tied)))])
        i, j = int(pairs[selected, 0]), int(pairs[selected, 1])
        (
            workspace.current_cost,
            workspace.best_cost,
            workspace.iter_without_improvement,
            workspace.iteration,
            accepted,
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
        tenure = base_tenure + int(rng.integers(0, max(instance.n // 5, 1) + 1))
        tabu_until[i, j] = tabu_until[j, i] = step + tenure
        moved += int(accepted)
    return moved


def _exact_destroy_repair(
    instance: QAPInstance,
    workspace: _Workspace,
    block: np.ndarray,
) -> tuple[int, int]:
    flow, distance = _integer_matrices(instance)
    block = np.sort(np.asarray(block, dtype=np.int64))
    if len(block) < 3 or len(np.unique(block)) != len(block):
        raise ValueError("destroy block must contain at least three unique facilities")
    if np.any(block < 0) or np.any(block >= instance.n):
        raise ValueError("destroy block contains an out-of-range facility")
    occupied = workspace.perm[block].tolist()
    best_cost = workspace.current_cost
    best_perm = workspace.perm.copy()
    objective_evals = 0
    for assignment in permutations(occupied):
        candidate = workspace.perm.copy()
        candidate[block] = assignment
        candidate_cost = _cost(candidate, flow, distance)
        objective_evals += 1
        if candidate_cost < best_cost:
            best_cost = candidate_cost
            best_perm = candidate
    if np.array_equal(best_perm, workspace.perm):
        return 0, objective_evals
    workspace.perm[:] = best_perm
    workspace.current_cost = int(best_cost)
    workspace.delta[:, :] = _initialize_delta(workspace.perm, flow, distance)
    workspace.iteration += 1
    if workspace.current_cost < workspace.best_cost:
        workspace.best_cost = workspace.current_cost
        workspace.best_perm[:] = workspace.perm
        workspace.iter_without_improvement = 0
    return 1, objective_evals


def _hotspot_block_repair(
    instance: QAPInstance,
    workspace: _Workspace,
    block_size: int,
) -> int:
    flow, distance = _integer_matrices(instance)
    assigned = distance[np.ix_(workspace.perm, workspace.perm)]
    interaction = np.abs(flow.astype(np.float64) * assigned.astype(np.float64))
    activity = interaction.sum(axis=0) + interaction.sum(axis=1) - np.diag(interaction)
    block = np.argsort(-activity, kind="stable")[: min(int(block_size), instance.n)]
    moved, _ = _exact_destroy_repair(instance, workspace, block)
    return moved


def _relink_branch(
    instance: QAPInstance,
    trace: OfficialBLSStateTrace,
    start: np.ndarray,
    guide: np.ndarray,
) -> tuple[_Workspace, int, int, int]:
    flow, distance = _integer_matrices(instance)
    workspace = _workspace(instance, trace, perm=start)
    path_best_cost = workspace.current_cost
    path_best_perm = workspace.perm.copy()
    moves = 0
    while not np.array_equal(workspace.perm, guide):
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
        selected = int(np.argmin(values))
        i, j = int(pairs[selected, 0]), int(pairs[selected, 1])
        (
            workspace.current_cost,
            workspace.best_cost,
            workspace.iter_without_improvement,
            workspace.iteration,
            accepted,
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
        moves += int(accepted)
        if workspace.current_cost < path_best_cost:
            path_best_cost = workspace.current_cost
            path_best_perm = workspace.perm.copy()

    workspace = _workspace(instance, trace, perm=path_best_perm)
    descent_moves, delta_evals = _descent(instance, workspace)
    return workspace, moves, descent_moves, delta_evals


def _bidirectional_relink(
    instance: QAPInstance,
    trace: OfficialBLSStateTrace,
    guide: np.ndarray,
) -> tuple[_Workspace, int, int, int]:
    source = np.asarray(trace.perm, dtype=np.int64)
    target = np.asarray(guide, dtype=np.int64)
    forward, forward_moves, forward_descent, forward_evals = _relink_branch(instance, trace, source, target)
    backward, backward_moves, backward_descent, backward_evals = _relink_branch(instance, trace, target, source)
    chosen = forward if forward.best_cost <= backward.best_cost else backward
    return (
        chosen,
        forward_moves + backward_moves,
        forward_descent + backward_descent,
        forward_evals + backward_evals,
    )


def run_macro_action(
    instance: QAPInstance,
    trace: OfficialBLSStateTrace,
    action: str,
    *,
    seed: int,
    guide: np.ndarray | None = None,
    block_size: int = 5,
    tabu_steps: int | None = None,
) -> MacroActionResult:
    """Apply one macro action and close it with full best-improvement descent."""
    if trace.best_perm is None:
        raise ValueError("macro actions require trace.best_perm")
    started = time.perf_counter()
    start_perm = np.asarray(trace.perm, dtype=np.int64)
    start_best = int(trace.best_cost)
    delta_evals = 0

    if action == "diverse_bidir_relink":
        if guide is None:
            raise ValueError("diverse_bidir_relink requires a guide permutation")
        workspace, action_moves, descent_moves, delta_evals = _bidirectional_relink(instance, trace, guide)
    else:
        workspace = _workspace(instance, trace)
        if action == "official_perturb":
            action_moves, _ = _official_perturb(instance, trace, workspace, seed)
        elif action == "tabu_burst":
            action_moves = _tabu_burst(
                instance,
                workspace,
                seed,
                instance.n if tabu_steps is None else int(tabu_steps),
            )
        elif action == "hotspot_block5":
            action_moves = _hotspot_block_repair(instance, workspace, int(block_size))
        else:
            raise ValueError(f"unsupported macro action: {action}")
        descent_moves, delta_evals = _descent(instance, workspace)

    return MacroActionResult(
        action=action,
        perm=workspace.perm.copy(),
        current_cost=int(workspace.current_cost),
        best_perm=workspace.best_perm.copy(),
        best_cost=int(workspace.best_cost),
        incumbent_gain=max(start_best - int(workspace.best_cost), 0),
        runtime_sec=float(time.perf_counter() - started),
        action_moves=int(action_moves),
        descent_moves=int(descent_moves),
        delta_evals=int(delta_evals),
        escape_hamming=int(np.count_nonzero(workspace.perm != start_perm)),
    )


def run_destroy_repair_macro(
    instance: QAPInstance,
    trace: OfficialBLSStateTrace,
    block: np.ndarray,
) -> MacroActionResult:
    """Exactly repair one destroy set and close it with full descent."""
    if trace.best_perm is None:
        raise ValueError("destroy-repair requires trace.best_perm")
    started = time.perf_counter()
    start_perm = np.asarray(trace.perm, dtype=np.int64)
    start_best = int(trace.best_cost)
    workspace = _workspace(instance, trace)
    action_moves, objective_evals = _exact_destroy_repair(instance, workspace, block)
    descent_moves, delta_evals = _descent(instance, workspace)
    return MacroActionResult(
        action=f"destroy_repair_k{len(block)}",
        perm=workspace.perm.copy(),
        current_cost=int(workspace.current_cost),
        best_perm=workspace.best_perm.copy(),
        best_cost=int(workspace.best_cost),
        incumbent_gain=max(start_best - int(workspace.best_cost), 0),
        runtime_sec=float(time.perf_counter() - started),
        action_moves=int(action_moves),
        descent_moves=int(descent_moves),
        delta_evals=int(delta_evals),
        escape_hamming=int(np.count_nonzero(workspace.perm != start_perm)),
        objective_evals=int(objective_evals),
    )


def run_macro_sequence(
    instance: QAPInstance,
    trace: OfficialBLSStateTrace,
    actions: tuple[str, ...],
    *,
    seed: int,
) -> MacroActionResult:
    """Execute a sequence of perturbation-to-local-optimum macro closures."""
    if trace.best_perm is None:
        raise ValueError("macro sequences require trace.best_perm")
    if not actions:
        raise ValueError("actions must not be empty")
    if any(action not in {"official_perturb", "tabu_burst"} for action in actions):
        raise ValueError("macro sequences support only official_perturb and tabu_burst")

    started = time.perf_counter()
    start_perm = np.asarray(trace.perm, dtype=np.int64)
    start_best = int(trace.best_cost)
    workspace = _workspace(instance, trace)
    current_trace = trace
    perturb_strength = float(trace.perturb_strength)
    total_action_moves = 0
    total_descent_moves = 0
    total_delta_evals = 0

    for step, action in enumerate(actions):
        step_seed = int(seed) + step * 1_009
        if action == "official_perturb":
            action_moves, perturb_strength = _official_perturb(
                instance,
                current_trace,
                workspace,
                step_seed,
            )
        else:
            action_moves = _tabu_burst(instance, workspace, step_seed, instance.n)
        descent_start_cost = int(workspace.current_cost)
        descent_moves, delta_evals = _descent(instance, workspace)
        total_action_moves += int(action_moves)
        total_descent_moves += int(descent_moves)
        total_delta_evals += int(delta_evals)
        current_trace = OfficialBLSStateTrace(
            outer_iteration=int(trace.outer_iteration) + step + 1,
            descent_step=0,
            state_source=action,
            perm=workspace.perm.copy(),
            current_cost=int(workspace.current_cost),
            best_cost=int(workspace.best_cost),
            iteration=int(workspace.iteration),
            iter_without_improvement=int(workspace.iter_without_improvement),
            perturb_strength=float(perturb_strength),
            last_swapped=workspace.last_swapped.copy(),
            best_perm=workspace.best_perm.copy(),
            outer_previous_cost=descent_start_cost,
            outer_descent_num=int(descent_moves),
        )

    return MacroActionResult(
        action=">".join(actions),
        perm=workspace.perm.copy(),
        current_cost=int(workspace.current_cost),
        best_perm=workspace.best_perm.copy(),
        best_cost=int(workspace.best_cost),
        incumbent_gain=max(start_best - int(workspace.best_cost), 0),
        runtime_sec=float(time.perf_counter() - started),
        action_moves=int(total_action_moves),
        descent_moves=int(total_descent_moves),
        delta_evals=int(total_delta_evals),
        escape_hamming=int(np.count_nonzero(workspace.perm != start_perm)),
    )


__all__ = [
    "MacroActionResult",
    "run_destroy_repair_macro",
    "run_macro_action",
    "run_macro_sequence",
]
