from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np

from .qap import QAPInstance, compute_cost, is_valid_perm


@dataclass(frozen=True)
class VDSSResult:
    perm: np.ndarray
    cost: float
    gain: float
    accepted: bool
    depth: int
    attempts: int
    start_node: int | None


def relocation_gain(
    locations: np.ndarray,
    F: np.ndarray,
    D: np.ndarray,
    facility: int,
    destination: int,
) -> float:
    """Exact objective reduction when one facility changes location.

    Intermediate location vectors may be infeasible during an ejection chain. The
    directed terms are evaluated explicitly so the formula also covers asymmetric
    QAP instances and nonzero diagonals.
    """
    loc = np.asarray(locations, dtype=np.int64)
    u = int(facility)
    old = int(loc[u])
    new = int(destination)
    if old == new:
        return 0.0
    old_cost = float(F[u, u] * D[old, old])
    new_cost = float(F[u, u] * D[new, new])
    for v in range(len(loc)):
        if v == u:
            continue
        lv = int(loc[v])
        old_cost += float(F[u, v] * D[old, lv] + F[v, u] * D[lv, old])
        new_cost += float(F[u, v] * D[new, lv] + F[v, u] * D[lv, new])
    return old_cost - new_cost


def _cost_from_locations(locations: np.ndarray, F: np.ndarray, D: np.ndarray) -> float:
    loc = np.asarray(locations, dtype=np.int64)
    return float(np.sum(F * D[np.ix_(loc, loc)]))


def find_vdss_improvement(
    instance: QAPInstance,
    perm: np.ndarray,
    rng: np.random.Generator,
    depths: tuple[int, ...] = (2, 5),
    max_attempts_per_start: int = 256,
    deadline: float | None = None,
) -> VDSSResult:
    """Return the first positive-gain relocation cycle found by bounded VDSS."""
    original = np.asarray(perm, dtype=np.int64)
    if not is_valid_perm(original, instance.n):
        raise ValueError("perm must be a valid QAP permutation.")
    ordered_depths = tuple(int(depth) for depth in depths)
    if not ordered_depths or any(depth < 2 for depth in ordered_depths):
        raise ValueError("VDSS depths must contain integers >= 2.")
    if any(right <= left for left, right in zip(ordered_depths, ordered_depths[1:])):
        raise ValueError("VDSS depths must be strictly increasing.")
    if int(max_attempts_per_start) <= 0:
        raise ValueError("max_attempts_per_start must be positive.")

    start_cost = compute_cost(original, instance.F, instance.D)
    total_attempts = 0
    n = instance.n
    start_order = rng.permutation(n).astype(np.int64)

    for max_depth in ordered_depths:
        for start_value in start_order:
            if deadline is not None and time.perf_counter() >= deadline:
                return VDSSResult(original.copy(), start_cost, 0.0, False, 0, total_attempts, None)
            start = int(start_value)
            locations = original.copy()
            attempts_for_start = 0

            def attempt_move(facility: int, destination: int) -> float | None:
                nonlocal attempts_for_start, total_attempts
                if attempts_for_start >= int(max_attempts_per_start):
                    return None
                if deadline is not None and time.perf_counter() >= deadline:
                    return None
                attempts_for_start += 1
                total_attempts += 1
                return relocation_gain(locations, instance.F, instance.D, facility, destination)

            def dfs(path: list[int], cumulative_gain: float) -> VDSSResult | None:
                current = int(path[-1])
                if len(path) >= 2:
                    close_gain = attempt_move(current, int(original[start]))
                    if close_gain is None:
                        return None
                    if cumulative_gain + close_gain > 1e-12:
                        old_location = int(locations[current])
                        locations[current] = int(original[start])
                        candidate = locations.copy()
                        locations[current] = old_location
                        if not is_valid_perm(candidate, n):
                            raise AssertionError("VDSS closed cycle did not produce a permutation.")
                        candidate_cost = compute_cost(candidate, instance.F, instance.D)
                        exact_gain = start_cost - candidate_cost
                        if exact_gain > 1e-12:
                            return VDSSResult(
                                perm=candidate,
                                cost=float(candidate_cost),
                                gain=float(exact_gain),
                                accepted=True,
                                depth=len(path),
                                attempts=total_attempts,
                                start_node=start,
                            )
                if len(path) >= int(max_depth):
                    return None

                candidate_moves: list[tuple[float, int]] = []
                used = set(path)
                for next_node in range(n):
                    if next_node in used:
                        continue
                    gain = attempt_move(current, int(original[next_node]))
                    if gain is None:
                        break
                    if cumulative_gain + gain > 1e-12:
                        candidate_moves.append((float(gain), int(next_node)))
                candidate_moves.sort(key=lambda item: (-item[0], item[1]))
                for gain, next_node in candidate_moves:
                    old_location = int(locations[current])
                    locations[current] = int(original[next_node])
                    result = dfs(path + [next_node], cumulative_gain + gain)
                    locations[current] = old_location
                    if result is not None:
                        return result
                    if attempts_for_start >= int(max_attempts_per_start):
                        break
                    if deadline is not None and time.perf_counter() >= deadline:
                        break
                return None

            result = dfs([start], 0.0)
            if result is not None:
                return result

    return VDSSResult(original.copy(), start_cost, 0.0, False, 0, total_attempts, None)


__all__ = ["VDSSResult", "find_vdss_improvement", "relocation_gain", "_cost_from_locations"]
