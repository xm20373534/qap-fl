"""Candidate destroy sets for the D35 factorized Neural-LNS oracle gate."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .features import all_swap_pairs
from .official_bls import _initialize_delta, _integer_matrices
from .qap import QAPInstance


@dataclass(frozen=True)
class DestroySetCandidate:
    facilities: np.ndarray
    source: str


def _connected_completion(
    interaction: np.ndarray,
    initial: list[int],
    k: int,
) -> np.ndarray:
    selected = list(dict.fromkeys(int(value) for value in initial))
    while len(selected) < int(k):
        remaining = np.asarray(
            [index for index in range(len(interaction)) if index not in selected],
            dtype=np.int64,
        )
        scores = np.sum(interaction[np.ix_(remaining, np.asarray(selected, dtype=np.int64))], axis=1)
        selected.append(int(remaining[int(np.argmax(scores))]))
    return np.sort(np.asarray(selected[: int(k)], dtype=np.int64))


def build_destroy_set_bank(
    instance: QAPInstance,
    perm: np.ndarray,
    *,
    k: int,
    count: int,
    rng: np.random.Generator,
    structural_count: int = 8,
) -> list[DestroySetCandidate]:
    """Build a small, diverse lower-bound sample of the factorized subset space."""
    n = int(instance.n)
    if not 3 <= int(k) <= n:
        raise ValueError("k must be in [3, n]")
    if int(count) < int(structural_count) or int(structural_count) < 2:
        raise ValueError("count must cover at least two structural candidates")

    flow, distance = _integer_matrices(instance)
    current = np.asarray(perm, dtype=np.int64)
    assigned = distance[np.ix_(current, current)]
    interaction = np.abs(flow.astype(np.float64) * assigned.astype(np.float64))
    interaction += interaction.T
    np.fill_diagonal(interaction, 0.0)
    activity = np.sum(interaction, axis=1)
    delta = _initialize_delta(current, flow, distance)
    pairs = all_swap_pairs(n)
    pair_order = np.argsort(delta[pairs[:, 0], pairs[:, 1]], kind="stable")

    selected: dict[tuple[int, ...], str] = {}
    connected_count = int(structural_count) // 2
    for seed in np.argsort(-activity, kind="stable")[:connected_count]:
        block = _connected_completion(interaction, [int(seed)], int(k))
        selected[tuple(int(value) for value in block)] = "interaction_connected"

    barrier_target = int(structural_count) - connected_count
    for pair_index in pair_order:
        pair = pairs[int(pair_index)]
        block = _connected_completion(interaction, [int(pair[0]), int(pair[1])], int(k))
        key = tuple(int(value) for value in block)
        if key in selected:
            continue
        selected[key] = "low_barrier_connected"
        if sum(source == "low_barrier_connected" for source in selected.values()) >= barrier_target:
            break

    attempts = 0
    while len(selected) < int(count):
        block = np.sort(rng.choice(n, size=int(k), replace=False).astype(np.int64))
        selected.setdefault(tuple(int(value) for value in block), "random")
        attempts += 1
        if attempts > 100 * int(count):
            raise RuntimeError("failed to sample enough unique destroy sets")
    return [
        DestroySetCandidate(np.asarray(block, dtype=np.int64), source)
        for block, source in selected.items()
    ]


__all__ = ["DestroySetCandidate", "build_destroy_set_bank"]
