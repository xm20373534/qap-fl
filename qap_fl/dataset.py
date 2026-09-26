from __future__ import annotations

import csv
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from .features import FEATURE_FIELDS, all_swap_pairs, build_swap_feature_cache, build_swap_features, sample_swap_pairs
from .local_search import two_opt_local_search
from .qap import QAPInstance, compute_cost, random_perm, swap_delta_cost
from .utils import write_csv


METADATA_FIELDS = [
    "instance",
    "n",
    "state_id",
    "state_source",
    "state_cost",
    "candidate_count",
    "pair_rank",
    "i",
    "j",
    "delta_cost",
    "best_delta_cost",
    "label",
]
DATASET_FIELDS = METADATA_FIELDS + FEATURE_FIELDS


def hard_topk_labels(deltas: np.ndarray, top_k: int) -> np.ndarray:
    labels = np.zeros(len(deltas), dtype=np.float64)
    k = max(1, min(int(top_k), len(deltas)))
    order = np.argsort(deltas)[:k]
    labels[order] = 1.0 / float(k)
    return labels


def greedy_trajectory_states(
    instance: QAPInstance,
    start_perm: np.ndarray,
    max_iters: int,
) -> list[np.ndarray]:
    current = np.asarray(start_perm, dtype=np.int64).copy()
    states: list[np.ndarray] = []
    pairs = all_swap_pairs(instance.n)
    for _ in range(int(max_iters)):
        states.append(current.copy())
        best_delta = 0.0
        best_pair: tuple[int, int] | None = None
        for i_raw, j_raw in pairs:
            i = int(i_raw)
            j = int(j_raw)
            delta = swap_delta_cost(current, instance.F, instance.D, i, j)
            if delta < best_delta:
                best_delta = float(delta)
                best_pair = (i, j)
        if best_pair is None:
            break
        current = current.copy()
        current[best_pair[0]], current[best_pair[1]] = current[best_pair[1]], current[best_pair[0]]
    return states


def generate_states(
    instance: QAPInstance,
    rng: np.random.Generator,
    random_states: int,
    trajectory_starts: int,
    trajectory_iters: int,
) -> list[tuple[str, np.ndarray]]:
    states: list[tuple[str, np.ndarray]] = []
    for _ in range(int(random_states)):
        states.append(("random", random_perm(instance.n, rng)))
    for _ in range(int(trajectory_starts)):
        start = random_perm(instance.n, rng)
        for perm in greedy_trajectory_states(instance, start, trajectory_iters):
            states.append(("greedy_trajectory", perm))
    return states


def build_rows_for_instance(
    instance: QAPInstance,
    states: Sequence[tuple[str, np.ndarray]],
    candidate_swaps: int | None,
    hard_label_top_k: int,
    rng: np.random.Generator,
) -> list[dict]:
    rows: list[dict] = []
    cache = build_swap_feature_cache(instance)
    for state_id, (state_source, perm) in enumerate(states):
        pairs = sample_swap_pairs(instance.n, candidate_swaps, rng)
        features = build_swap_features(instance, perm, pairs, feature_cache=cache)
        deltas = np.asarray(
            [swap_delta_cost(perm, instance.F, instance.D, int(i), int(j)) for i, j in pairs],
            dtype=np.float64,
        )
        labels = hard_topk_labels(deltas, hard_label_top_k)
        state_cost = compute_cost(perm, instance.F, instance.D)
        best_delta = float(np.min(deltas))
        for rank, ((i, j), delta, label, feature_row) in enumerate(zip(pairs, deltas, labels, features)):
            row = {
                "instance": instance.name,
                "n": instance.n,
                "state_id": state_id,
                "state_source": state_source,
                "state_cost": state_cost,
                "candidate_count": len(pairs),
                "pair_rank": rank,
                "i": int(i),
                "j": int(j),
                "delta_cost": float(delta),
                "best_delta_cost": best_delta,
                "label": float(label),
            }
            for name, value in zip(FEATURE_FIELDS, feature_row):
                row[name] = float(value)
            rows.append(row)
    return rows


def build_dataset(
    instances: Sequence[QAPInstance],
    random_states: int,
    trajectory_starts: int,
    trajectory_iters: int,
    candidate_swaps: int | None,
    hard_label_top_k: int,
    seed: int,
    out_path: str | Path,
) -> list[dict]:
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    for instance in instances:
        states = generate_states(instance, rng, random_states, trajectory_starts, trajectory_iters)
        rows.extend(build_rows_for_instance(instance, states, candidate_swaps, hard_label_top_k, rng))
    write_csv(out_path, rows, DATASET_FIELDS)
    return rows


def dataset_feature_fields(path: str | Path) -> list[str]:
    with Path(path).open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no header.")
        return [field for field in reader.fieldnames if field.startswith("feature_")]

