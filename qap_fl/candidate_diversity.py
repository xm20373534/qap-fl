from __future__ import annotations

import numpy as np


def endpoint_graph_metrics(pairs: np.ndarray, n: int) -> dict[str, float]:
    pairs = np.asarray(pairs, dtype=np.int64)
    degrees = np.zeros(int(n), dtype=np.float64)
    if len(pairs):
        np.add.at(degrees, pairs[:, 0], 1.0)
        np.add.at(degrees, pairs[:, 1], 1.0)
    total = float(degrees.sum())
    if total <= 0.0:
        return {
            "unique_endpoints": 0.0,
            "max_endpoint_degree": 0.0,
            "endpoint_degree_gini": 0.0,
            "effective_endpoints": 0.0,
        }
    sorted_degrees = np.sort(degrees)
    ranks = np.arange(1, len(sorted_degrees) + 1, dtype=np.float64)
    gini = float(
        (2.0 * np.sum(ranks * sorted_degrees) / (len(sorted_degrees) * total))
        - (len(sorted_degrees) + 1.0) / len(sorted_degrees)
    )
    effective = float(total * total / max(float(np.sum(degrees * degrees)), 1e-12))
    return {
        "unique_endpoints": float(np.count_nonzero(degrees)),
        "max_endpoint_degree": float(degrees.max()),
        "endpoint_degree_gini": gini,
        "effective_endpoints": effective,
    }


def top_score_indices(scores: np.ndarray, k: int) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    use_k = min(max(int(k), 0), len(scores))
    return np.argsort(-scores, kind="stable")[:use_k].astype(np.int64)


def degree_cap_indices(
    pairs: np.ndarray,
    scores: np.ndarray,
    k: int,
    initial_cap: int,
) -> np.ndarray:
    pairs = np.asarray(pairs, dtype=np.int64)
    order = top_score_indices(scores, len(pairs)).tolist()
    use_k = min(max(int(k), 0), len(order))
    if use_k == 0:
        return np.zeros(0, dtype=np.int64)
    n = int(pairs.max()) + 1
    degree = np.zeros(n, dtype=np.int64)
    selected: list[int] = []
    remaining = order
    cap = max(int(initial_cap), 1)
    while len(selected) < use_k:
        blocked: list[int] = []
        for index in remaining:
            i, j = map(int, pairs[index])
            if degree[i] < cap and degree[j] < cap:
                selected.append(index)
                degree[i] += 1
                degree[j] += 1
                if len(selected) == use_k:
                    break
            else:
                blocked.append(index)
        if len(selected) == use_k:
            break
        remaining = blocked
        cap += 1
    return np.asarray(selected, dtype=np.int64)


def diversity_penalty_indices(
    pairs: np.ndarray,
    scores: np.ndarray,
    k: int,
    penalty: float,
) -> np.ndarray:
    pairs = np.asarray(pairs, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    use_k = min(max(int(k), 0), len(pairs))
    if use_k == 0:
        return np.zeros(0, dtype=np.int64)
    score_std = float(np.std(scores))
    normalized = scores - float(np.mean(scores))
    if score_std > 1e-12:
        normalized = normalized / score_std
    n = int(pairs.max()) + 1
    degree = np.zeros(n, dtype=np.int64)
    available = np.ones(len(pairs), dtype=bool)
    selected = np.zeros(use_k, dtype=np.int64)
    for position in range(use_k):
        endpoint_use = degree[pairs[:, 0]] + degree[pairs[:, 1]]
        adjusted = normalized - float(penalty) * endpoint_use
        adjusted[~available] = -np.inf
        index = int(np.argmax(adjusted))
        selected[position] = index
        available[index] = False
        i, j = map(int, pairs[index])
        degree[i] += 1
        degree[j] += 1
    return selected

