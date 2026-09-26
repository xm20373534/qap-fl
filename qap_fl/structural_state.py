"""QAP-specific state summaries for macro-action selection."""

from __future__ import annotations

import numpy as np

from .features import all_swap_pairs
from .official_bls import OfficialBLSStateTrace, _initialize_delta, _integer_matrices
from .qap import QAPInstance


EPS = 1e-12


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64).reshape(-1)
    right = np.asarray(right, dtype=np.float64).reshape(-1)
    if left.size == 0 or np.std(left) <= EPS or np.std(right) <= EPS:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def _average_ranks(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1)
        start = stop
    return ranks


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    return _correlation(_average_ranks(left), _average_ranks(right))


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64).reshape(-1)
    right = np.asarray(right, dtype=np.float64).reshape(-1)
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / max(denominator, EPS))


def build_structural_state_features(
    instance: QAPInstance,
    trace: OfficialBLSStateTrace,
    *,
    delta: np.ndarray | None = None,
) -> dict[str, float]:
    """Summarize the current F-D coupling and macro-controller Markov state."""

    perm = np.asarray(trace.perm, dtype=np.int64)
    flow = np.asarray(instance.F, dtype=np.float64)
    distance = np.asarray(instance.D, dtype=np.float64)
    assigned_distance = distance[np.ix_(perm, perm)]
    n = instance.n
    off_diagonal = ~np.eye(n, dtype=bool)
    edge_flow = flow[off_diagonal]
    edge_distance = assigned_distance[off_diagonal]
    abs_flow = np.abs(edge_flow)
    abs_distance = np.abs(edge_distance)
    interaction = edge_flow * edge_distance
    abs_interaction = np.abs(interaction)
    mean_flow = max(float(np.mean(np.abs(flow))), EPS)
    mean_distance = max(float(np.mean(np.abs(distance))), EPS)

    top_count = max(int(np.ceil(0.10 * len(abs_flow))), 1)
    top_flow_indices = np.argsort(abs_flow)[-top_count:]
    distance_q25 = float(np.quantile(abs_distance, 0.25))
    interaction_total = max(float(np.sum(abs_interaction)), EPS)

    flow_strength = np.sum(np.abs(flow), axis=1)
    assigned_distance_strength = np.sum(np.abs(assigned_distance), axis=1)
    signed_node_contribution = (
        np.sum(flow * assigned_distance, axis=0)
        + np.sum(flow * assigned_distance, axis=1)
        - np.diag(flow * assigned_distance)
    )
    absolute_node_contribution = np.abs(signed_node_contribution)

    if delta is None:
        integer_flow, integer_distance = _integer_matrices(instance)
        delta = _initialize_delta(perm, integer_flow, integer_distance)
    pairs = all_swap_pairs(n)
    barriers = np.asarray(delta, dtype=np.float64)[pairs[:, 0], pairs[:, 1]]
    barrier_scale = mean_flow * mean_distance * max(n, 1)
    scaled_barriers = barriers / max(barrier_scale, EPS)
    node_minimum = np.full(n, np.inf, dtype=np.float64)
    for (left, right), value in zip(pairs, scaled_barriers):
        node_minimum[int(left)] = min(node_minimum[int(left)], float(value))
        node_minimum[int(right)] = min(node_minimum[int(right)], float(value))
    finite_node_minimum = node_minimum[np.isfinite(node_minimum)]
    near_zero_scale = max(float(np.median(np.abs(scaled_barriers))), EPS)

    last_swapped = np.asarray(trace.last_swapped, dtype=np.float64)
    pair_last_swapped = last_swapped[pairs[:, 0], pairs[:, 1]]
    pair_age = np.maximum(float(trace.iteration) - pair_last_swapped, 0.0) / max(n, 1)
    best_gap = max(float(trace.current_cost) - float(trace.best_cost), 0.0) / max(
        abs(float(trace.best_cost)), 1.0
    )

    return {
        "state_d36_edge_coupling_cosine": _cosine(edge_flow, edge_distance),
        "state_d36_edge_coupling_pearson": _correlation(edge_flow, edge_distance),
        "state_d36_edge_coupling_spearman": _spearman(edge_flow, edge_distance),
        "state_d36_abs_edge_coupling_pearson": _correlation(abs_flow, abs_distance),
        "state_d36_abs_edge_coupling_spearman": _spearman(abs_flow, abs_distance),
        "state_d36_top_flow_distance_ratio": float(
            np.mean(abs_distance[top_flow_indices]) / mean_distance
        ),
        "state_d36_top_flow_short_fraction": float(
            np.mean(abs_distance[top_flow_indices] <= distance_q25)
        ),
        "state_d36_top_flow_interaction_fraction": float(
            np.sum(abs_interaction[top_flow_indices]) / interaction_total
        ),
        "state_d36_node_strength_pearson": _correlation(
            flow_strength, assigned_distance_strength
        ),
        "state_d36_node_strength_spearman": _spearman(
            flow_strength, assigned_distance_strength
        ),
        "state_d36_signed_node_contribution_cv": float(
            np.std(signed_node_contribution)
            / max(abs(float(np.mean(signed_node_contribution))), np.std(signed_node_contribution), EPS)
        ),
        "state_d36_top_node_contribution_fraction": float(
            np.sum(np.sort(absolute_node_contribution)[-min(3, n) :])
            / max(float(np.sum(absolute_node_contribution)), EPS)
        ),
        "state_d36_delta_min": float(np.min(scaled_barriers)),
        "state_d36_delta_q10": float(np.quantile(scaled_barriers, 0.10)),
        "state_d36_delta_q25": float(np.quantile(scaled_barriers, 0.25)),
        "state_d36_delta_median": float(np.median(scaled_barriers)),
        "state_d36_delta_q75": float(np.quantile(scaled_barriers, 0.75)),
        "state_d36_delta_q90": float(np.quantile(scaled_barriers, 0.90)),
        "state_d36_delta_mean": float(np.mean(scaled_barriers)),
        "state_d36_delta_std": float(np.std(scaled_barriers)),
        "state_d36_delta_near_zero_fraction": float(
            np.mean(np.abs(scaled_barriers) <= 0.10 * near_zero_scale)
        ),
        "state_d36_node_min_delta_mean": float(np.mean(finite_node_minimum)),
        "state_d36_node_min_delta_cv": float(
            np.std(finite_node_minimum)
            / max(abs(float(np.mean(finite_node_minimum))), np.std(finite_node_minimum), EPS)
        ),
        "state_d36_current_best_gap": best_gap,
        "state_d36_stagnation_ratio": float(
            np.log1p(max(int(trace.iter_without_improvement), 0))
            / max(np.log1p(max(int(trace.iteration), 1)), EPS)
        ),
        "state_d36_perturb_strength_norm": float(trace.perturb_strength) / max(n, 1),
        "state_d36_outer_descent_norm": float(trace.outer_descent_num) / max(n, 1),
        "state_d36_tabu_unseen_fraction": float(np.mean(pair_last_swapped <= 0.0)),
        "state_d36_tabu_age_median": float(np.median(pair_age)),
        "state_d36_tabu_age_q90": float(np.quantile(pair_age, 0.90)),
    }

