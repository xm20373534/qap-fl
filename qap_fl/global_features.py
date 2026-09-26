from __future__ import annotations

import numpy as np

from .features import EPS, FEATURE_FIELDS, SwapFeatureCache, build_swap_feature_cache, build_swap_features
from .qap import QAPInstance


GLOBAL_FEATURE_NAMES = [
    "global_wfd_current",
    "global_wfd_delta",
    "global_topflow_wfd_current",
    "global_topflow_wfd_delta",
    "global_incident_flow_weight",
    "global_incident_topflow_weight",
    "global_incident_stress_current",
    "global_incident_stress_delta",
    "global_strength_centrality_old",
    "global_strength_centrality_swap",
    "global_strength_centrality_delta",
    "global_top_stress_touched",
]
GLOBAL_FEATURE_FIELDS = [f"feature_{name}" for name in GLOBAL_FEATURE_NAMES]
GLOBAL_SWAP_FEATURE_FIELDS = FEATURE_FIELDS + GLOBAL_FEATURE_FIELDS


def _upper_pairs(n: int) -> tuple[np.ndarray, np.ndarray]:
    return np.triu_indices(int(n), k=1)


def _pair_weight_matrix(F: np.ndarray) -> np.ndarray:
    weights = np.abs(F) + np.abs(F.T)
    np.fill_diagonal(weights, 0.0)
    return weights.astype(np.float64)


def _symmetric_distance_matrix(D: np.ndarray) -> np.ndarray:
    distances = 0.5 * (np.abs(D) + np.abs(D.T))
    np.fill_diagonal(distances, 0.0)
    return distances.astype(np.float64)


def _top_mask_from_upper_values(values: np.ndarray, n: int, fraction: float) -> np.ndarray:
    mask = np.zeros((n, n), dtype=bool)
    if len(values) == 0:
        return mask
    count = max(1, int(round(float(fraction) * len(values))))
    count = min(count, len(values))
    order = np.argsort(-values[:, 2], kind="stable")[:count]
    for idx in order:
        i = int(values[idx, 0])
        j = int(values[idx, 1])
        mask[i, j] = True
        mask[j, i] = True
    return mask


def _layout_state_values(
    perm: np.ndarray,
    pair_weights: np.ndarray,
    distances: np.ndarray,
    top_flow_mask: np.ndarray,
) -> tuple[float, float, np.ndarray]:
    n = len(perm)
    tri = _upper_pairs(n)
    assigned = distances[np.ix_(perm, perm)]
    weights = pair_weights[tri]
    total_weight = max(float(weights.sum()), EPS)
    dist_scale = float(np.mean(distances[tri])) + EPS
    wfd = float(np.sum(weights * assigned[tri]) / total_weight / dist_scale)

    top_weights_matrix = np.where(top_flow_mask, pair_weights, 0.0)
    top_weights = top_weights_matrix[tri]
    top_total = max(float(top_weights.sum()), EPS)
    top_wfd = float(np.sum(top_weights * assigned[tri]) / top_total / dist_scale)
    stress = pair_weights * assigned / dist_scale
    np.fill_diagonal(stress, 0.0)
    return wfd, top_wfd, stress


def _swap_weighted_distance_delta(
    perm: np.ndarray,
    i: int,
    j: int,
    weights: np.ndarray,
    distances: np.ndarray,
    total_weight: float,
    dist_scale: float,
) -> float:
    pi = int(perm[i])
    pj = int(perm[j])
    delta = 0.0
    n = len(perm)
    for k in range(n):
        if k == i or k == j:
            continue
        pk = int(perm[k])
        delta += weights[i, k] * (distances[pj, pk] - distances[pi, pk])
        delta += weights[j, k] * (distances[pi, pk] - distances[pj, pk])
    delta += weights[i, j] * (distances[pj, pi] - distances[pi, pj])
    return float(delta / max(total_weight, EPS) / dist_scale)


def build_global_swap_features(
    instance: QAPInstance,
    perm: np.ndarray,
    pairs: np.ndarray,
    feature_cache: SwapFeatureCache | None = None,
    top_flow_fraction: float = 0.15,
    top_stress_fraction: float = 0.15,
) -> np.ndarray:
    perm = np.asarray(perm, dtype=np.int64)
    pairs = np.asarray(pairs, dtype=np.int64)
    if len(pairs) == 0:
        return np.zeros((0, len(GLOBAL_FEATURE_NAMES)), dtype=np.float32)

    cache = build_swap_feature_cache(instance) if feature_cache is None else feature_cache
    F = cache.F.astype(np.float64)
    D = cache.D.astype(np.float64)
    n = instance.n
    tri = _upper_pairs(n)
    pair_weights = _pair_weight_matrix(F)
    distances = _symmetric_distance_matrix(D)
    dist_scale = float(np.mean(distances[tri])) + EPS
    total_weight = max(float(pair_weights[tri].sum()), EPS)

    upper_flow_values = np.stack(
        [tri[0].astype(np.float64), tri[1].astype(np.float64), pair_weights[tri]],
        axis=1,
    )
    top_flow_mask = _top_mask_from_upper_values(upper_flow_values, n, top_flow_fraction)
    top_flow_weights = np.where(top_flow_mask, pair_weights, 0.0)
    top_flow_total = max(float(top_flow_weights[tri].sum()), EPS)

    current_wfd, current_top_wfd, current_stress = _layout_state_values(
        perm=perm,
        pair_weights=pair_weights,
        distances=distances,
        top_flow_mask=top_flow_mask,
    )
    stress_values = np.stack(
        [tri[0].astype(np.float64), tri[1].astype(np.float64), current_stress[tri]],
        axis=1,
    )
    top_stress_mask = _top_mask_from_upper_values(stress_values, n, top_stress_fraction)

    facility_strength = pair_weights.sum(axis=1)
    facility_strength = facility_strength / (float(np.mean(facility_strength)) + EPS)
    location_centrality = distances.sum(axis=1)
    location_centrality = location_centrality / (float(np.mean(location_centrality)) + EPS)

    features = np.zeros((len(pairs), len(GLOBAL_FEATURE_NAMES)), dtype=np.float32)
    for row, (i_raw, j_raw) in enumerate(pairs):
        i = int(i_raw)
        j = int(j_raw)
        pi = int(perm[i])
        pj = int(perm[j])

        wfd_delta = _swap_weighted_distance_delta(perm, i, j, pair_weights, distances, total_weight, dist_scale)
        top_wfd_delta = _swap_weighted_distance_delta(perm, i, j, top_flow_weights, distances, top_flow_total, dist_scale)
        incident_weight = pair_weights[i].sum() + pair_weights[j].sum() - 2.0 * pair_weights[i, j]
        incident_top_weight = top_flow_weights[i].sum() + top_flow_weights[j].sum() - 2.0 * top_flow_weights[i, j]
        incident_stress = current_stress[i].sum() + current_stress[j].sum() - 2.0 * current_stress[i, j]
        incident_delta = _swap_weighted_distance_delta(perm, i, j, pair_weights, distances, max(incident_weight, EPS), dist_scale)

        old_match = facility_strength[i] * location_centrality[pi] + facility_strength[j] * location_centrality[pj]
        swap_match = facility_strength[i] * location_centrality[pj] + facility_strength[j] * location_centrality[pi]
        top_stress_touched = (
            top_stress_mask[i].sum() + top_stress_mask[j].sum() - 2.0 * int(top_stress_mask[i, j])
        )

        features[row, 0] = current_wfd
        features[row, 1] = wfd_delta
        features[row, 2] = current_top_wfd
        features[row, 3] = top_wfd_delta
        features[row, 4] = incident_weight / total_weight
        features[row, 5] = incident_top_weight / top_flow_total
        features[row, 6] = incident_stress / max(float(n), 1.0)
        features[row, 7] = incident_delta
        features[row, 8] = old_match
        features[row, 9] = swap_match
        features[row, 10] = swap_match - old_match
        features[row, 11] = float(top_stress_touched) / max(float(top_stress_mask.sum() / 2.0), 1.0)
    return features


def build_swap_features_for_fields(
    instance: QAPInstance,
    perm: np.ndarray,
    pairs: np.ndarray,
    feature_fields: list[str],
    feature_cache: SwapFeatureCache | None = None,
) -> np.ndarray:
    requested = list(feature_fields)
    base_index = {field: idx for idx, field in enumerate(FEATURE_FIELDS)}
    global_index = {field: idx for idx, field in enumerate(GLOBAL_FEATURE_FIELDS)}
    base_features = None
    global_features = None
    columns = []
    for field in requested:
        if field in base_index:
            if base_features is None:
                base_features = build_swap_features(instance, perm, pairs, feature_cache=feature_cache)
            columns.append(base_features[:, base_index[field]])
        elif field in global_index:
            if global_features is None:
                global_features = build_global_swap_features(instance, perm, pairs, feature_cache=feature_cache)
            columns.append(global_features[:, global_index[field]])
        else:
            raise ValueError(f"unsupported swap feature field: {field}")
    if not columns:
        return np.zeros((len(pairs), 0), dtype=np.float32)
    return np.stack(columns, axis=1).astype(np.float32)
