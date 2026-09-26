from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .qap import QAPInstance, is_valid_perm


EPS = 1e-9

FEATURE_NAMES = [
    "facility_sum_i",
    "facility_sum_j",
    "facility_mean_i",
    "facility_mean_j",
    "facility_std_i",
    "facility_std_j",
    "location_sum_pi_i",
    "location_sum_pi_j",
    "location_mean_pi_i",
    "location_mean_pi_j",
    "location_std_pi_i",
    "location_std_pi_j",
    "facility_sum_absdiff",
    "facility_std_absdiff",
    "location_sum_absdiff",
    "location_std_absdiff",
    "facility_index_distance",
    "location_assignment_distance",
    "qap_pair_old",
    "qap_pair_swap",
    "qap_pair_delta",
    "facility_row_cosine",
    "facility_row_l1",
    "location_row_cosine",
    "location_row_l1",
    "anchor_external_old",
    "anchor_external_swap",
    "anchor_external_delta_estimate",
    "anchor_delta_mean",
    "anchor_delta_std",
    "anchor_delta_min",
    "anchor_delta_max",
    "assignment_index_i",
    "assignment_index_j",
    "assignment_index_absdiff",
    "node_contrib_i",
    "node_contrib_j",
    "node_contrib_absdiff",
    "node_contrib_sum",
    "instance_log_n",
]

FEATURE_FIELDS = [f"feature_{name}" for name in FEATURE_NAMES]


@dataclass(frozen=True)
class SwapFeatureCache:
    F: np.ndarray
    D: np.ndarray
    f_sum: np.ndarray
    f_mean: np.ndarray
    f_std: np.ndarray
    d_sum: np.ndarray
    d_mean: np.ndarray
    d_std: np.ndarray
    qap_scale: float
    denom: float
    facility_cosine: np.ndarray
    facility_l1: np.ndarray
    location_cosine: np.ndarray
    location_l1: np.ndarray
    anchors: np.ndarray


def all_swap_pairs(n: int) -> np.ndarray:
    pairs = [(i, j) for i in range(int(n) - 1) for j in range(i + 1, int(n))]
    return np.asarray(pairs, dtype=np.int64)


def sample_swap_pairs(n: int, num_pairs: int | None, rng: np.random.Generator) -> np.ndarray:
    pairs = all_swap_pairs(n)
    if num_pairs is None or int(num_pairs) >= len(pairs):
        return pairs
    if int(num_pairs) <= 0:
        raise ValueError("num_pairs must be positive or None.")
    idx = rng.choice(len(pairs), size=int(num_pairs), replace=False)
    return pairs[idx]


def normalized_row_stats(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    matrix = np.asarray(matrix, dtype=np.float64)
    scale = float(np.mean(np.abs(matrix))) + EPS
    n = matrix.shape[0]
    return matrix.sum(axis=1) / (n * scale), matrix.mean(axis=1) / scale, matrix.std(axis=1) / scale


def _qap_scale(F: np.ndarray, D: np.ndarray, n: int) -> float:
    return (float(np.mean(np.abs(F))) + EPS) * (float(np.mean(np.abs(D))) + EPS) * max(int(n), 1)


def _row_cosine_matrix(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    norms = np.linalg.norm(matrix, axis=1)
    return matrix @ matrix.T / (np.outer(norms, norms) + EPS)


def _row_l1_matrix(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    scale = float(np.mean(np.abs(matrix))) + EPS
    return np.mean(np.abs(matrix[:, None, :] - matrix[None, :, :]), axis=2) / scale


def _anchor_index_matrix(F: np.ndarray, max_anchors: int = 5) -> np.ndarray:
    F = np.asarray(F, dtype=np.float64)
    n = F.shape[0]
    degree_order = np.argsort(-np.sum(np.abs(F), axis=1))
    anchors = np.full((n, n, max_anchors), -1, dtype=np.int64)
    for i in range(n):
        for j in range(n):
            selected = [int(k) for k in degree_order if int(k) != i and int(k) != j]
            selected = selected[:max_anchors]
            if selected:
                anchors[i, j, : len(selected)] = np.asarray(selected, dtype=np.int64)
    return anchors


def build_swap_feature_cache(instance: QAPInstance) -> SwapFeatureCache:
    F = np.asarray(instance.F, dtype=np.float64)
    D = np.asarray(instance.D, dtype=np.float64)
    n = instance.n
    f_sum, f_mean, f_std = normalized_row_stats(F)
    d_sum, d_mean, d_std = normalized_row_stats(D)
    return SwapFeatureCache(
        F=F,
        D=D,
        f_sum=f_sum,
        f_mean=f_mean,
        f_std=f_std,
        d_sum=d_sum,
        d_mean=d_mean,
        d_std=d_std,
        qap_scale=_qap_scale(F, D, n),
        denom=float(max(n - 1, 1)),
        facility_cosine=_row_cosine_matrix(F),
        facility_l1=_row_l1_matrix(F),
        location_cosine=_row_cosine_matrix(D),
        location_l1=_row_l1_matrix(D),
        anchors=_anchor_index_matrix(F),
    )


def _anchor_interaction_features_batch(
    cache: SwapFeatureCache,
    perm: np.ndarray,
    i: np.ndarray,
    j: np.ndarray,
    pi_i: np.ndarray,
    pi_j: np.ndarray,
) -> np.ndarray:
    features = np.zeros((len(i), 7), dtype=np.float32)
    if len(i) == 0:
        return features
    anchors = cache.anchors[i, j]
    mask = anchors >= 0
    counts = mask.sum(axis=1)
    active = counts > 0
    if not np.any(active):
        return features

    safe_anchors = np.where(mask, anchors, 0)
    pk = perm[safe_anchors]
    ii = i[:, None]
    jj = j[:, None]
    pii = pi_i[:, None]
    pij = pi_j[:, None]
    F = cache.F
    D = cache.D
    old_terms = (
        F[ii, safe_anchors] * D[pii, pk]
        + F[safe_anchors, ii] * D[pk, pii]
        + F[jj, safe_anchors] * D[pij, pk]
        + F[safe_anchors, jj] * D[pk, pij]
    )
    swap_terms = (
        F[ii, safe_anchors] * D[pij, pk]
        + F[safe_anchors, ii] * D[pk, pij]
        + F[jj, safe_anchors] * D[pii, pk]
        + F[safe_anchors, jj] * D[pk, pii]
    )
    old_terms = np.where(mask, old_terms, 0.0)
    swap_terms = np.where(mask, swap_terms, 0.0)
    delta_terms = swap_terms - old_terms
    expansion = np.zeros(len(i), dtype=np.float64)
    expansion[active] = max(len(perm) - 2, 1) / counts[active]
    scaled_delta = np.where(mask, delta_terms * expansion[:, None] / cache.qap_scale, 0.0)
    mean_delta = np.zeros(len(i), dtype=np.float64)
    mean_delta[active] = scaled_delta.sum(axis=1)[active] / counts[active]
    centered = np.where(mask, scaled_delta - mean_delta[:, None], 0.0)
    std_delta = np.zeros(len(i), dtype=np.float64)
    std_delta[active] = np.sqrt((centered * centered).sum(axis=1)[active] / counts[active])
    min_delta = np.zeros(len(i), dtype=np.float64)
    max_delta = np.zeros(len(i), dtype=np.float64)
    min_delta[active] = np.where(mask, scaled_delta, np.inf).min(axis=1)[active]
    max_delta[active] = np.where(mask, scaled_delta, -np.inf).max(axis=1)[active]

    features[:, 0] = old_terms.sum(axis=1) * expansion / cache.qap_scale
    features[:, 1] = swap_terms.sum(axis=1) * expansion / cache.qap_scale
    features[:, 2] = delta_terms.sum(axis=1) * expansion / cache.qap_scale
    features[:, 3] = mean_delta
    features[:, 4] = std_delta
    features[:, 5] = min_delta
    features[:, 6] = max_delta
    return features


def _node_current_contributions(cache: SwapFeatureCache, perm: np.ndarray) -> np.ndarray:
    perm = np.asarray(perm, dtype=np.int64)
    D_perm = cache.D[np.ix_(perm, perm)]
    incoming = np.sum(cache.F * D_perm, axis=1)
    outgoing = np.sum(cache.F * D_perm, axis=0)
    diagonal = np.diag(cache.F) * np.diag(D_perm)
    return (incoming + outgoing - diagonal) / cache.qap_scale


def build_swap_features(
    instance: QAPInstance,
    perm: np.ndarray,
    pairs: np.ndarray,
    feature_cache: SwapFeatureCache | None = None,
) -> np.ndarray:
    perm = np.asarray(perm, dtype=np.int64)
    pairs = np.asarray(pairs, dtype=np.int64)
    n = instance.n
    if not is_valid_perm(perm, n):
        raise ValueError("perm is not a valid permutation.")
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError("pairs must have shape [num_pairs, 2].")
    if len(pairs) == 0:
        return np.zeros((0, len(FEATURE_NAMES)), dtype=np.float32)

    cache = build_swap_feature_cache(instance) if feature_cache is None else feature_cache
    i = pairs[:, 0].astype(np.int64)
    j = pairs[:, 1].astype(np.int64)
    pi_i = perm[i]
    pi_j = perm[j]
    F = cache.F
    D = cache.D
    pair_old = (
        F[i, i] * D[pi_i, pi_i]
        + F[j, j] * D[pi_j, pi_j]
        + F[i, j] * D[pi_i, pi_j]
        + F[j, i] * D[pi_j, pi_i]
    )
    pair_swap = (
        F[i, i] * D[pi_j, pi_j]
        + F[j, j] * D[pi_i, pi_i]
        + F[i, j] * D[pi_j, pi_i]
        + F[j, i] * D[pi_i, pi_j]
    )
    node_contrib = _node_current_contributions(cache, perm)

    features = np.zeros((len(pairs), len(FEATURE_NAMES)), dtype=np.float32)
    features[:, 0] = cache.f_sum[i]
    features[:, 1] = cache.f_sum[j]
    features[:, 2] = cache.f_mean[i]
    features[:, 3] = cache.f_mean[j]
    features[:, 4] = cache.f_std[i]
    features[:, 5] = cache.f_std[j]
    features[:, 6] = cache.d_sum[pi_i]
    features[:, 7] = cache.d_sum[pi_j]
    features[:, 8] = cache.d_mean[pi_i]
    features[:, 9] = cache.d_mean[pi_j]
    features[:, 10] = cache.d_std[pi_i]
    features[:, 11] = cache.d_std[pi_j]
    features[:, 12] = np.abs(cache.f_sum[i] - cache.f_sum[j])
    features[:, 13] = np.abs(cache.f_std[i] - cache.f_std[j])
    features[:, 14] = np.abs(cache.d_sum[pi_i] - cache.d_sum[pi_j])
    features[:, 15] = np.abs(cache.d_std[pi_i] - cache.d_std[pi_j])
    features[:, 16] = np.abs(i - j) / cache.denom
    features[:, 17] = np.abs(pi_i - pi_j) / cache.denom
    features[:, 18] = pair_old / cache.qap_scale
    features[:, 19] = pair_swap / cache.qap_scale
    features[:, 20] = (pair_swap - pair_old) / cache.qap_scale
    features[:, 21] = cache.facility_cosine[i, j]
    features[:, 22] = cache.facility_l1[i, j]
    features[:, 23] = cache.location_cosine[pi_i, pi_j]
    features[:, 24] = cache.location_l1[pi_i, pi_j]
    features[:, 25:32] = _anchor_interaction_features_batch(cache, perm, i, j, pi_i, pi_j)
    features[:, 32] = pi_i / cache.denom
    features[:, 33] = pi_j / cache.denom
    features[:, 34] = np.abs(pi_i - pi_j) / cache.denom
    features[:, 35] = node_contrib[i]
    features[:, 36] = node_contrib[j]
    features[:, 37] = np.abs(node_contrib[i] - node_contrib[j])
    features[:, 38] = node_contrib[i] + node_contrib[j]
    features[:, 39] = np.log(float(instance.n))
    return features

