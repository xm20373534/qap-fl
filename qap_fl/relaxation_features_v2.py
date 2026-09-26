from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from .features import EPS, FEATURE_FIELDS, SwapFeatureCache, build_swap_features
from .qap import QAPInstance


RELAXATION_FEATURE_NAMES = [
    "spectral_old",
    "spectral_swap",
    "spectral_delta",
    "spectral_i_old",
    "spectral_i_swap",
    "spectral_j_old",
    "spectral_j_swap",
    "faq_current",
    "faq_swap",
    "faq_delta",
    "faq_i_current",
    "faq_i_swap",
    "faq_j_current",
    "faq_j_swap",
    "faq_row_entropy_i",
    "faq_row_entropy_j",
    "faq_row_margin_i",
    "faq_row_margin_j",
    "faq_col_entropy_pi_i",
    "faq_col_entropy_pi_j",
    "faq_perm_score_current",
    "faq_perm_score_delta",
]
RELAXATION_FEATURE_FIELDS = [f"feature_{name}" for name in RELAXATION_FEATURE_NAMES]
RELAXATION_SWAP_FEATURE_FIELDS = FEATURE_FIELDS + RELAXATION_FEATURE_FIELDS


@dataclass(frozen=True)
class RelaxationFeatureCache:
    spectral_cost: np.ndarray
    faq_p: np.ndarray
    faq_row_entropy: np.ndarray
    faq_col_entropy: np.ndarray
    faq_row_margin: np.ndarray
    faq_perm_score_scale: float


def _standardize_symmetric(matrix: np.ndarray) -> np.ndarray:
    matrix = 0.5 * (np.asarray(matrix, dtype=np.float64) + np.asarray(matrix, dtype=np.float64).T)
    centered = matrix - float(np.mean(matrix))
    return centered / (float(np.std(centered)) + EPS)


def _spectral_embedding(matrix: np.ndarray, dim: int) -> tuple[np.ndarray, np.ndarray]:
    values, vectors = np.linalg.eigh(matrix)
    order = np.argsort(-np.abs(values), kind="stable")
    k = max(1, min(int(dim), matrix.shape[0]))
    emb = vectors[:, order[:k]] * np.sqrt(np.abs(values[order[:k]]) + EPS)
    return emb, values[order[:k]]


def _spectral_cost_matrix(F: np.ndarray, D: np.ndarray, p: np.ndarray, dim: int = 4) -> np.ndarray:
    f_emb, _ = _spectral_embedding(_standardize_symmetric(F), dim)
    d_emb, _ = _spectral_embedding(_standardize_symmetric(D), dim)
    cross = f_emb.T @ p @ d_emb
    u, _, vt = np.linalg.svd(cross, full_matrices=False)
    rotation = u @ vt
    aligned_d = d_emb @ rotation
    cost = np.sum((f_emb[:, None, :] - aligned_d[None, :, :]) ** 2, axis=2)
    return cost / (float(np.mean(cost)) + EPS)


def _qap_surrogate(F: np.ndarray, D: np.ndarray, p: np.ndarray) -> float:
    return float(np.sum(F * (p @ D @ p.T)))


def _faq_soft_assignment(
    F: np.ndarray,
    D: np.ndarray,
    n_iters: int = 8,
) -> np.ndarray:
    f = _standardize_symmetric(F)
    d = _standardize_symmetric(D)
    n = int(F.shape[0])
    p = np.full((n, n), 1.0 / max(n, 1), dtype=np.float64)
    for _ in range(int(n_iters)):
        gradient = f @ p @ d.T + f.T @ p @ d
        row_ind, col_ind = linear_sum_assignment(gradient)
        q = np.zeros_like(p)
        q[row_ind, col_ind] = 1.0
        direction = q - p
        direction_gradient = float(np.sum(gradient * direction))
        quadratic = _qap_surrogate(f, d, direction)
        if quadratic > EPS:
            alpha = float(np.clip(-direction_gradient / (2.0 * quadratic), 0.0, 1.0))
        else:
            alpha = 1.0 if direction_gradient < 0.0 else 0.0
        p = p + alpha * direction
    return p.astype(np.float64)


def _entropy_rows(p: np.ndarray) -> np.ndarray:
    denom = np.log(max(int(p.shape[1]), 2))
    return -np.sum(p * np.log(np.maximum(p, EPS)), axis=1) / denom


def _row_margin(p: np.ndarray) -> np.ndarray:
    if p.shape[1] <= 1:
        return np.ones(p.shape[0], dtype=np.float64)
    part = np.partition(p, kth=p.shape[1] - 2, axis=1)
    return part[:, -1] - part[:, -2]


def build_relaxation_feature_cache(
    instance: QAPInstance,
    spectral_dim: int = 4,
    faq_iters: int = 8,
) -> RelaxationFeatureCache:
    F = np.asarray(instance.F, dtype=np.float64)
    D = np.asarray(instance.D, dtype=np.float64)
    faq_p = _faq_soft_assignment(F, D, n_iters=int(faq_iters))
    spectral_cost = _spectral_cost_matrix(F, D, faq_p, dim=int(spectral_dim))
    return RelaxationFeatureCache(
        spectral_cost=spectral_cost,
        faq_p=faq_p,
        faq_row_entropy=_entropy_rows(faq_p),
        faq_col_entropy=_entropy_rows(faq_p.T),
        faq_row_margin=_row_margin(faq_p),
        faq_perm_score_scale=float(np.mean(np.abs(faq_p))) + EPS,
    )


def build_relaxation_swap_features(
    instance: QAPInstance,
    perm: np.ndarray,
    pairs: np.ndarray,
    relaxation_cache: RelaxationFeatureCache | None = None,
) -> np.ndarray:
    perm = np.asarray(perm, dtype=np.int64)
    pairs = np.asarray(pairs, dtype=np.int64)
    if len(pairs) == 0:
        return np.zeros((0, len(RELAXATION_FEATURE_NAMES)), dtype=np.float32)
    cache = relaxation_cache if relaxation_cache is not None else build_relaxation_feature_cache(instance)
    i = pairs[:, 0].astype(np.int64)
    j = pairs[:, 1].astype(np.int64)
    pi_i = perm[i]
    pi_j = perm[j]
    spectral = cache.spectral_cost
    p = cache.faq_p
    spec_i_old = spectral[i, pi_i]
    spec_j_old = spectral[j, pi_j]
    spec_i_swap = spectral[i, pi_j]
    spec_j_swap = spectral[j, pi_i]
    faq_i_current = p[i, pi_i]
    faq_j_current = p[j, pi_j]
    faq_i_swap = p[i, pi_j]
    faq_j_swap = p[j, pi_i]
    faq_current = faq_i_current + faq_j_current
    faq_swap = faq_i_swap + faq_j_swap
    perm_score_current = float(np.sum(p[np.arange(instance.n), perm]) * instance.n / max(instance.n, 1))
    faq_perm_delta = (faq_i_swap + faq_j_swap) - (faq_i_current + faq_j_current)

    features = np.zeros((len(pairs), len(RELAXATION_FEATURE_NAMES)), dtype=np.float32)
    features[:, 0] = spec_i_old + spec_j_old
    features[:, 1] = spec_i_swap + spec_j_swap
    features[:, 2] = features[:, 1] - features[:, 0]
    features[:, 3] = spec_i_old
    features[:, 4] = spec_i_swap
    features[:, 5] = spec_j_old
    features[:, 6] = spec_j_swap
    features[:, 7] = faq_current
    features[:, 8] = faq_swap
    features[:, 9] = faq_swap - faq_current
    features[:, 10] = faq_i_current
    features[:, 11] = faq_i_swap
    features[:, 12] = faq_j_current
    features[:, 13] = faq_j_swap
    features[:, 14] = cache.faq_row_entropy[i]
    features[:, 15] = cache.faq_row_entropy[j]
    features[:, 16] = cache.faq_row_margin[i]
    features[:, 17] = cache.faq_row_margin[j]
    features[:, 18] = cache.faq_col_entropy[pi_i]
    features[:, 19] = cache.faq_col_entropy[pi_j]
    features[:, 20] = perm_score_current
    features[:, 21] = faq_perm_delta
    return features


def build_swap_features_for_fields(
    instance: QAPInstance,
    perm: np.ndarray,
    pairs: np.ndarray,
    feature_fields: list[str],
    feature_cache: SwapFeatureCache | None = None,
    relaxation_cache: RelaxationFeatureCache | None = None,
) -> np.ndarray:
    base_index = {field: idx for idx, field in enumerate(FEATURE_FIELDS)}
    relaxation_index = {field: idx for idx, field in enumerate(RELAXATION_FEATURE_FIELDS)}
    base_features = None
    relaxation_features = None
    columns = []
    for field in feature_fields:
        if field in base_index:
            if base_features is None:
                base_features = build_swap_features(instance, perm, pairs, feature_cache=feature_cache)
            columns.append(base_features[:, base_index[field]])
        elif field in relaxation_index:
            if relaxation_features is None:
                relaxation_features = build_relaxation_swap_features(
                    instance=instance,
                    perm=perm,
                    pairs=pairs,
                    relaxation_cache=relaxation_cache,
                )
            columns.append(relaxation_features[:, relaxation_index[field]])
        else:
            raise ValueError(f"unsupported relaxation feature field: {field}")
    if not columns:
        return np.zeros((len(pairs), 0), dtype=np.float32)
    return np.stack(columns, axis=1).astype(np.float32)
