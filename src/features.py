from __future__ import annotations

import numpy as np


RELAXATION_PAIR_FEATURE_NAMES = (
    "faq_assignment",
    "spectral_compatibility",
    "faq_row_entropy",
    "faq_column_entropy",
    "faq_row_margin",
    "faq_column_margin",
    "strength_centrality_product",
    "strength_centrality_rank_match",
)

CONDITIONAL_EXPECTATION_FEATURE_NAME = "negative_fixed_assignment_mean_cost_z"


def normalize_matrix(M: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    M = np.asarray(M, dtype=np.float64)
    scale = np.max(np.abs(M))
    if scale < eps:
        return np.zeros_like(M, dtype=np.float32)
    return (M / scale).astype(np.float32)


def _row_features(M: np.ndarray) -> np.ndarray:
    M = normalize_matrix(M)
    feats = np.stack(
        [
            M.sum(axis=1),
            M.mean(axis=1),
            M.std(axis=1),
            M.max(axis=1),
        ],
        axis=1,
    )
    return feats.astype(np.float32)


def facility_features(F: np.ndarray) -> np.ndarray:
    return _row_features(F)


def location_features(D: np.ndarray) -> np.ndarray:
    return _row_features(D)


def build_features(F: np.ndarray, D: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return facility_features(F), location_features(D)


def _average_rank(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    less = (values[:, None] > values[None, :]).sum(axis=1)
    equal = (values[:, None] == values[None, :]).sum(axis=1)
    return ((less + 0.5 * equal - 0.5) / max(len(values), 1)).astype(np.float64)


def _enhanced_node_features(M: np.ndarray) -> np.ndarray:
    X = normalize_matrix(M).astype(np.float64)
    n = X.shape[0]

    def summaries(rows: np.ndarray) -> np.ndarray:
        return np.stack(
            [
                rows.sum(axis=1),
                rows.mean(axis=1),
                rows.std(axis=1),
                rows.min(axis=1),
                rows.max(axis=1),
                np.abs(rows).sum(axis=1),
                np.sqrt(np.square(rows).sum(axis=1)),
                *[np.quantile(rows, q, axis=1) for q in (0.10, 0.25, 0.50, 0.75, 0.90)],
                (rows > 0).mean(axis=1),
                (np.abs(rows) > 1e-12).mean(axis=1),
            ],
            axis=1,
        )

    row = summaries(X)
    column = summaries(X.T)
    rank_values = np.stack(
        [
            _average_rank(row[:, 0]),
            _average_rank(row[:, 2]),
            _average_rank(row[:, 5]),
            _average_rank(column[:, 0]),
            _average_rank(column[:, 2]),
            _average_rank(column[:, 5]),
        ],
        axis=1,
    )
    row_strength = row[:, 5]
    column_strength = column[:, 5]
    two_hop = np.stack(
        [
            X @ row_strength / max(n, 1),
            X.T @ column_strength / max(n, 1),
            np.abs(X) @ row_strength / max(n, 1),
            np.abs(X.T) @ column_strength / max(n, 1),
        ],
        axis=1,
    )
    continuous = np.concatenate([row, column, two_hop], axis=1)
    center = continuous.mean(axis=0, keepdims=True)
    scale = continuous.std(axis=0, keepdims=True)
    continuous = (continuous - center) / np.maximum(scale, 1e-6)
    log_size = np.full((n, 1), np.log1p(n) / np.log1p(256.0), dtype=np.float64)
    return np.concatenate([continuous, rank_values, log_size], axis=1).astype(np.float32)


def build_enhanced_features(F: np.ndarray, D: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return 39-dimensional permutation-equivariant node signatures."""
    return _enhanced_node_features(F), _enhanced_node_features(D)


def _standardize_symmetric(matrix: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    symmetric = 0.5 * (matrix + matrix.T)
    return (symmetric - symmetric.mean()) / max(float(symmetric.std()), eps)


def _standardize_feature(values: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return (values - values.mean()) / max(float(values.std()), eps)


def _entropy_rows(probabilities: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    denominator = np.log(max(int(probabilities.shape[1]), 2))
    return -np.sum(probabilities * np.log(np.maximum(probabilities, eps)), axis=1) / denominator


def _row_margin(probabilities: np.ndarray) -> np.ndarray:
    if probabilities.shape[1] <= 1:
        return np.ones(probabilities.shape[0], dtype=np.float64)
    partitioned = np.partition(probabilities, kth=probabilities.shape[1] - 2, axis=1)
    return partitioned[:, -1] - partitioned[:, -2]


def _faq_soft_assignment(F: np.ndarray, D: np.ndarray, iterations: int) -> np.ndarray:
    from scipy.optimize import linear_sum_assignment

    flow = _standardize_symmetric(F)
    distance = _standardize_symmetric(D)
    n = int(flow.shape[0])
    assignment = np.full((n, n), 1.0 / max(n, 1), dtype=np.float64)
    for _ in range(int(iterations)):
        gradient = flow @ assignment @ distance.T + flow.T @ assignment @ distance
        rows, columns = linear_sum_assignment(gradient)
        direction_vertex = np.zeros_like(assignment)
        direction_vertex[rows, columns] = 1.0
        direction = direction_vertex - assignment
        directional_gradient = float(np.sum(gradient * direction))
        quadratic = float(np.sum(flow * (direction @ distance @ direction.T)))
        if quadratic > 1e-12:
            step = float(np.clip(-directional_gradient / (2.0 * quadratic), 0.0, 1.0))
        else:
            step = 1.0 if directional_gradient < 0.0 else 0.0
        assignment = assignment + step * direction
    return assignment


def _spectral_compatibility(
    F: np.ndarray,
    D: np.ndarray,
    soft_assignment: np.ndarray,
    dimension: int,
) -> np.ndarray:
    def embedding(matrix: np.ndarray) -> np.ndarray:
        values, vectors = np.linalg.eigh(_standardize_symmetric(matrix))
        order = np.argsort(-np.abs(values), kind="stable")[: max(1, min(int(dimension), len(values)))]
        return vectors[:, order] * np.sqrt(np.abs(values[order]) + 1e-12)

    facility = embedding(F)
    location = embedding(D)
    cross = facility.T @ soft_assignment @ location
    left, _, right = np.linalg.svd(cross, full_matrices=False)
    # Orthogonal Procrustes: maximize tr(cross @ rotation), so R = V U^T.
    # This orientation also cancels independent eigenvector sign choices.
    aligned_location = location @ (right.T @ left.T)
    squared_distance = np.square(facility[:, None, :] - aligned_location[None, :, :]).sum(axis=2)
    return -_standardize_feature(squared_distance)


def build_relaxation_pair_features(
    F: np.ndarray,
    D: np.ndarray,
    faq_iterations: int = 8,
    spectral_dimension: int = 4,
) -> np.ndarray:
    """Build static, permutation-equivariant facility-location compatibility features."""
    F = np.asarray(F, dtype=np.float64)
    D = np.asarray(D, dtype=np.float64)
    if F.ndim != 2 or F.shape[0] != F.shape[1] or D.shape != F.shape:
        raise ValueError("F and D must be matching square matrices")
    n = F.shape[0]
    faq = _faq_soft_assignment(F, D, faq_iterations)
    spectral = _spectral_compatibility(F, D, faq, spectral_dimension)

    row_entropy = _standardize_feature(_entropy_rows(faq))
    column_entropy = _standardize_feature(_entropy_rows(faq.T))
    row_margin = _standardize_feature(_row_margin(faq))
    column_margin = _standardize_feature(_row_margin(faq.T))

    facility_strength = np.abs(F).sum(axis=1) + np.abs(F).sum(axis=0)
    # Small total distance means a central location, hence the minus sign.
    location_centrality = -(np.abs(D).sum(axis=1) + np.abs(D).sum(axis=0))
    facility_z = _standardize_feature(facility_strength)
    location_z = _standardize_feature(location_centrality)
    facility_rank = _average_rank(facility_strength)
    location_rank = _average_rank(location_centrality)

    pair_features = np.stack(
        [
            _standardize_feature(faq),
            spectral,
            np.broadcast_to(row_entropy[:, None], (n, n)),
            np.broadcast_to(column_entropy[None, :], (n, n)),
            np.broadcast_to(row_margin[:, None], (n, n)),
            np.broadcast_to(column_margin[None, :], (n, n)),
            facility_z[:, None] * location_z[None, :],
            -np.abs(facility_rank[:, None] - location_rank[None, :]),
        ],
        axis=2,
    ).astype(np.float32)
    if not np.all(np.isfinite(pair_features)):
        raise ValueError("relaxation pair features contain non-finite values")
    return pair_features


def conditional_assignment_mean_costs(F: np.ndarray, D: np.ndarray) -> np.ndarray:
    """Return E[Q(pi) | pi(x)=y] for every facility-location pair in O(n^2)."""
    F = np.asarray(F, dtype=np.float64)
    D = np.asarray(D, dtype=np.float64)
    if F.ndim != 2 or F.shape[0] != F.shape[1] or D.shape != F.shape:
        raise ValueError("F and D must be matching square matrices")
    n = int(F.shape[0])
    if n < 3:
        raise ValueError("conditional assignment means require n >= 3")

    def components(matrix: np.ndarray):
        diagonal = np.diag(matrix)
        row_off = matrix.sum(axis=1) - diagonal
        column_off = matrix.sum(axis=0) - diagonal
        total_off = float(matrix.sum() - diagonal.sum())
        remaining_off = total_off - row_off - column_off
        remaining_diagonal = float(diagonal.sum()) - diagonal
        return diagonal, row_off, column_off, remaining_off, remaining_diagonal

    f_diag, f_row, f_column, f_remaining, f_diag_remaining = components(F)
    d_diag, d_row, d_column, d_remaining, d_diag_remaining = components(D)
    means = (
        np.outer(f_remaining, d_remaining) / ((n - 1) * (n - 2))
        + np.outer(f_diag_remaining, d_diag_remaining) / (n - 1)
        + np.outer(f_column, d_column) / (n - 1)
        + np.outer(f_row, d_row) / (n - 1)
        + np.outer(f_diag, d_diag)
    )
    if not np.all(np.isfinite(means)):
        raise ValueError("conditional assignment means contain non-finite values")
    return means


def build_conditional_mean_pair_feature(F: np.ndarray, D: np.ndarray) -> np.ndarray:
    """Standardized score where larger values mean a lower conditional mean cost."""
    means = conditional_assignment_mean_costs(F, D)
    feature = -_standardize_feature(means)
    return feature[:, :, None].astype(np.float32)


def _check() -> None:
    rng = np.random.default_rng(0)
    F = rng.uniform(size=(5, 5))
    D = rng.uniform(size=(5, 5))
    fac_x, loc_x = build_features(F, D)
    if fac_x.shape != (5, 4) or loc_x.shape != (5, 4):
        raise AssertionError(f"unexpected feature shapes: {fac_x.shape}, {loc_x.shape}")
    if not np.isfinite(fac_x).all() or not np.isfinite(loc_x).all():
        raise AssertionError("features contain non-finite values")
    enhanced_fac, enhanced_loc = build_enhanced_features(F, D)
    if enhanced_fac.shape != (5, 39) or enhanced_loc.shape != (5, 39):
        raise AssertionError(f"unexpected enhanced feature shapes: {enhanced_fac.shape}, {enhanced_loc.shape}")
    pair_features = build_relaxation_pair_features(F, D)
    if pair_features.shape != (5, 5, len(RELAXATION_PAIR_FEATURE_NAMES)):
        raise AssertionError(f"unexpected pair feature shape: {pair_features.shape}")
    facility_order = np.array([2, 4, 0, 1, 3])
    location_order = np.array([1, 3, 4, 0, 2])
    permuted = build_relaxation_pair_features(
        F[np.ix_(facility_order, facility_order)],
        D[np.ix_(location_order, location_order)],
    )
    expected = pair_features[np.ix_(facility_order, location_order)]
    if not np.allclose(permuted, expected, atol=2e-4, rtol=2e-4):
        raise AssertionError("pair features are not permutation equivariant")
    conditional = build_conditional_mean_pair_feature(F, D)
    if conditional.shape != (5, 5, 1) or not np.isfinite(conditional).all():
        raise AssertionError(f"unexpected conditional feature shape: {conditional.shape}")
    conditional_permuted = build_conditional_mean_pair_feature(
        F[np.ix_(facility_order, facility_order)],
        D[np.ix_(location_order, location_order)],
    )
    conditional_expected = conditional[np.ix_(facility_order, location_order)]
    if not np.allclose(conditional_permuted, conditional_expected, atol=1e-6, rtol=1e-6):
        raise AssertionError("conditional feature is not permutation equivariant")
    print("features checks passed")


if __name__ == "__main__":
    _check()
