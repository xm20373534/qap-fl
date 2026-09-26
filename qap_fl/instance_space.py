from __future__ import annotations

import numpy as np

from src.features import conditional_assignment_mean_costs


EPS = 1e-12


def _zero_minimum(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    n = matrix.shape[0]
    diagonal = np.eye(n, dtype=bool)
    output = matrix.copy()
    output[diagonal] -= float(np.min(matrix[diagonal]))
    if n > 1:
        output[~diagonal] -= float(np.min(matrix[~diagonal]))
    return output


def _unit_range(matrix: np.ndarray) -> np.ndarray:
    matrix = _zero_minimum(matrix)
    scale = float(np.max(np.abs(matrix)))
    return matrix / scale if scale > EPS else np.zeros_like(matrix)


def _moments(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    centered = values - values.mean()
    scale = float(np.sqrt(np.mean(np.square(centered))))
    if scale <= EPS:
        return 0.0, 0.0
    standardized = centered / scale
    skewness = float(np.mean(standardized ** 3))
    excess_kurtosis = float(np.mean(standardized ** 4) - 3.0)
    return skewness, excess_kurtosis


def _dominance(matrix: np.ndarray) -> float:
    values = _zero_minimum(matrix).reshape(-1)
    mean = float(np.mean(values))
    if mean <= EPS:
        return 0.0
    cv = float(np.std(values) / mean)
    n = int(matrix.shape[0])
    high = np.zeros(n * n, dtype=np.float64)
    high[0] = 1.0
    low = (np.ones((n, n), dtype=np.float64) - np.eye(n)).reshape(-1)
    cv_high = float(np.std(high) / np.mean(high))
    cv_low = float(np.std(low) / np.mean(low))
    denominator = cv_high - cv_low
    if abs(denominator) <= EPS:
        return 0.0
    return float(np.clip((cv_high - cv) / denominator, 0.0, 1.0))


def _beta_parameters(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    mean = float(np.mean(values))
    variance = float(np.var(values))
    if variance <= EPS or mean <= EPS or mean >= 1.0 - EPS:
        return 0.0, 0.0
    common = mean * (1.0 - mean) / variance - 1.0
    if common <= 0.0:
        return 0.0, 0.0
    alpha = mean * common
    beta = (1.0 - mean) * common
    return (
        float(np.arctan(alpha) / (np.pi / 2.0)),
        float(np.arctan(beta) / (np.pi / 2.0)),
    )


def _triangle_satisfaction(matrix: np.ndarray) -> float:
    values = _unit_range(matrix)
    n = int(values.shape[0])
    if n < 3:
        return 1.0
    valid = np.ones((n, n), dtype=bool)
    np.fill_diagonal(valid, False)
    satisfied = 0
    total = 0
    for middle in range(n):
        mask = valid.copy()
        mask[middle, :] = False
        mask[:, middle] = False
        right = values[:, middle, None] + values[middle, None, :]
        satisfied += int(np.count_nonzero((values <= right + 1e-12) & mask))
        total += int(np.count_nonzero(mask))
    return float(satisfied / max(total, 1))


def _matrix_features(matrix: np.ndarray, prefix: str) -> dict[str, float]:
    matrix = np.asarray(matrix, dtype=np.float64)
    n = int(matrix.shape[0])
    zero_minimum = _zero_minimum(matrix)
    unit = _unit_range(matrix)
    total = float(np.sum(zero_minimum))
    diagonal_total = float(np.trace(zero_minimum))
    mean = float(np.mean(matrix))
    standard_deviation = float(np.std(matrix))
    outlier = (
        float(np.mean(np.abs(matrix - mean) > 3.0 * standard_deviation))
        if standard_deviation > EPS else 0.0
    )
    skewness, kurtosis = _moments(matrix)
    alpha, beta = _beta_parameters(unit)
    off_diagonal = ~np.eye(n, dtype=bool)
    symmetry = (
        float(np.mean(np.isclose(matrix[off_diagonal], matrix.T[off_diagonal], rtol=1e-9, atol=1e-12)))
        if n > 1 else 1.0
    )
    return {
        f"{prefix}_normalized_mean": float(np.mean(unit)),
        f"{prefix}_trace_proportion": diagonal_total / total if abs(total) > EPS else 0.0,
        f"{prefix}_sparsity": float(np.mean(np.abs(zero_minimum) <= EPS)),
        f"{prefix}_dominance": _dominance(matrix),
        f"{prefix}_outliers": outlier,
        f"{prefix}_skewness": float(np.arctan(skewness / 2.0) / (np.pi / 2.0)),
        f"{prefix}_kurtosis": float(np.arctan(kurtosis / 10.0) / (np.pi / 2.0)),
        f"{prefix}_beta_alpha": alpha,
        f"{prefix}_beta_beta": beta,
        f"{prefix}_symmetry": symmetry,
        f"{prefix}_triangle_satisfaction": _triangle_satisfaction(matrix),
    }


def _distribution_similarity(flow: np.ndarray, distance: np.ndarray) -> float:
    flow_values = _unit_range(flow).reshape(-1)
    distance_values = 1.0 - _unit_range(distance).reshape(-1)
    flow_values = np.sort(flow_values)
    distance_values = np.sort(distance_values)
    return float(np.clip(1.0 - np.mean(np.abs(flow_values - distance_values)), 0.0, 1.0))


def build_instance_features(flow: np.ndarray, distance: np.ndarray) -> dict[str, float]:
    flow = np.asarray(flow, dtype=np.float64)
    distance = np.asarray(distance, dtype=np.float64)
    if flow.ndim != 2 or flow.shape[0] != flow.shape[1] or distance.shape != flow.shape:
        raise ValueError("flow and distance must be matching square matrices")
    n = int(flow.shape[0])
    output = {"log_size": float(np.log1p(n) / np.log1p(256.0))}
    output.update(_matrix_features(flow, "flow"))
    output.update(_matrix_features(distance, "distance"))
    output["least_dominance"] = min(output["flow_dominance"], output["distance_dominance"])
    output["most_dominance"] = max(output["flow_dominance"], output["distance_dominance"])
    output["distribution_similarity"] = _distribution_similarity(flow, distance)
    conditional = conditional_assignment_mean_costs(flow, distance)
    span = float(np.max(conditional) - np.min(conditional))
    output["conditional_contrast"] = (
        float((np.quantile(conditional, 0.9) - np.quantile(conditional, 0.1)) / span)
        if span > EPS else 0.0
    )
    if not all(np.isfinite(value) for value in output.values()):
        raise ValueError("instance features contain non-finite values")
    return output

