from __future__ import annotations

import numpy as np


HYBRID_GATE_FEATURE_NAMES = [
    "n_scaled",
    "iteration_frac",
    "pool_fraction",
    "improving_fraction",
    "selected_fraction",
    "model_prob",
    "model_delta_rank",
    "delta_best_z",
    "delta_model_z",
    "delta_gap_scaled",
    "score_best_z",
    "score_model_z",
    "score_gap_scaled",
    "combined_best_z",
    "combined_model_z",
    "combined_gap_scaled",
]
HYBRID_GATE_FEATURE_FIELDS = [f"feature_{name}" for name in HYBRID_GATE_FEATURE_NAMES]


def _safe_scale(value: float) -> float:
    return float(max(abs(float(value)), 1e-9))


def _std(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    return _safe_scale(float(np.std(values)))


def _zscore(value: float, values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    return (float(value) - float(np.mean(values))) / _std(values)


def _scaled(value: float, values: np.ndarray) -> float:
    return float(value) / _std(values)


def build_hybrid_gate_features(
    *,
    n: int,
    iteration: int,
    max_iters: int,
    pool_size: int,
    total_pairs: int,
    n_improving: int,
    selected_count: int,
    best_delta: float,
    model_delta: float,
    best_score: float,
    model_score: float,
    best_combined: float,
    model_combined: float,
    model_prob: float,
    model_delta_rank: float,
    selected_deltas: np.ndarray,
    selected_scores: np.ndarray,
    selected_combined: np.ndarray,
) -> np.ndarray:
    selected_deltas = np.asarray(selected_deltas, dtype=np.float64)
    selected_scores = np.asarray(selected_scores, dtype=np.float64)
    selected_combined = np.asarray(selected_combined, dtype=np.float64)
    if len(selected_deltas) == 0 or len(selected_scores) == 0 or len(selected_combined) == 0:
        raise ValueError("selected candidate arrays must be non-empty.")

    total_pairs = max(int(total_pairs), 1)
    pool_size = max(int(pool_size), 1)
    n_improving = max(int(n_improving), 1)
    selected_count = max(int(selected_count), 1)
    iteration_frac = float(iteration) / float(max(int(max_iters) - 1, 1))

    return np.asarray(
        [
            float(n) / 100.0,
            iteration_frac,
            float(pool_size) / float(total_pairs),
            float(n_improving) / float(pool_size),
            float(selected_count) / float(pool_size),
            float(model_prob),
            float(model_delta_rank),
            _zscore(best_delta, selected_deltas),
            _zscore(model_delta, selected_deltas),
            _scaled(model_delta - best_delta, selected_deltas),
            _zscore(best_score, selected_scores),
            _zscore(model_score, selected_scores),
            _scaled(model_score - best_score, selected_scores),
            _zscore(best_combined, selected_combined),
            _zscore(model_combined, selected_combined),
            _scaled(model_combined - best_combined, selected_combined),
        ],
        dtype=np.float32,
    )


def build_hybrid_gate_feature_row(**kwargs: object) -> dict[str, float]:
    values = build_hybrid_gate_features(**kwargs)  # type: ignore[arg-type]
    return {field: float(value) for field, value in zip(HYBRID_GATE_FEATURE_FIELDS, values)}
