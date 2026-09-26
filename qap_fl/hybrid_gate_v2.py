from __future__ import annotations

import numpy as np

from .features import EPS, SwapFeatureCache, build_swap_feature_cache
from .global_features import GLOBAL_FEATURE_NAMES, build_global_swap_features
from .hybrid_gate import HYBRID_GATE_FEATURE_FIELDS, build_hybrid_gate_feature_row
from .qap import QAPInstance, compute_cost


STATE_FEATURE_NAMES = [
    "state_current_cost_scaled",
    "state_best_delta_scaled",
    "state_model_delta_scaled",
    "state_delta_loss_scaled",
    "state_delta_mean_scaled",
    "state_delta_std_scaled",
    "state_delta_min_scaled",
    "state_delta_p10_scaled",
    "state_delta_p50_scaled",
    "state_delta_p90_scaled",
    "state_improving_delta_mean_scaled",
    "state_improving_delta_std_scaled",
    "state_score_mean",
    "state_score_std",
    "state_combined_mean",
    "state_combined_std",
    "instance_flow_cv",
    "instance_flow_zero_fraction",
    "instance_distance_cv",
    "instance_distance_zero_fraction",
]

MODEL_GLOBAL_FEATURE_NAMES = [f"model_{name}" for name in GLOBAL_FEATURE_NAMES]
BEST_GLOBAL_FEATURE_NAMES = [f"best_{name}" for name in GLOBAL_FEATURE_NAMES]
GAP_GLOBAL_FEATURE_NAMES = [f"gap_{name}" for name in GLOBAL_FEATURE_NAMES]

HYBRID_ADVANTAGE_V2_FEATURE_NAMES = (
    [field.removeprefix("feature_") for field in HYBRID_GATE_FEATURE_FIELDS]
    + STATE_FEATURE_NAMES
    + MODEL_GLOBAL_FEATURE_NAMES
    + BEST_GLOBAL_FEATURE_NAMES
    + GAP_GLOBAL_FEATURE_NAMES
)
HYBRID_ADVANTAGE_V2_FEATURE_FIELDS = [f"feature_{name}" for name in HYBRID_ADVANTAGE_V2_FEATURE_NAMES]


def _safe_std(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    return max(float(np.std(values)), EPS)


def _safe_mean_abs(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    return max(float(np.mean(np.abs(values))), EPS)


def _offdiag_values(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    mask = ~np.eye(matrix.shape[0], dtype=bool)
    return matrix[mask]


def build_hybrid_advantage_v2_feature_row(
    *,
    instance: QAPInstance,
    perm: np.ndarray,
    pool: np.ndarray,
    scores: np.ndarray,
    deltas: np.ndarray,
    delta_rank: np.ndarray,
    combined: np.ndarray,
    selected_improving: np.ndarray,
    best_idx: int,
    model_idx: int,
    model_prob: float,
    iteration: int,
    max_iters: int,
    feature_cache: SwapFeatureCache | None = None,
) -> dict[str, float]:
    cache = build_swap_feature_cache(instance) if feature_cache is None else feature_cache
    deltas = np.asarray(deltas, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    combined = np.asarray(combined, dtype=np.float64)
    selected_improving = np.asarray(selected_improving, dtype=np.int64)
    pool = np.asarray(pool, dtype=np.int64)
    total_pairs = int(instance.n * (instance.n - 1) / 2)

    row = build_hybrid_gate_feature_row(
        n=instance.n,
        iteration=int(iteration),
        max_iters=int(max_iters),
        pool_size=len(pool),
        total_pairs=total_pairs,
        n_improving=int(np.sum(deltas < -1e-12)),
        selected_count=len(selected_improving),
        best_delta=float(deltas[int(best_idx)]),
        model_delta=float(deltas[int(model_idx)]),
        best_score=float(scores[int(best_idx)]),
        model_score=float(scores[int(model_idx)]),
        best_combined=float(combined[int(best_idx)]),
        model_combined=float(combined[int(model_idx)]),
        model_prob=float(model_prob),
        model_delta_rank=float(delta_rank[int(model_idx)]),
        selected_deltas=deltas[selected_improving],
        selected_scores=scores[selected_improving],
        selected_combined=combined[selected_improving],
    )

    qap_scale = max(float(cache.qap_scale), EPS)
    current_cost = compute_cost(np.asarray(perm, dtype=np.int64), instance.F, instance.D)
    cost_scale = qap_scale * max(float(instance.n), 1.0)
    improving_deltas = deltas[deltas < -1e-12]
    if len(improving_deltas) == 0:
        improving_deltas = np.asarray([0.0], dtype=np.float64)

    F_values = np.abs(_offdiag_values(instance.F))
    D_values = np.abs(_offdiag_values(instance.D))
    F_mean = _safe_mean_abs(F_values)
    D_mean = _safe_mean_abs(D_values)

    state_values = {
        "feature_state_current_cost_scaled": float(current_cost) / max(cost_scale, EPS),
        "feature_state_best_delta_scaled": float(deltas[int(best_idx)]) / qap_scale,
        "feature_state_model_delta_scaled": float(deltas[int(model_idx)]) / qap_scale,
        "feature_state_delta_loss_scaled": float(deltas[int(model_idx)] - deltas[int(best_idx)]) / qap_scale,
        "feature_state_delta_mean_scaled": float(np.mean(deltas)) / qap_scale,
        "feature_state_delta_std_scaled": _safe_std(deltas) / qap_scale,
        "feature_state_delta_min_scaled": float(np.min(deltas)) / qap_scale,
        "feature_state_delta_p10_scaled": float(np.percentile(deltas, 10)) / qap_scale,
        "feature_state_delta_p50_scaled": float(np.percentile(deltas, 50)) / qap_scale,
        "feature_state_delta_p90_scaled": float(np.percentile(deltas, 90)) / qap_scale,
        "feature_state_improving_delta_mean_scaled": float(np.mean(improving_deltas)) / qap_scale,
        "feature_state_improving_delta_std_scaled": _safe_std(improving_deltas) / qap_scale,
        "feature_state_score_mean": float(np.mean(scores)),
        "feature_state_score_std": _safe_std(scores),
        "feature_state_combined_mean": float(np.mean(combined)),
        "feature_state_combined_std": _safe_std(combined),
        "feature_instance_flow_cv": float(np.std(F_values)) / F_mean,
        "feature_instance_flow_zero_fraction": float(np.mean(F_values <= 1e-12)),
        "feature_instance_distance_cv": float(np.std(D_values)) / D_mean,
        "feature_instance_distance_zero_fraction": float(np.mean(D_values <= 1e-12)),
    }
    row.update(state_values)

    global_features = build_global_swap_features(instance, np.asarray(perm, dtype=np.int64), pool, feature_cache=cache)
    best_global = global_features[int(best_idx)]
    model_global = global_features[int(model_idx)]
    for name, best_value, model_value in zip(GLOBAL_FEATURE_NAMES, best_global, model_global):
        row[f"feature_best_{name}"] = float(best_value)
        row[f"feature_model_{name}"] = float(model_value)
        row[f"feature_gap_{name}"] = float(model_value - best_value)
    return row
