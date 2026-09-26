from __future__ import annotations

import numpy as np

from .hybrid_gate import (
    HYBRID_GATE_FEATURE_FIELDS,
    build_hybrid_gate_features,
)
from .relaxation_features_v2 import (
    RELAXATION_FEATURE_NAMES,
    build_relaxation_swap_features,
)
from .qap import QAPInstance


RELAXATION_GATE_FEATURE_NAMES = [
    "relax_faq_best",
    "relax_faq_model",
    "relax_faq_gap",
    "relax_spectral_best",
    "relax_spectral_model",
    "relax_spectral_gap",
    "relax_model_row_margin_min",
    "relax_best_row_margin_min",
    "relax_model_row_entropy_mean",
    "relax_best_row_entropy_mean",
    "relax_faq_perm_alignment",
    "relax_faq_perm_score_delta",
]
RELAXATION_GATE_FEATURE_FIELDS = [
    f"feature_{name}" for name in RELAXATION_GATE_FEATURE_NAMES
]
HYBRID_GATE_RELAXATION_FEATURE_FIELDS = (
    HYBRID_GATE_FEATURE_FIELDS + RELAXATION_GATE_FEATURE_FIELDS
)


_RELAXATION_INDEX = {
    name: index for index, name in enumerate(RELAXATION_FEATURE_NAMES)
}


def _pair_values(
    instance: QAPInstance,
    perm: np.ndarray,
    pair: tuple[int, int],
    relaxation_cache,
) -> np.ndarray:
    pairs = np.asarray([pair], dtype=np.int64)
    return build_relaxation_swap_features(
        instance=instance,
        perm=perm,
        pairs=pairs,
        relaxation_cache=relaxation_cache,
    )[0].astype(np.float64)


def _pair_faq(values: np.ndarray) -> float:
    return float(values[_RELAXATION_INDEX["faq_delta"]])


def _pair_spectral(values: np.ndarray) -> float:
    return float(values[_RELAXATION_INDEX["spectral_delta"]])


def _pair_row_margin_min(values: np.ndarray) -> float:
    return float(
        min(
            values[_RELAXATION_INDEX["faq_row_margin_i"]],
            values[_RELAXATION_INDEX["faq_row_margin_j"]],
        )
    )


def _pair_row_entropy_mean(values: np.ndarray) -> float:
    return float(
        0.5
        * (
            values[_RELAXATION_INDEX["faq_row_entropy_i"]]
            + values[_RELAXATION_INDEX["faq_row_entropy_j"]]
        )
    )


def build_hybrid_gate_relaxation_features(
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
    instance: QAPInstance,
    perm: np.ndarray,
    best_pair: tuple[int, int],
    model_pair: tuple[int, int],
    relaxation_cache,
) -> np.ndarray:
    base = build_hybrid_gate_features(
        n=n,
        iteration=iteration,
        max_iters=max_iters,
        pool_size=pool_size,
        total_pairs=total_pairs,
        n_improving=n_improving,
        selected_count=selected_count,
        best_delta=best_delta,
        model_delta=model_delta,
        best_score=best_score,
        model_score=model_score,
        best_combined=best_combined,
        model_combined=model_combined,
        model_prob=model_prob,
        model_delta_rank=model_delta_rank,
        selected_deltas=selected_deltas,
        selected_scores=selected_scores,
        selected_combined=selected_combined,
    ).astype(np.float64)

    best_values = _pair_values(instance, perm, best_pair, relaxation_cache)
    model_values = _pair_values(instance, perm, model_pair, relaxation_cache)
    best_faq = _pair_faq(best_values)
    model_faq = _pair_faq(model_values)
    best_spectral = _pair_spectral(best_values)
    model_spectral = _pair_spectral(model_values)
    best_margin = _pair_row_margin_min(best_values)
    model_margin = _pair_row_margin_min(model_values)
    best_entropy = _pair_row_entropy_mean(best_values)
    model_entropy = _pair_row_entropy_mean(model_values)
    global_alignment = float(
        best_values[_RELAXATION_INDEX["faq_perm_score_current"]]
    )
    relaxation = np.asarray(
        [
            best_faq,
            model_faq,
            model_faq - best_faq,
            best_spectral,
            model_spectral,
            model_spectral - best_spectral,
            model_margin,
            best_margin,
            model_entropy,
            best_entropy,
            global_alignment,
            float(model_values[_RELAXATION_INDEX["faq_perm_score_delta"]]),
        ],
        dtype=np.float64,
    )
    return np.concatenate([base, relaxation]).astype(np.float32)


def build_hybrid_gate_relaxation_feature_row(**kwargs: object) -> dict[str, float]:
    values = build_hybrid_gate_relaxation_features(**kwargs)  # type: ignore[arg-type]
    return {
        field: float(value)
        for field, value in zip(HYBRID_GATE_RELAXATION_FEATURE_FIELDS, values)
    }
