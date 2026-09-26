"""Lightweight inference for the D29 cost-aware macro-action controller."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .qap import QAPInstance, compute_cost


EPS = 1e-12


def _matrix_features(prefix: str, matrix: np.ndarray) -> dict[str, float]:
    values = np.asarray(matrix, dtype=np.float64)
    absolute = np.abs(values)
    total = max(float(np.sum(absolute)), EPS)
    row_activity = np.sum(absolute, axis=1)
    return {
        f"state_{prefix}_cv": float(
            np.std(values) / max(abs(float(np.mean(values))), float(np.std(values)), EPS)
        ),
        f"state_{prefix}_zero_fraction": float(np.mean(absolute <= EPS)),
        f"state_{prefix}_symmetry_error": float(
            np.linalg.norm(values - values.T) / max(float(np.linalg.norm(values)), EPS)
        ),
        f"state_{prefix}_diag_fraction": float(np.sum(np.abs(np.diag(values))) / total),
        f"state_{prefix}_row_cv": float(
            np.std(row_activity) / max(float(np.mean(row_activity)), EPS)
        ),
    }


def _static_instance_features(instance: QAPInstance) -> tuple[dict[str, float], float]:
    scale = (
        max(float(np.mean(np.abs(instance.F))), EPS)
        * max(float(np.mean(np.abs(instance.D))), EPS)
        * max(instance.n * instance.n, 1)
    )
    features = {}
    features.update(_matrix_features("flow", instance.F))
    features.update(_matrix_features("distance", instance.D))
    return features, scale


def build_deployment_features(
    instance: QAPInstance,
    perm: np.ndarray,
    *,
    current_cost: float | None = None,
    static_features: dict[str, float] | None = None,
    cost_scale: float | None = None,
) -> dict[str, float]:
    current = np.asarray(perm, dtype=np.int64)
    assigned_distance = instance.D[np.ix_(current, current)]
    interaction = np.abs(instance.F * assigned_distance).astype(np.float64)
    flat = interaction.reshape(-1)
    total = max(float(np.sum(flat)), EPS)
    node_activity = interaction.sum(axis=0) + interaction.sum(axis=1) - np.diag(interaction)
    flow_strength = np.sum(np.abs(instance.F), axis=1)
    distance_strength = np.sum(np.abs(instance.D), axis=1)[current]
    alignment = (
        float(np.corrcoef(flow_strength, distance_strength)[0, 1])
        if np.std(flow_strength) > EPS and np.std(distance_strength) > EPS
        else 0.0
    )
    if static_features is None or cost_scale is None:
        static_features, cost_scale = _static_instance_features(instance)
    features = {
        "state_cost_scaled": float(
            compute_cost(current, instance.F, instance.D) if current_cost is None else current_cost
        )
        / float(cost_scale),
        "state_interaction_cv": float(np.std(flat) / max(float(np.mean(flat)), EPS)),
        "state_interaction_zero_fraction": float(np.mean(flat <= EPS)),
        "state_interaction_top10_fraction": float(
            np.sum(np.sort(flat)[-max(1, len(flat) // 10) :]) / total
        ),
        "state_node_activity_cv": float(
            np.std(node_activity) / max(float(np.mean(node_activity)), EPS)
        ),
        "state_node_top2_fraction": float(
            np.sum(np.sort(node_activity)[-min(2, instance.n) :])
            / max(float(np.sum(node_activity)), EPS)
        ),
        "state_strength_alignment": alignment,
    }
    features.update(static_features)
    return features


@dataclass
class RidgeMacroController:
    coef: np.ndarray
    x_mean: np.ndarray
    x_std: np.ndarray
    y_mean: np.ndarray
    feature_names: tuple[str, ...]
    actions: tuple[str, ...]
    _instance_cache: dict[int, tuple[dict[str, float], float]] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )

    @classmethod
    def load(cls, path: str | Path) -> "RidgeMacroController":
        with np.load(Path(path), allow_pickle=False) as data:
            return cls(
                coef=np.asarray(data["coef"], dtype=np.float64),
                x_mean=np.asarray(data["x_mean"], dtype=np.float64),
                x_std=np.asarray(data["x_std"], dtype=np.float64),
                y_mean=np.asarray(data["y_mean"], dtype=np.float64),
                feature_names=tuple(str(value) for value in data["feature_names"].tolist()),
                actions=tuple(str(value) for value in data["actions"].tolist()),
            )

    def predict_scores(
        self,
        instance: QAPInstance,
        perm: np.ndarray,
        *,
        current_cost: float | None = None,
    ) -> np.ndarray:
        cache_key = id(instance)
        cached = self._instance_cache.get(cache_key)
        if cached is None:
            cached = _static_instance_features(instance)
            self._instance_cache[cache_key] = cached
        values = build_deployment_features(
            instance,
            perm,
            current_cost=current_cost,
            static_features=cached[0],
            cost_scale=cached[1],
        )
        x = np.asarray([values[name] for name in self.feature_names], dtype=np.float64)
        return ((x - self.x_mean) / self.x_std) @ self.coef + self.y_mean

    def predict_action(
        self,
        instance: QAPInstance,
        perm: np.ndarray,
        *,
        current_cost: float | None = None,
    ) -> str:
        scores = self.predict_scores(instance, perm, current_cost=current_cost)
        return self.actions[int(np.argmax(scores))]
