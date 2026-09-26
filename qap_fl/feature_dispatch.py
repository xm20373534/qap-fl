from __future__ import annotations

import numpy as np

from .features import FEATURE_FIELDS, SwapFeatureCache, build_swap_features
from .global_features import GLOBAL_FEATURE_FIELDS, build_global_swap_features
from .qap import QAPInstance
from .relaxation_features import RELAXATION_FEATURE_FIELDS, RelaxationFeatureCache, build_relaxation_swap_features


def build_swap_features_for_fields(
    instance: QAPInstance,
    perm: np.ndarray,
    pairs: np.ndarray,
    feature_fields: list[str],
    feature_cache: SwapFeatureCache | None = None,
    relaxation_cache: RelaxationFeatureCache | None = None,
) -> np.ndarray:
    base_index = {field: idx for idx, field in enumerate(FEATURE_FIELDS)}
    global_index = {field: idx for idx, field in enumerate(GLOBAL_FEATURE_FIELDS)}
    relaxation_index = {field: idx for idx, field in enumerate(RELAXATION_FEATURE_FIELDS)}
    base_features = None
    global_features = None
    relaxation_features = None
    columns = []
    for field in feature_fields:
        if field in base_index:
            if base_features is None:
                base_features = build_swap_features(instance, perm, pairs, feature_cache=feature_cache)
            columns.append(base_features[:, base_index[field]])
        elif field in global_index:
            if global_features is None:
                global_features = build_global_swap_features(instance, perm, pairs, feature_cache=feature_cache)
            columns.append(global_features[:, global_index[field]])
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
            raise ValueError(f"unsupported swap feature field: {field}")
    if not columns:
        return np.zeros((len(pairs), 0), dtype=np.float32)
    return np.stack(columns, axis=1).astype(np.float32)
