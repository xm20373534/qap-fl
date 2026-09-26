from __future__ import annotations

import numpy as np
import torch

from .candidate_proposal import NODE_FEATURE_NAMES, build_node_features
from .features import EPS, SwapFeatureCache, build_swap_feature_cache
from .global_features import build_swap_features_for_fields
from .model import QAPRelationalSwapScorer
from .qap import QAPInstance


RQS_NODE_FEATURE_NAMES = [name for name in NODE_FEATURE_NAMES if name != "assignment_index"]
RQS_EDGE_FEATURE_NAMES = [
    "flow_forward",
    "flow_reverse",
    "distance_forward",
    "distance_reverse",
    "interaction_forward",
    "interaction_reverse",
]


def build_rqs_node_features(
    instance: QAPInstance,
    perm: np.ndarray,
    cache: SwapFeatureCache | None = None,
) -> np.ndarray:
    features = build_node_features(instance, perm, feature_cache=cache)
    keep = [NODE_FEATURE_NAMES.index(name) for name in RQS_NODE_FEATURE_NAMES]
    return features[:, keep].astype(np.float32)


def build_rqs_edge_features(
    instance: QAPInstance,
    perm: np.ndarray,
    cache: SwapFeatureCache | None = None,
) -> np.ndarray:
    cache = build_swap_feature_cache(instance) if cache is None else cache
    perm = np.asarray(perm, dtype=np.int64)
    f_scale = float(np.mean(np.abs(cache.F))) + EPS
    d_scale = float(np.mean(np.abs(cache.D))) + EPS
    flow_forward = cache.F / f_scale
    flow_reverse = cache.F.T / f_scale
    assigned_distance = cache.D[np.ix_(perm, perm)] / d_scale
    distance_reverse = assigned_distance.T
    features = np.stack(
        [
            flow_forward,
            flow_reverse,
            assigned_distance,
            distance_reverse,
            flow_forward * assigned_distance,
            flow_reverse * distance_reverse,
        ],
        axis=-1,
    ).astype(np.float32)
    diagonal = np.arange(instance.n)
    features[diagonal, diagonal] = 0.0
    if not np.all(np.isfinite(features)):
        raise ValueError("RQS edge features contain non-finite values")
    return features


class RelationalSelector:
    def __init__(
        self,
        model: QAPRelationalSwapScorer,
        feature_fields: list[str],
        normalizers: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        self.model = model
        self.feature_fields = list(feature_fields)
        self.normalizers = tuple(np.asarray(value, dtype=np.float32) for value in normalizers)

    def __call__(
        self,
        instance: QAPInstance,
        perm: np.ndarray,
        pairs: np.ndarray,
        feature_cache: SwapFeatureCache,
    ) -> np.ndarray:
        pair_mean, pair_std, node_mean, node_std, edge_mean, edge_std = self.normalizers
        pair_features = build_swap_features_for_fields(
            instance=instance,
            perm=perm,
            pairs=pairs,
            feature_fields=self.feature_fields,
            feature_cache=feature_cache,
        ).astype(np.float32)
        nodes = build_rqs_node_features(instance, perm, feature_cache)
        edges = build_rqs_edge_features(instance, perm, feature_cache)
        device = next(self.model.parameters()).device
        self.model.eval()
        with torch.no_grad():
            scores = self.model(
                torch.from_numpy((nodes - node_mean) / node_std).to(device=device),
                torch.from_numpy((edges - edge_mean) / edge_std).to(device=device),
                torch.from_numpy(np.asarray(pairs, dtype=np.int64)).to(device=device),
                torch.from_numpy((pair_features - pair_mean) / pair_std).to(device=device),
            )
        return scores.detach().cpu().numpy().astype(np.float64)


__all__ = [
    "RQS_EDGE_FEATURE_NAMES",
    "RQS_NODE_FEATURE_NAMES",
    "RelationalSelector",
    "build_rqs_edge_features",
    "build_rqs_node_features",
]
