from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .features import EPS, SwapFeatureCache, all_swap_pairs, build_swap_feature_cache
from .qap import QAPInstance


NODE_FEATURE_NAMES = [
    "facility_sum",
    "facility_std",
    "assigned_location_sum",
    "assigned_location_std",
    "facility_mean_l1",
    "assigned_location_mean_l1",
    "node_contribution",
    "assignment_index",
    "facility_strength_percentile",
    "location_centrality_percentile",
    "instance_log_n",
]


def _percentile_rank(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if len(values) <= 1:
        return np.zeros(len(values), dtype=np.float32)
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    return (ranks / float(len(values) - 1)).astype(np.float32)


def build_node_features(
    instance: QAPInstance,
    perm: np.ndarray,
    feature_cache: SwapFeatureCache | None = None,
    node_contribution: np.ndarray | None = None,
) -> np.ndarray:
    perm = np.asarray(perm, dtype=np.int64)
    cache = build_swap_feature_cache(instance) if feature_cache is None else feature_cache
    n = instance.n
    if perm.shape != (n,) or len(np.unique(perm)) != n:
        raise ValueError("perm must be a valid permutation")
    if node_contribution is None:
        d_perm = cache.D[np.ix_(perm, perm)]
        incoming = np.sum(cache.F * d_perm, axis=1)
        outgoing = np.sum(cache.F * d_perm, axis=0)
        diagonal = np.diag(cache.F) * np.diag(d_perm)
        contribution = (incoming + outgoing - diagonal) / cache.qap_scale
    else:
        contribution = np.asarray(node_contribution, dtype=np.float64)
        if contribution.shape != (n,):
            raise ValueError("node_contribution must have shape [n]")
    facility_l1 = np.mean(cache.facility_l1, axis=1)
    location_l1 = np.mean(cache.location_l1, axis=1)
    denom = float(max(n - 1, 1))
    features = np.column_stack(
        [
            cache.f_sum,
            cache.f_std,
            cache.d_sum[perm],
            cache.d_std[perm],
            facility_l1,
            location_l1[perm],
            contribution,
            perm.astype(np.float64) / denom,
            _percentile_rank(cache.f_sum),
            _percentile_rank(cache.d_sum)[perm],
            np.full(n, np.log(float(n))),
        ]
    ).astype(np.float32)
    if not np.all(np.isfinite(features)):
        raise ValueError("node features contain non-finite values")
    return features


class FactorizedProposalNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, embedding_dim: int = 32) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.embedding_dim = int(embedding_dim)
        self.node_encoder = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.embedding_dim),
            nn.ReLU(),
        )
        self.anchor_head = nn.Linear(self.embedding_dim, 1)
        self.pair_head = nn.Sequential(
            nn.Linear(3 * self.embedding_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1),
        )

    def encode(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embedding = self.node_encoder(features)
        anchors = self.anchor_head(embedding).squeeze(-1)
        return embedding, anchors

    def pair_scores(
        self, embedding: torch.Tensor, left: torch.Tensor, right: torch.Tensor
    ) -> torch.Tensor:
        a = embedding.index_select(0, left)
        b = embedding.index_select(0, right)
        symmetric = torch.cat([a + b, torch.abs(a - b), a * b], dim=1)
        return self.pair_head(symmetric).squeeze(-1)


def _canonical_pair(i: int, j: int) -> tuple[int, int]:
    return (i, j) if i < j else (j, i)


def _fill_random_pairs(
    selected: list[tuple[int, int]],
    n: int,
    target: int,
    rng: np.random.Generator,
) -> list[tuple[int, int]]:
    seen = set(selected)
    available = [tuple(map(int, pair)) for pair in all_swap_pairs(n) if tuple(map(int, pair)) not in seen]
    if target > len(seen) and available:
        count = min(target - len(seen), len(available))
        choice = rng.choice(len(available), size=count, replace=False)
        selected.extend(available[int(index)] for index in np.asarray(choice).reshape(-1))
    return selected


def random_pool(n: int, pool_size: int, rng: np.random.Generator) -> np.ndarray:
    pairs = all_swap_pairs(n)
    count = min(int(pool_size), len(pairs))
    choice = rng.choice(len(pairs), size=count, replace=False)
    return pairs[choice].astype(np.int64)


def hand_pressure_pool(
    node_features: np.ndarray,
    rng: np.random.Generator,
    *,
    anchor_count: int = 8,
    partners_per_anchor: int = 6,
    structured_count: int = 48,
    pool_size: int = 64,
) -> np.ndarray:
    n = len(node_features)
    contribution_index = NODE_FEATURE_NAMES.index("node_contribution")
    anchors = np.argsort(-node_features[:, contribution_index], kind="stable")[: min(anchor_count, n)]
    selected: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for anchor in anchors:
        partners = np.delete(np.arange(n, dtype=np.int64), int(anchor))
        count = min(int(partners_per_anchor), len(partners))
        for partner in partners[rng.choice(len(partners), size=count, replace=False)]:
            pair = _canonical_pair(int(anchor), int(partner))
            if pair not in seen:
                selected.append(pair)
                seen.add(pair)
    selected = _fill_random_pairs(selected, n, min(int(structured_count), int(pool_size)), rng)
    selected = _fill_random_pairs(selected, n, min(int(pool_size), n * (n - 1) // 2), rng)
    return np.asarray(selected[:pool_size], dtype=np.int64)


def learned_proposal_pool(
    model: FactorizedProposalNet,
    node_features: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    rng: np.random.Generator,
    device: torch.device,
    *,
    anchor_count: int = 8,
    partners_per_anchor: int = 6,
    structured_count: int = 48,
    pool_size: int = 64,
) -> np.ndarray:
    n = len(node_features)
    normalized = (np.asarray(node_features, dtype=np.float32) - mean) / std
    x = torch.from_numpy(normalized).to(device=device, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        embedding, anchor_scores = model.encode(x)
        anchor_order = np.argsort(-anchor_scores.detach().cpu().numpy(), kind="stable")
        anchors = anchor_order[: min(int(anchor_count), n)]
        scored: dict[tuple[int, int], float] = {}
        primary: list[tuple[int, int]] = []
        for anchor_raw in anchors:
            anchor = int(anchor_raw)
            partners = np.delete(np.arange(n, dtype=np.int64), anchor)
            left = torch.full((len(partners),), anchor, dtype=torch.long, device=device)
            right = torch.from_numpy(partners).to(device=device, dtype=torch.long)
            values = model.pair_scores(embedding, left, right).detach().cpu().numpy()
            order = np.argsort(-values, kind="stable")
            for index in order:
                pair = _canonical_pair(anchor, int(partners[int(index)]))
                scored[pair] = max(scored.get(pair, -np.inf), float(values[int(index)]))
            for index in order[: min(int(partners_per_anchor), len(order))]:
                pair = _canonical_pair(anchor, int(partners[int(index)]))
                if pair not in primary:
                    primary.append(pair)
        selected = primary[: int(structured_count)]
        if len(selected) < int(structured_count):
            remainder = sorted(scored, key=lambda pair: (-scored[pair], pair))
            selected.extend(pair for pair in remainder if pair not in selected)
            selected = selected[: int(structured_count)]
    selected = _fill_random_pairs(selected, n, min(int(pool_size), n * (n - 1) // 2), rng)
    return np.asarray(selected[:pool_size], dtype=np.int64)


__all__ = [
    "NODE_FEATURE_NAMES",
    "FactorizedProposalNet",
    "build_node_features",
    "hand_pressure_pool",
    "learned_proposal_pool",
    "random_pool",
]
