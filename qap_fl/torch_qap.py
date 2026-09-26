from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .features import SwapFeatureCache, build_swap_feature_cache
from .qap import QAPInstance


def _as_batched_pairs(values: torch.Tensor, batch: int) -> torch.Tensor:
    values = values.to(dtype=torch.long)
    if values.ndim == 1:
        values = values.unsqueeze(0).expand(batch, -1)
    if values.ndim != 2 or values.shape[0] != batch:
        raise ValueError("pair indices must have shape [pairs] or [batch, pairs]")
    return values


def batched_cost(flow: torch.Tensor, distance: torch.Tensor, perm: torch.Tensor) -> torch.Tensor:
    """Exact int64 QAP objective for a batch of permutations of one instance."""
    if perm.ndim != 2:
        raise ValueError("perm must have shape [batch, n]")
    assigned = distance[perm[:, :, None], perm[:, None, :]]
    return (flow.unsqueeze(0) * assigned).sum(dim=(1, 2))


def batched_pair_deltas(
    flow: torch.Tensor,
    distance: torch.Tensor,
    perm: torch.Tensor,
    first: torch.Tensor,
    second: torch.Tensor,
) -> torch.Tensor:
    """Exact swap deltas for shared or lane-specific pairs."""
    batch, n = perm.shape
    first = _as_batched_pairs(first, batch)
    second = _as_batched_pairs(second, batch)
    if first.shape != second.shape:
        raise ValueError("first and second must have matching shapes")
    pi = perm.gather(1, first)
    pj = perm.gather(1, second)

    values = (flow[first, first] - flow[second, second]) * (
        distance[pj, pj] - distance[pi, pi]
    )
    values = values + (flow[first, second] - flow[second, first]) * (
        distance[pj, pi] - distance[pi, pj]
    )

    k = torch.arange(n, device=perm.device, dtype=torch.long)
    mask = (k.view(1, 1, n) != first.unsqueeze(2)) & (
        k.view(1, 1, n) != second.unsqueeze(2)
    )
    perm_k = perm.unsqueeze(1)
    incoming = (
        flow[k.view(1, 1, n), first.unsqueeze(2)]
        - flow[k.view(1, 1, n), second.unsqueeze(2)]
    ) * (
        distance[perm_k, pj.unsqueeze(2)] - distance[perm_k, pi.unsqueeze(2)]
    )
    outgoing = (
        flow[first.unsqueeze(2), k.view(1, 1, n)]
        - flow[second.unsqueeze(2), k.view(1, 1, n)]
    ) * (
        distance[pj.unsqueeze(2), perm_k] - distance[pi.unsqueeze(2), perm_k]
    )
    return values + torch.where(mask, incoming + outgoing, 0).sum(dim=2)


def batched_initialize_delta(
    flow: torch.Tensor,
    distance: torch.Tensor,
    perm: torch.Tensor,
    pair_chunk: int = 256,
) -> torch.Tensor:
    """Build exact dense deltas without materializing a [B,n,n,n] tensor."""
    batch, n = perm.shape
    tri = torch.triu_indices(n, n, offset=1, device=perm.device)
    delta = torch.zeros((batch, n, n), dtype=torch.int64, device=perm.device)
    for offset in range(0, tri.shape[1], int(pair_chunk)):
        pairs = tri[:, offset : offset + int(pair_chunk)]
        values = batched_pair_deltas(flow, distance, perm, pairs[0], pairs[1])
        delta[:, pairs[0], pairs[1]] = values
        delta[:, pairs[1], pairs[0]] = values
    return delta


def batched_apply_swaps(
    flow: torch.Tensor,
    distance: torch.Tensor,
    perm: torch.Tensor,
    delta: torch.Tensor,
    first: torch.Tensor,
    second: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply one swap per lane and perform Taillard's exact O(n^2) update."""
    batch, n = perm.shape
    first = first.to(device=perm.device, dtype=torch.long).reshape(batch)
    second = second.to(device=perm.device, dtype=torch.long).reshape(batch)
    lanes = torch.arange(batch, device=perm.device)
    move_delta = delta[lanes, first, second].clone()

    updated_perm = perm.clone()
    old_first = updated_perm[lanes, first].clone()
    updated_perm[lanes, first] = updated_perm[lanes, second]
    updated_perm[lanes, second] = old_first

    p_first = updated_perm[lanes, first]
    p_second = updated_perm[lanes, second]
    flow_out = flow[first] - flow[second]
    distance_out = distance[p_second[:, None], updated_perm] - distance[p_first[:, None], updated_perm]
    flow_in = flow[:, first].T - flow[:, second].T
    distance_in = distance[updated_perm, p_second[:, None]] - distance[updated_perm, p_first[:, None]]
    updated_delta = delta + (
        (flow_out[:, :, None] - flow_out[:, None, :])
        * (distance_out[:, :, None] - distance_out[:, None, :])
        + (flow_in[:, :, None] - flow_in[:, None, :])
        * (distance_in[:, :, None] - distance_in[:, None, :])
    )

    k = torch.arange(n, device=perm.device).unsqueeze(0).expand(batch, -1)
    incident_first = torch.cat((first[:, None].expand(-1, n), second[:, None].expand(-1, n)), dim=1)
    incident_second = torch.cat((k, k), dim=1)
    incident_values = batched_pair_deltas(
        flow, distance, updated_perm, incident_first, incident_second
    )
    lane_index = lanes[:, None].expand_as(incident_first)
    updated_delta[lane_index, incident_first, incident_second] = incident_values
    updated_delta[lane_index, incident_second, incident_first] = incident_values
    diagonal = torch.arange(n, device=perm.device)
    updated_delta[:, diagonal, diagonal] = 0
    return updated_perm, updated_delta, move_delta


@dataclass(frozen=True)
class TorchSwapFeatureCache:
    F: torch.Tensor
    D: torch.Tensor
    f_sum: torch.Tensor
    f_mean: torch.Tensor
    f_std: torch.Tensor
    d_sum: torch.Tensor
    d_mean: torch.Tensor
    d_std: torch.Tensor
    qap_scale: float
    denom: float
    facility_cosine: torch.Tensor
    facility_l1: torch.Tensor
    location_cosine: torch.Tensor
    location_l1: torch.Tensor
    anchors: torch.Tensor

    @classmethod
    def from_numpy(cls, cache: SwapFeatureCache, device: torch.device) -> "TorchSwapFeatureCache":
        def floating(value):
            # The selector consumes float32 inputs. Keeping the entire feature
            # path in float64 is especially expensive on consumer CUDA GPUs and
            # provides no extra precision after the final cast.
            return torch.as_tensor(value, dtype=torch.float32, device=device)

        return cls(
            F=floating(cache.F), D=floating(cache.D),
            f_sum=floating(cache.f_sum), f_mean=floating(cache.f_mean), f_std=floating(cache.f_std),
            d_sum=floating(cache.d_sum), d_mean=floating(cache.d_mean), d_std=floating(cache.d_std),
            qap_scale=float(cache.qap_scale), denom=float(cache.denom),
            facility_cosine=floating(cache.facility_cosine), facility_l1=floating(cache.facility_l1),
            location_cosine=floating(cache.location_cosine), location_l1=floating(cache.location_l1),
            anchors=torch.as_tensor(cache.anchors, dtype=torch.long, device=device),
        )

    @classmethod
    def from_instance(cls, instance: QAPInstance, device: torch.device) -> "TorchSwapFeatureCache":
        return cls.from_numpy(build_swap_feature_cache(instance), device)


def batched_swap_features(
    cache: TorchSwapFeatureCache,
    perm: torch.Tensor,
    pairs: torch.Tensor,
) -> torch.Tensor:
    """Torch equivalent of the frozen D97 selector's 39 feature columns."""
    if pairs.ndim == 2:
        pairs = pairs.unsqueeze(0).expand(perm.shape[0], -1, -1)
    if pairs.ndim != 3 or pairs.shape[0] != perm.shape[0] or pairs.shape[2] != 2:
        raise ValueError("pairs must have shape [pairs,2] or [batch,pairs,2]")
    pairs = pairs.to(device=perm.device, dtype=torch.long)
    batch, count, _ = pairs.shape
    i, j = pairs[:, :, 0], pairs[:, :, 1]
    pi = perm.gather(1, i)
    pj = perm.gather(1, j)
    F, D = cache.F, cache.D

    pair_old = (
        F[i, i] * D[pi, pi] + F[j, j] * D[pj, pj]
        + F[i, j] * D[pi, pj] + F[j, i] * D[pj, pi]
    )
    pair_swap = (
        F[i, i] * D[pj, pj] + F[j, j] * D[pi, pi]
        + F[i, j] * D[pj, pi] + F[j, i] * D[pi, pj]
    )

    assigned = D[perm[:, :, None], perm[:, None, :]]
    weighted = F.unsqueeze(0) * assigned
    node = (
        weighted.sum(dim=2) + weighted.sum(dim=1)
        - torch.diagonal(F).unsqueeze(0) * torch.diagonal(assigned, dim1=1, dim2=2)
    ) / cache.qap_scale
    node_i, node_j = node.gather(1, i), node.gather(1, j)

    anchors = cache.anchors[i, j]
    anchor_mask = anchors >= 0
    safe = anchors.clamp_min(0)
    pk = perm.gather(1, safe.reshape(batch, -1)).reshape_as(safe)
    ii, jj, pii, pjj = i.unsqueeze(2), j.unsqueeze(2), pi.unsqueeze(2), pj.unsqueeze(2)
    old_terms = (
        F[ii, safe] * D[pii, pk] + F[safe, ii] * D[pk, pii]
        + F[jj, safe] * D[pjj, pk] + F[safe, jj] * D[pk, pjj]
    )
    swap_terms = (
        F[ii, safe] * D[pjj, pk] + F[safe, ii] * D[pk, pjj]
        + F[jj, safe] * D[pii, pk] + F[safe, jj] * D[pk, pii]
    )
    zeros = torch.zeros((), dtype=F.dtype, device=perm.device)
    old_terms = torch.where(anchor_mask, old_terms, zeros)
    swap_terms = torch.where(anchor_mask, swap_terms, zeros)
    delta_terms = swap_terms - old_terms
    counts = anchor_mask.sum(dim=2)
    active = counts > 0
    expansion = torch.where(
        active, float(max(perm.shape[1] - 2, 1)) / counts.clamp_min(1), 0.0
    ).to(F.dtype)
    scaled = torch.where(
        anchor_mask, delta_terms * expansion.unsqueeze(2) / cache.qap_scale, zeros
    )
    anchor_mean = scaled.sum(dim=2) / counts.clamp_min(1)
    centered = torch.where(anchor_mask, scaled - anchor_mean.unsqueeze(2), zeros)
    anchor_std = torch.sqrt((centered * centered).sum(dim=2) / counts.clamp_min(1))
    anchor_min = torch.where(anchor_mask, scaled, torch.inf).min(dim=2).values
    anchor_max = torch.where(anchor_mask, scaled, -torch.inf).max(dim=2).values
    anchor_min = torch.where(active, anchor_min, zeros)
    anchor_max = torch.where(active, anchor_max, zeros)

    features = torch.zeros((batch, count, 39), dtype=F.dtype, device=perm.device)
    features[:, :, 0] = cache.f_sum[i]
    features[:, :, 1] = cache.f_sum[j]
    features[:, :, 2] = cache.f_mean[i]
    features[:, :, 3] = cache.f_mean[j]
    features[:, :, 4] = cache.f_std[i]
    features[:, :, 5] = cache.f_std[j]
    features[:, :, 6] = cache.d_sum[pi]
    features[:, :, 7] = cache.d_sum[pj]
    features[:, :, 8] = cache.d_mean[pi]
    features[:, :, 9] = cache.d_mean[pj]
    features[:, :, 10] = cache.d_std[pi]
    features[:, :, 11] = cache.d_std[pj]
    features[:, :, 12] = (cache.f_sum[i] - cache.f_sum[j]).abs()
    features[:, :, 13] = (cache.f_std[i] - cache.f_std[j]).abs()
    features[:, :, 14] = (cache.d_sum[pi] - cache.d_sum[pj]).abs()
    features[:, :, 15] = (cache.d_std[pi] - cache.d_std[pj]).abs()
    features[:, :, 16] = (i - j).abs() / cache.denom
    features[:, :, 17] = (pi - pj).abs() / cache.denom
    features[:, :, 18] = pair_old / cache.qap_scale
    features[:, :, 19] = pair_swap / cache.qap_scale
    features[:, :, 20] = (pair_swap - pair_old) / cache.qap_scale
    features[:, :, 21] = cache.facility_cosine[i, j]
    features[:, :, 22] = cache.facility_l1[i, j]
    features[:, :, 23] = cache.location_cosine[pi, pj]
    features[:, :, 24] = cache.location_l1[pi, pj]
    features[:, :, 25] = old_terms.sum(dim=2) * expansion / cache.qap_scale
    features[:, :, 26] = swap_terms.sum(dim=2) * expansion / cache.qap_scale
    features[:, :, 27] = delta_terms.sum(dim=2) * expansion / cache.qap_scale
    features[:, :, 28] = anchor_mean
    features[:, :, 29] = anchor_std
    features[:, :, 30] = anchor_min
    features[:, :, 31] = anchor_max
    features[:, :, 32] = pi / cache.denom
    features[:, :, 33] = pj / cache.denom
    features[:, :, 34] = (pi - pj).abs() / cache.denom
    features[:, :, 35] = node_i
    features[:, :, 36] = node_j
    features[:, :, 37] = (node_i - node_j).abs()
    features[:, :, 38] = node_i + node_j
    return features


__all__ = [
    "TorchSwapFeatureCache",
    "batched_apply_swaps",
    "batched_cost",
    "batched_initialize_delta",
    "batched_pair_deltas",
    "batched_swap_features",
]
