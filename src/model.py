from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_mlp(input_dim: int, hidden_dim: int, output_dim: int, n_layers: int, dropout: float) -> nn.Sequential:
    if n_layers < 1:
        raise ValueError("n_layers must be at least 1.")
    layers: list[nn.Module] = []
    dim = input_dim
    for _ in range(n_layers - 1):
        layers.append(nn.Linear(dim, hidden_dim))
        layers.append(nn.ReLU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        dim = hidden_dim
    layers.append(nn.Linear(dim, output_dim))
    return nn.Sequential(*layers)


class AdditivePhiModel(nn.Module):
    def __init__(
        self,
        facility_dim: int,
        location_dim: int,
        hidden_dim: int = 128,
        n_layers: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.facility_encoder = _make_mlp(facility_dim, hidden_dim, hidden_dim, n_layers, dropout)
        self.location_encoder = _make_mlp(location_dim, hidden_dim, hidden_dim, n_layers, dropout)
        self.pair_head = _make_mlp(hidden_dim * 2, hidden_dim, 1, n_layers, dropout)

    def forward(self, facility_x: torch.Tensor, location_x: torch.Tensor) -> torch.Tensor:
        if facility_x.ndim != 3 or location_x.ndim != 3:
            raise ValueError("facility_x and location_x must have shape [batch, n, dim].")
        if facility_x.shape[0] != location_x.shape[0] or facility_x.shape[1] != location_x.shape[1]:
            raise ValueError("facility_x and location_x must have matching batch and n dimensions.")

        fac_h = self.facility_encoder(facility_x)
        loc_h = self.location_encoder(location_x)
        batch, n, hidden_dim = fac_h.shape

        fac_pair = fac_h.unsqueeze(2).expand(batch, n, n, hidden_dim)
        loc_pair = loc_h.unsqueeze(1).expand(batch, n, n, hidden_dim)
        pair = torch.cat([fac_pair, loc_pair], dim=-1)
        return self.pair_head(pair).squeeze(-1)


class EnhancedAdditivePhiModel(nn.Module):
    """Pairwise potential MLP over richer facility/location signatures."""

    def __init__(
        self,
        facility_dim: int = 39,
        location_dim: int = 39,
        hidden_dim: int = 192,
        n_layers: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.facility_encoder = _make_mlp(facility_dim, hidden_dim, hidden_dim, n_layers, dropout)
        self.location_encoder = _make_mlp(location_dim, hidden_dim, hidden_dim, n_layers, dropout)
        self.pair_head = _make_mlp(hidden_dim * 4, hidden_dim, 1, n_layers, dropout)

    def forward(self, facility_x: torch.Tensor, location_x: torch.Tensor) -> torch.Tensor:
        if facility_x.ndim != 3 or location_x.ndim != 3:
            raise ValueError("facility_x and location_x must have shape [batch, n, dim].")
        fac_h = self.facility_encoder(facility_x)
        loc_h = self.location_encoder(location_x)
        batch, n, hidden_dim = fac_h.shape
        fac_pair = fac_h.unsqueeze(2).expand(batch, n, n, hidden_dim)
        loc_pair = loc_h.unsqueeze(1).expand(batch, n, n, hidden_dim)
        pair = torch.cat(
            [fac_pair, loc_pair, fac_pair * loc_pair, torch.abs(fac_pair - loc_pair)], dim=-1
        )
        return self.pair_head(pair).squeeze(-1)


class FeaturePoolPhiModel(nn.Module):
    """Frozen enhanced-node prior plus a learned structured pair-feature residual."""

    def __init__(
        self,
        base_model: EnhancedAdditivePhiModel,
        pair_feature_dim: int = 8,
        hidden_dim: int = 96,
        n_layers: int = 3,
    ):
        super().__init__()
        self.base_model = base_model
        for parameter in self.base_model.parameters():
            parameter.requires_grad_(False)
        self.pair_residual = _make_mlp(pair_feature_dim, hidden_dim, 1, n_layers, 0.0)
        last = self.pair_residual[-1]
        if not isinstance(last, nn.Linear):
            raise TypeError("pair residual must end in a linear layer")
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

    def forward(
        self,
        facility_x: torch.Tensor,
        location_x: torch.Tensor,
        pair_x: torch.Tensor,
    ) -> torch.Tensor:
        if pair_x.ndim != 4 or pair_x.shape[:3] != (
            facility_x.shape[0], facility_x.shape[1], location_x.shape[1]
        ):
            raise ValueError("pair_x must have shape [batch, n, n, pair_feature_dim]")
        with torch.no_grad():
            base = self.base_model(facility_x, location_x)
        return base + self.pair_residual(pair_x).squeeze(-1)


def center_matrix(M: torch.Tensor) -> torch.Tensor:
    if M.ndim != 3:
        raise ValueError("M must have shape [batch, n, n].")
    return M - M.mean(dim=(1, 2), keepdim=True)


def normalize_weight_matrix(M: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    M = center_matrix(M)
    scale = M.abs().amax(dim=(1, 2), keepdim=True).clamp_min(eps)
    return M / scale


def log_sinkhorn(log_alpha: torch.Tensor, n_iters: int = 1) -> torch.Tensor:
    if log_alpha.ndim != 3:
        raise ValueError("log_alpha must have shape [batch, n, n].")
    out = log_alpha
    for _ in range(n_iters):
        out = out - torch.logsumexp(out, dim=2, keepdim=True)
        out = out - torch.logsumexp(out, dim=1, keepdim=True)
    return out


class WeightedMessagePassingLayer(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.location_linear = nn.Linear(hidden_dim, hidden_dim)
        self.facility_linear = nn.Linear(hidden_dim, hidden_dim)
        self.location_norm = nn.LayerNorm(hidden_dim)
        self.facility_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(
        self,
        loc_h: torch.Tensor,
        fac_h: torch.Tensor,
        D: torch.Tensor,
        F_mat: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        loc_msg = torch.bmm(D, self.location_linear(loc_h))
        fac_msg = torch.bmm(F_mat, self.facility_linear(fac_h))
        loc_h = self.location_norm(loc_h + self.dropout(torch.relu(loc_msg)))
        fac_h = self.facility_norm(fac_h + self.dropout(torch.relu(fac_msg)))
        return loc_h, fac_h


class CrossAttentionBlock(nn.Module):
    def __init__(self, hidden_dim: int, n_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        if hidden_dim % n_heads != 0:
            raise ValueError("hidden_dim must be divisible by n_heads.")
        self.loc_to_fac = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.fac_to_loc = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.loc_attn_norm = nn.LayerNorm(hidden_dim)
        self.fac_attn_norm = nn.LayerNorm(hidden_dim)
        self.loc_ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.fac_ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.loc_ffn_norm = nn.LayerNorm(hidden_dim)
        self.fac_ffn_norm = nn.LayerNorm(hidden_dim)

    def forward(self, loc_h: torch.Tensor, fac_h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        loc_ctx, _ = self.loc_to_fac(query=loc_h, key=fac_h, value=fac_h, need_weights=False)
        fac_ctx, _ = self.fac_to_loc(query=fac_h, key=loc_h, value=loc_h, need_weights=False)
        loc_h = self.loc_attn_norm(loc_h + loc_ctx)
        fac_h = self.fac_attn_norm(fac_h + fac_ctx)
        loc_h = self.loc_ffn_norm(loc_h + self.loc_ffn(loc_h))
        fac_h = self.fac_ffn_norm(fac_h + self.fac_ffn(fac_h))
        return loc_h, fac_h


class CrossGraphAttentionModel(nn.Module):
    def __init__(
        self,
        initial_dim: int = 16,
        hidden_dim: int = 128,
        gnn_layers: int = 4,
        cross_attention_layers: int = 1,
        attention_heads: int = 4,
        sinkhorn_iters: int = 1,
        tanh_clip: float = 10.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        if gnn_layers < 0:
            raise ValueError("gnn_layers must be non-negative.")
        if cross_attention_layers < 0:
            raise ValueError("cross_attention_layers must be non-negative.")
        if sinkhorn_iters < 0:
            raise ValueError("sinkhorn_iters must be non-negative.")
        self.initial_dim = int(initial_dim)
        self.hidden_dim = int(hidden_dim)
        self.sinkhorn_iters = int(sinkhorn_iters)
        self.tanh_clip = float(tanh_clip)

        self.location_initial = nn.Parameter(torch.empty(initial_dim))
        self.facility_initial = nn.Parameter(torch.empty(initial_dim))
        self.location_proj = nn.Linear(initial_dim, hidden_dim)
        self.facility_proj = nn.Linear(initial_dim, hidden_dim)
        self.gnn_layers = nn.ModuleList(
            [WeightedMessagePassingLayer(hidden_dim, dropout=dropout) for _ in range(gnn_layers)]
        )
        self.cross_attention_layers = nn.ModuleList(
            [
                CrossAttentionBlock(hidden_dim, n_heads=attention_heads, dropout=dropout)
                for _ in range(cross_attention_layers)
            ]
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.location_initial, mean=0.0, std=0.02)
        nn.init.normal_(self.facility_initial, mean=0.0, std=0.02)

    def forward(self, F_mat: torch.Tensor, D: torch.Tensor) -> torch.Tensor:
        if F_mat.ndim != 3 or D.ndim != 3:
            raise ValueError("F_mat and D must have shape [batch, n, n].")
        if F_mat.shape != D.shape:
            raise ValueError(f"F_mat and D must have matching shape, got {F_mat.shape} and {D.shape}.")
        batch, n, _ = F_mat.shape
        F_norm = normalize_weight_matrix(F_mat)
        D_norm = normalize_weight_matrix(D)

        loc_x = self.location_initial.view(1, 1, -1).expand(batch, n, -1)
        fac_x = self.facility_initial.view(1, 1, -1).expand(batch, n, -1)
        loc_h = self.location_proj(loc_x)
        fac_h = self.facility_proj(fac_x)

        for layer in self.gnn_layers:
            loc_h, fac_h = layer(loc_h, fac_h, D_norm, F_norm)
        for layer in self.cross_attention_layers:
            loc_h, fac_h = layer(loc_h, fac_h)

        loc_h = self.output_norm(loc_h)
        fac_h = self.output_norm(fac_h)
        logits = torch.bmm(fac_h, loc_h.transpose(1, 2)) / np.sqrt(float(self.hidden_dim))
        logits = self.tanh_clip * torch.tanh(logits)
        if self.sinkhorn_iters > 0:
            return log_sinkhorn(logits, n_iters=self.sinkhorn_iters)
        return logits


def permutation_score(phi: torch.Tensor, perm: torch.Tensor) -> torch.Tensor:
    if phi.ndim != 3 or perm.ndim != 2:
        raise ValueError("phi must be [batch, n, n] and perm must be [batch, n].")
    if phi.shape[0] != perm.shape[0] or phi.shape[1] != perm.shape[1]:
        raise ValueError("phi and perm must have matching batch and n dimensions.")
    idx = perm.unsqueeze(-1)
    return torch.gather(phi, dim=2, index=idx).squeeze(-1).sum(dim=1)


def swap_delta_score(phi: np.ndarray, perm: np.ndarray, a: int, b: int) -> float:
    return float(phi[a, perm[b]] + phi[b, perm[a]] - phi[a, perm[a]] - phi[b, perm[b]])


def _check() -> None:
    batch, n = 2, 6
    model = AdditivePhiModel(4, 4, hidden_dim=16, n_layers=2)
    facility_x = torch.randn(batch, n, 4)
    location_x = torch.randn(batch, n, 4)
    phi = model(facility_x, location_x)
    if phi.shape != (batch, n, n):
        raise AssertionError(f"unexpected phi shape: {phi.shape}")
    perm = torch.stack([torch.randperm(n) for _ in range(batch)])
    score = permutation_score(phi, perm)
    if score.shape != (batch,) or not torch.isfinite(score).all():
        raise AssertionError("invalid permutation score")
    cg_model = CrossGraphAttentionModel(
        initial_dim=8,
        hidden_dim=32,
        gnn_layers=2,
        cross_attention_layers=1,
        attention_heads=4,
        sinkhorn_iters=2,
    )
    F_mat = torch.rand(batch, n, n)
    D = torch.rand(batch, n, n)
    phi = cg_model(F_mat, D)
    if phi.shape != (batch, n, n) or not torch.isfinite(phi).all():
        raise AssertionError("invalid cross-graph phi")
    probs = phi.exp()
    if not torch.allclose(probs.sum(dim=1), torch.ones(batch, n), atol=1e-4):
        raise AssertionError("Sinkhorn column sums are invalid")
    print("model checks passed")


if __name__ == "__main__":
    _check()
