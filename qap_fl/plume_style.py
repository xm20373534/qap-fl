from __future__ import annotations

import math

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn


def _mlp(input_dim: int, hidden_dim: int, output_dim: int, layers: int = 2) -> nn.Sequential:
    modules: list[nn.Module] = []
    current = input_dim
    for _ in range(layers - 1):
        modules.extend((nn.Linear(current, hidden_dim), nn.ReLU()))
        current = hidden_dim
    modules.append(nn.Linear(current, output_dim))
    return nn.Sequential(*modules)


def row_normalize(values: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return values / values.sum(dim=-1, keepdim=True).clamp_min(eps)


def log_sinkhorn(logits: torch.Tensor, iterations: int) -> torch.Tensor:
    output = logits
    for _ in range(int(iterations)):
        output = output - torch.logsumexp(output, dim=2, keepdim=True)
        output = output - torch.logsumexp(output, dim=1, keepdim=True)
    return output


class PairEncoder(nn.Module):
    def __init__(self, hidden_dim: int, inverse_kernel: bool) -> None:
        super().__init__()
        self.inverse_kernel = bool(inverse_kernel)
        self.entry = _mlp(1, hidden_dim, hidden_dim, layers=2)
        self.mix = _mlp(3 * hidden_dim, hidden_dim, hidden_dim, layers=2)
        self.message = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, matrix: torch.Tensor) -> torch.Tensor:
        embedded = self.entry(matrix.unsqueeze(-1))
        pooled = torch.cat(
            (embedded.sum(dim=2), embedded.mean(dim=2), embedded.amax(dim=2)),
            dim=-1,
        )
        state = self.mix(pooled)
        if self.inverse_kernel:
            batch, n, _ = matrix.shape
            eye = torch.eye(n, dtype=torch.bool, device=matrix.device).unsqueeze(0)
            weights = torch.where(eye, torch.zeros_like(matrix), 1.0 / (matrix + 1e-3))
        else:
            weights = matrix.clamp_min(0.0)
        weights = row_normalize(weights)
        return state + torch.bmm(weights, self.message(state))


class PlumeStyleModel(nn.Module):
    """Equation-level reproduction of the architecture in arXiv:2503.20001v2.

    This is independently implemented from the paper because no official repository
    or checkpoint was found locally. It must be reported as PLUME-style, not official.
    """

    def __init__(self, hidden_dim: int = 128, layers: int = 3, alpha: float = 40.0) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.alpha = float(alpha)
        self.facility_encoder = PairEncoder(hidden_dim, inverse_kernel=False)
        self.location_encoder = PairEncoder(hidden_dim, inverse_kernel=True)
        self.position = _mlp(2, hidden_dim, hidden_dim, layers=2)
        self.fusion = nn.ModuleList(
            [_mlp(3 * hidden_dim, hidden_dim, hidden_dim, layers=3) for _ in range(layers)]
        )

    def embeddings(self, flow: torch.Tensor, distance: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        facility = self.facility_encoder(flow)
        location = self.location_encoder(distance)
        position = self.position(coords)
        state = None
        for fusion in self.fusion:
            state = fusion(torch.cat((facility, location, position), dim=-1))
            facility = state
            location = state
        if state is None:
            raise RuntimeError("PLUME-style model requires at least one fusion layer")
        return state

    def logits(self, flow: torch.Tensor, distance: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        state = self.embeddings(flow, distance, coords)
        return self.alpha * torch.tanh(torch.bmm(state, state.transpose(1, 2)))

    def soft_permutation(
        self,
        flow: torch.Tensor,
        distance: torch.Tensor,
        coords: torch.Tensor,
        tau: float = 3.0,
        noise_scale: float = 0.01,
        sinkhorn_iterations: int = 100,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.logits(flow, distance, coords)
        if noise_scale > 0:
            uniform = torch.rand_like(logits).clamp_(1e-8, 1.0 - 1e-8)
            noise = -torch.log(-torch.log(uniform))
            logits = logits + float(noise_scale) * noise
        scaled = logits / float(tau)
        return torch.exp(log_sinkhorn(scaled, sinkhorn_iterations)), logits


def qap_relaxation_loss(soft: torch.Tensor, flow: torch.Tensor, distance: torch.Tensor) -> torch.Tensor:
    placed_flow = torch.bmm(torch.bmm(soft, flow), soft.transpose(1, 2))
    raw = (placed_flow * distance).sum(dim=(1, 2))
    scale = flow.abs().sum(dim=(1, 2)) * distance.abs().mean(dim=(1, 2))
    return (raw / scale.clamp_min(1e-8)).mean()


def decode_logits(logits: np.ndarray) -> np.ndarray:
    """Return facility-to-location permutation for P F P^T in the paper."""
    location_rows, facility_columns = linear_sum_assignment(-np.asarray(logits, dtype=np.float64))
    if not np.array_equal(location_rows, np.arange(len(location_rows))):
        raise RuntimeError("unexpected Hungarian row ordering")
    facility_to_location = np.empty(len(facility_columns), dtype=np.int64)
    facility_to_location[facility_columns] = location_rows
    return facility_to_location


def geometric_er_batch(
    batch_size: int,
    n: int,
    density: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coords = rng.uniform(0.0, 1.0, size=(batch_size, n, 2)).astype(np.float32)
    difference = coords[:, :, None, :] - coords[:, None, :, :]
    distance = np.linalg.norm(difference, axis=-1).astype(np.float32)
    flow = np.zeros((batch_size, n, n), dtype=np.float32)
    upper = np.triu_indices(n, k=1)
    values = rng.uniform(0.0, 1.0, size=(batch_size, len(upper[0]))).astype(np.float32)
    mask = rng.random(size=values.shape) < float(density)
    values *= mask
    flow[:, upper[0], upper[1]] = values
    flow[:, upper[1], upper[0]] = values
    return flow, distance, coords


def _check() -> None:
    rng = np.random.default_rng(0)
    flow, distance, coords = geometric_er_batch(3, 12, 0.5, rng)
    model = PlumeStyleModel(hidden_dim=32, layers=3)
    soft, logits = model.soft_permutation(
        torch.from_numpy(flow), torch.from_numpy(distance), torch.from_numpy(coords), sinkhorn_iterations=20
    )
    if soft.shape != (3, 12, 12) or logits.shape != (3, 12, 12):
        raise AssertionError("unexpected PLUME-style shapes")
    if not torch.allclose(soft.sum(dim=1), torch.ones(3, 12), atol=1e-4):
        raise AssertionError("invalid Sinkhorn columns")
    permutation = decode_logits(logits[0].detach().numpy())
    if not np.array_equal(np.sort(permutation), np.arange(12)):
        raise AssertionError("invalid Hungarian permutation")
    print("PLUME-style checks passed")


if __name__ == "__main__":
    _check()
