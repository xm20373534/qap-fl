from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class QAPInstance:
    name: str
    F: np.ndarray
    D: np.ndarray
    optimum: float | None = None

    def __post_init__(self) -> None:
        if self.F.ndim != 2 or self.D.ndim != 2:
            raise ValueError("F and D must be 2D matrices.")
        if self.F.shape[0] != self.F.shape[1]:
            raise ValueError(f"F must be square, got {self.F.shape}.")
        if self.D.shape[0] != self.D.shape[1]:
            raise ValueError(f"D must be square, got {self.D.shape}.")
        if self.F.shape != self.D.shape:
            raise ValueError(f"F and D shape mismatch: {self.F.shape} vs {self.D.shape}.")

    @property
    def n(self) -> int:
        return int(self.F.shape[0])


def is_valid_perm(perm: np.ndarray, n: int | None = None) -> bool:
    perm = np.asarray(perm)
    if perm.ndim != 1:
        return False
    if n is None:
        n = len(perm)
    if len(perm) != n:
        return False
    if n == 0:
        return True
    if np.any(perm < 0) or np.any(perm >= n):
        return False
    return len(np.unique(perm)) == n


def random_perm(n: int, rng: np.random.Generator) -> np.ndarray:
    return rng.permutation(int(n)).astype(np.int64)


def compute_cost(perm: np.ndarray, F: np.ndarray, D: np.ndarray) -> float:
    perm = np.asarray(perm, dtype=np.int64)
    if not is_valid_perm(perm, F.shape[0]):
        raise ValueError("perm is not a valid permutation.")
    return float(np.sum(F * D[np.ix_(perm, perm)]))


def swap_perm(perm: np.ndarray, a: int, b: int) -> np.ndarray:
    out = np.asarray(perm, dtype=np.int64).copy()
    out[a], out[b] = out[b], out[a]
    return out


def swap_delta_cost(perm: np.ndarray, F: np.ndarray, D: np.ndarray, a: int, b: int) -> float:
    perm = np.asarray(perm, dtype=np.int64)
    if a == b:
        return 0.0
    n = len(perm)
    if a < 0 or b < 0 or a >= n or b >= n:
        raise IndexError("swap indices are out of range.")

    pa = int(perm[a])
    pb = int(perm[b])
    old_pair = (
        F[a, a] * D[pa, pa]
        + F[b, b] * D[pb, pb]
        + F[a, b] * D[pa, pb]
        + F[b, a] * D[pb, pa]
    )
    new_pair = (
        F[a, a] * D[pb, pb]
        + F[b, b] * D[pa, pa]
        + F[a, b] * D[pb, pa]
        + F[b, a] * D[pa, pb]
    )
    delta = float(new_pair - old_pair)

    for k in range(n):
        if k == a or k == b:
            continue
        pk = int(perm[k])
        old_terms = (
            F[a, k] * D[pa, pk]
            + F[k, a] * D[pk, pa]
            + F[b, k] * D[pb, pk]
            + F[k, b] * D[pk, pb]
        )
        new_terms = (
            F[a, k] * D[pb, pk]
            + F[k, a] * D[pk, pb]
            + F[b, k] * D[pa, pk]
            + F[k, b] * D[pk, pa]
        )
        delta += float(new_terms - old_terms)
    return float(delta)


def gap_percent(cost: float, optimum: float | None) -> float | None:
    if optimum is None or abs(float(optimum)) <= 1e-12:
        return None
    return (float(cost) - float(optimum)) / abs(float(optimum)) * 100.0


def check_swap_delta(F: np.ndarray, D: np.ndarray, trials: int = 100, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    n = F.shape[0]
    for _ in range(int(trials)):
        perm = random_perm(n, rng)
        a, b = rng.choice(n, 2, replace=False)
        delta = swap_delta_cost(perm, F, D, int(a), int(b))
        old_cost = compute_cost(perm, F, D)
        new_cost = compute_cost(swap_perm(perm, int(a), int(b)), F, D)
        if not np.isclose(old_cost + delta, new_cost, rtol=1e-9, atol=1e-9):
            raise AssertionError(f"swap delta mismatch: old={old_cost}, delta={delta}, new={new_cost}")

