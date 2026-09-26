from __future__ import annotations

import numpy as np

from .qap import QAPInstance


def _uniform_symmetric_matrix(
    n: int,
    rng: np.random.Generator,
    zero_diagonal: bool = True,
    symmetrize_mode: str = "average",
) -> np.ndarray:
    if symmetrize_mode == "average":
        M = rng.uniform(0.0, 1.0, size=(n, n))
        M = (M + M.T) / 2.0
    elif symmetrize_mode == "upper_mirror":
        M = np.zeros((n, n), dtype=np.float64)
        upper = np.triu_indices(n, k=1 if zero_diagonal else 0)
        values = rng.uniform(0.0, 1.0, size=len(upper[0]))
        M[upper] = values
        M = M + np.triu(M, k=1).T
    else:
        raise ValueError(f"Unsupported symmetrize_mode: {symmetrize_mode}")
    if zero_diagonal:
        np.fill_diagonal(M, 0.0)
    return M.astype(np.float64)


def _euclidean_distance(n: int, rng: np.random.Generator) -> np.ndarray:
    coords = rng.uniform(0.0, 1.0, size=(n, 2))
    diff = coords[:, None, :] - coords[None, :, :]
    D = np.linalg.norm(diff, axis=-1)
    np.fill_diagonal(D, 0.0)
    return D.astype(np.float64)


def _sparsify_symmetric_off_diagonal(
    M: np.ndarray,
    rng: np.random.Generator,
    sparsity: float,
) -> np.ndarray:
    if not 0.0 <= sparsity <= 1.0:
        raise ValueError("sparsity must be in [0, 1].")
    n = M.shape[0]
    keep = rng.random(size=(n, n)) >= sparsity
    keep = np.triu(keep, k=1)
    keep = keep | keep.T
    out = np.asarray(M, dtype=np.float64).copy()
    out[~keep] = 0.0
    np.fill_diagonal(out, 0.0)
    return out


def _sawt_geometric_qap(
    n: int,
    rng: np.random.Generator,
    flow_sparsity: float = 0.7,
    symmetrize_mode: str = "average",
) -> tuple[np.ndarray, np.ndarray]:
    F = _uniform_symmetric_matrix(n, rng, symmetrize_mode=symmetrize_mode)
    F = _sparsify_symmetric_off_diagonal(F, rng, sparsity=flow_sparsity)
    D = _euclidean_distance(n, rng)
    return F, D


def _tai_a_qap(
    n: int,
    rng: np.random.Generator,
    symmetrize_mode: str = "average",
) -> tuple[np.ndarray, np.ndarray]:
    F = _uniform_symmetric_matrix(n, rng, symmetrize_mode=symmetrize_mode)
    D = _uniform_symmetric_matrix(n, rng, symmetrize_mode=symmetrize_mode)
    return F, D


def _hierarchical_block_qap(
    n: int,
    rng: np.random.Generator,
    levels: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a relabelled, noisy hierarchical-block QAP."""
    if n < 4:
        raise ValueError("hierarchical_block requires n >= 4")
    if levels is None:
        levels = max(2, min(4, int(np.ceil(np.log2(n / 4.0))) + 1))
    latent = np.arange(n, dtype=np.int64)
    shared = np.zeros((n, n), dtype=np.float64)
    weights = np.geomspace(0.12, 0.48, num=int(levels))
    weights = weights / weights.sum()
    for level, weight in enumerate(weights, start=1):
        group_count = min(2**level, n)
        group = np.minimum((latent * group_count) // n, group_count - 1)
        shared += float(weight) * (group[:, None] == group[None, :])
    flow_noise = rng.uniform(0.0, 0.04, size=(n, n))
    flow_noise = (flow_noise + flow_noise.T) / 2.0
    distance_noise = rng.uniform(-0.025, 0.025, size=(n, n))
    distance_noise = (distance_noise + distance_noise.T) / 2.0
    flow = np.clip(0.02 + 0.93 * shared + flow_noise, 0.0, 1.0)
    distance = np.clip(1.00 - 0.88 * shared + distance_noise, 0.05, 1.0)
    np.fill_diagonal(flow, 0.0)
    np.fill_diagonal(distance, 0.0)
    facility_labels = rng.permutation(n)
    location_labels = rng.permutation(n)
    return (
        flow[np.ix_(facility_labels, facility_labels)].astype(np.float64),
        distance[np.ix_(location_labels, location_labels)].astype(np.float64),
    )


def _hierarchical_sparse_discrete_qap(
    n: int,
    rng: np.random.Generator,
    levels: int | None = None,
    keep_probability: float = 0.55,
    quantization_levels: int = 32,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a sparse, quantized observation of a hidden hierarchy.

    This generator uses only the published qualitative Taixxeyy structure:
    facilities in the same latent groups interact more strongly, while their
    matched locations are closer. Sparsification noise and relabeling are
    sampled independently for the two matrices.
    """
    if n < 4:
        raise ValueError("hierarchical_sparse_discrete requires n >= 4")
    if levels is None:
        levels = max(2, min(5, int(np.ceil(np.log2(n / 4.0))) + 1))
    if not 0.0 < float(keep_probability) <= 1.0:
        raise ValueError("keep_probability must be in (0, 1]")
    if int(quantization_levels) < 3:
        raise ValueError("quantization_levels must be at least 3")

    latent = np.arange(n, dtype=np.int64)
    shared = np.zeros((n, n), dtype=np.float64)
    weights = np.geomspace(0.10, 0.55, num=int(levels))
    weights /= weights.sum()
    for level, weight in enumerate(weights, start=1):
        group_count = min(2**level, n)
        group = np.minimum((latent * group_count) // n, group_count - 1)
        shared += float(weight) * (group[:, None] == group[None, :])

    def symmetric_noise(scale: float) -> np.ndarray:
        upper = rng.uniform(-scale, scale, size=n * (n - 1) // 2)
        matrix = np.zeros((n, n), dtype=np.float64)
        tri = np.triu_indices(n, k=1)
        matrix[tri] = upper
        return matrix + matrix.T

    flow_latent = np.clip(0.08 + 0.84 * shared + symmetric_noise(0.06), 0.0, 1.0)
    distance_latent = np.clip(0.92 - 0.78 * shared + symmetric_noise(0.06), 0.0, 1.0)
    tri = np.triu_indices(n, k=1)
    for matrix in (flow_latent, distance_latent):
        keep = rng.random(size=len(tri[0])) < float(keep_probability)
        values = np.rint(matrix[tri] * float(int(quantization_levels) - 1))
        values = np.where(keep, np.maximum(values, 1.0), 0.0)
        matrix.fill(0.0)
        matrix[tri] = values
        matrix[:] = matrix + matrix.T

    facility_labels = rng.permutation(n)
    location_labels = rng.permutation(n)
    return (
        flow_latent[np.ix_(facility_labels, facility_labels)].astype(np.float64),
        distance_latent[np.ix_(location_labels, location_labels)].astype(np.float64),
    )


def _legacy_uniform_euclidean_qap(n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    F = _uniform_symmetric_matrix(n, rng)
    D = _euclidean_distance(n, rng)
    return F, D


def generate_synthetic_qap(
    n: int,
    rng: np.random.Generator,
    flow_type: str = "uniform_symmetric",
    distance_type: str = "euclidean",
    distribution: str | None = None,
    flow_sparsity: float = 0.7,
    symmetrize_mode: str = "average",
    name: str | None = None,
) -> QAPInstance:
    if distribution is None:
        if flow_type == "uniform_symmetric" and distance_type == "euclidean":
            distribution = "legacy_uniform_euclidean"
        else:
            distribution = f"{flow_type}_{distance_type}"

    if distribution == "legacy_uniform_euclidean":
        F, D = _legacy_uniform_euclidean_qap(n, rng)
    elif distribution == "sawt_geometric":
        F, D = _sawt_geometric_qap(
            n,
            rng,
            flow_sparsity=flow_sparsity,
            symmetrize_mode=symmetrize_mode,
        )
    elif distribution in {"tai_a", "uniform_random"}:
        F, D = _tai_a_qap(n, rng, symmetrize_mode=symmetrize_mode)
    elif distribution in {"hierarchical_block", "taix_block"}:
        F, D = _hierarchical_block_qap(n, rng)
    elif distribution in {"hierarchical_sparse_discrete", "taix_sparse_discrete"}:
        F, D = _hierarchical_sparse_discrete_qap(n, rng)
    else:
        raise ValueError(f"Unsupported synthetic distribution: {distribution}")
    return QAPInstance(name=name or f"synthetic_n{n}", F=F, D=D, optimum=None)


def generate_synthetic_instances(config: dict) -> list[QAPInstance]:
    train_cfg = config["data"]["train"]
    if train_cfg.get("type", "synthetic") != "synthetic":
        raise ValueError(f"Unsupported train data type: {train_cfg.get('type')}")

    n = int(train_cfg["n"])
    num_instances = int(train_cfg["num_instances"])
    flow_type = train_cfg.get("flow_type", "uniform_symmetric")
    distance_type = train_cfg.get("distance_type", "euclidean")
    distribution = train_cfg.get("distribution", None)
    flow_sparsity = float(train_cfg.get("flow_sparsity", 0.7))
    symmetrize_mode = str(train_cfg.get("symmetrize_mode", "average"))
    rng = np.random.default_rng(int(train_cfg.get("seed", config.get("seed", 0))))

    return [
        generate_synthetic_qap(
            n=n,
            rng=rng,
            flow_type=flow_type,
            distance_type=distance_type,
            distribution=distribution,
            flow_sparsity=flow_sparsity,
            symmetrize_mode=symmetrize_mode,
            name=f"synthetic_n{n}_{idx:05d}",
        )
        for idx in range(num_instances)
    ]
