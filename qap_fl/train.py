from __future__ import annotations

import csv
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as torch_f

from .global_features import GLOBAL_FEATURE_FIELDS
from .model import RelationalSwapMLP, SwapMLP
from .utils import ensure_dir


@dataclass(frozen=True)
class TrainingGroup:
    key: tuple[str, str]
    features: np.ndarray
    labels: np.ndarray


RELATIONAL_ENDPOINT_FIELD_PAIRS = [
    ("feature_facility_sum_i", "feature_facility_sum_j"),
    ("feature_facility_mean_i", "feature_facility_mean_j"),
    ("feature_facility_std_i", "feature_facility_std_j"),
    ("feature_location_sum_pi_i", "feature_location_sum_pi_j"),
    ("feature_location_mean_pi_i", "feature_location_mean_pi_j"),
    ("feature_location_std_pi_i", "feature_location_std_pi_j"),
    ("feature_node_contrib_i", "feature_node_contrib_j"),
]
RELATIONAL_INDEX_FEATURE_FIELDS = [
    "feature_facility_index_distance",
    "feature_location_assignment_distance",
    "feature_assignment_index_i",
    "feature_assignment_index_j",
    "feature_assignment_index_absdiff",
]
RELATIONAL_ANCHOR_FEATURE_FIELDS = [
    "feature_anchor_external_old",
    "feature_anchor_external_swap",
    "feature_anchor_external_delta_estimate",
    "feature_anchor_delta_mean",
    "feature_anchor_delta_std",
    "feature_anchor_delta_min",
    "feature_anchor_delta_max",
]
RELATIONAL_EXCLUDED_FEATURE_FIELDS = [
    *RELATIONAL_INDEX_FEATURE_FIELDS,
    *GLOBAL_FEATURE_FIELDS,
]
RELATIONAL_STRICT_EXCLUDED_FEATURE_FIELDS = [
    *RELATIONAL_INDEX_FEATURE_FIELDS,
    *RELATIONAL_ANCHOR_FEATURE_FIELDS,
    *GLOBAL_FEATURE_FIELDS,
]


def _normalize_labels(labels: np.ndarray) -> np.ndarray:
    total = float(np.sum(labels))
    if not np.isfinite(total) or total <= 0:
        return np.full(len(labels), 1.0 / len(labels), dtype=np.float32)
    return (labels / total).astype(np.float32)


def load_groups(
    path: str | Path,
    feature_fields: list[str] | None = None,
    exclude_feature_fields: list[str] | None = None,
) -> tuple[list[TrainingGroup], list[str]]:
    groups: OrderedDict[tuple[str, str], tuple[list[list[float]], list[float]]] = OrderedDict()
    with Path(path).open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no header.")
        available_feature_fields = [field for field in reader.fieldnames if field.startswith("feature_")]
        if feature_fields is None:
            selected_feature_fields = list(available_feature_fields)
        else:
            missing = sorted(set(feature_fields) - set(available_feature_fields))
            if missing:
                raise ValueError(f"requested feature fields are missing from {path}: {missing}")
            selected_feature_fields = list(feature_fields)
        if exclude_feature_fields:
            excluded = set(exclude_feature_fields)
            selected_feature_fields = [field for field in selected_feature_fields if field not in excluded]
        if not selected_feature_fields:
            raise ValueError("no feature fields selected for training.")
        for row in reader:
            key = (row["instance"], row["state_id"])
            if key not in groups:
                groups[key] = ([], [])
            feature_rows, labels = groups[key]
            feature_rows.append([float(row[field]) for field in selected_feature_fields])
            labels.append(float(row["label"]))
    out = [
        TrainingGroup(
            key=key,
            features=np.asarray(feature_rows, dtype=np.float32),
            labels=_normalize_labels(np.asarray(labels, dtype=np.float32)),
        )
        for key, (feature_rows, labels) in groups.items()
    ]
    return out, selected_feature_fields


def split_groups(groups: list[TrainingGroup], val_fraction: float, seed: int) -> tuple[list[TrainingGroup], list[TrainingGroup]]:
    idx = np.arange(len(groups))
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    n_val = int(round(len(groups) * float(val_fraction)))
    if val_fraction > 0 and len(groups) > 1:
        n_val = max(1, min(len(groups) - 1, n_val))
    val_idx = {int(i) for i in idx[:n_val]}
    train = [group for i, group in enumerate(groups) if i not in val_idx]
    val = [group for i, group in enumerate(groups) if i in val_idx]
    return train, val


def split_groups_by_instance(
    groups: list[TrainingGroup], val_fraction: float, seed: int,
) -> tuple[list[TrainingGroup], list[TrainingGroup]]:
    instances = sorted({group.key[0] for group in groups})
    rng = np.random.default_rng(seed)
    rng.shuffle(instances)
    n_val = int(round(len(instances) * float(val_fraction)))
    if val_fraction > 0 and len(instances) > 1:
        n_val = max(1, min(len(instances) - 1, n_val))
    validation = set(instances[:n_val])
    train = [group for group in groups if group.key[0] not in validation]
    val = [group for group in groups if group.key[0] in validation]
    return train, val


def relational_endpoint_pairs(feature_fields: list[str]) -> list[list[int]]:
    index = {field: position for position, field in enumerate(feature_fields)}
    missing = [
        field for pair in RELATIONAL_ENDPOINT_FIELD_PAIRS for field in pair
        if field not in index
    ]
    if missing:
        raise ValueError(f"relational endpoint fields are missing: {missing}")
    return [[index[left], index[right]] for left, right in RELATIONAL_ENDPOINT_FIELD_PAIRS]


def feature_normalizer(groups: list[TrainingGroup]) -> tuple[np.ndarray, np.ndarray]:
    features = np.concatenate([group.features for group in groups], axis=0)
    mean = features.mean(axis=0).astype(np.float32)
    std = features.std(axis=0).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    return mean, std


def relational_feature_normalizer(
    groups: list[TrainingGroup], endpoint_pairs: list[list[int]],
) -> tuple[np.ndarray, np.ndarray]:
    features = np.concatenate([group.features for group in groups], axis=0)
    mean = features.mean(axis=0).astype(np.float32)
    std = features.std(axis=0).astype(np.float32)
    for left, right in endpoint_pairs:
        endpoint_values = features[:, [left, right]].reshape(-1)
        shared_mean = np.float32(endpoint_values.mean())
        shared_std = np.float32(endpoint_values.std())
        mean[[left, right]] = shared_mean
        std[[left, right]] = shared_std
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    return mean, std


def group_loss(model: torch.nn.Module, group: TrainingGroup, mean: np.ndarray, std: np.ndarray, device: torch.device) -> torch.Tensor:
    features = (group.features - mean) / std
    x = torch.from_numpy(features).to(device=device, dtype=torch.float32)
    target = torch.from_numpy(group.labels).to(device=device, dtype=torch.float32)
    scores = model(x)
    positive = target > 0
    if bool(torch.all(positive)):
        positive = target >= target.max()
    negative = ~positive
    if not bool(torch.any(positive)) or not bool(torch.any(negative)):
        return scores.sum() * 0.0
    diff = scores[positive][:, None] - scores[negative][None, :]
    return torch_f.softplus(-diff).mean()


def evaluate_loss(model: torch.nn.Module, groups: list[TrainingGroup], mean: np.ndarray, std: np.ndarray, device: torch.device) -> float | None:
    if not groups:
        return None
    model.eval()
    values = []
    with torch.no_grad():
        for group in groups:
            values.append(float(group_loss(model, group, mean, std, device).detach().cpu()))
    return float(np.mean(values))


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    feature_fields: list[str],
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    train_args: dict[str, Any],
    history: list[dict[str, Any]],
) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": int(epoch),
            "model_type": "relational" if isinstance(model, RelationalSwapMLP) else "mlp",
            "model_config": {
                "input_dim": model.input_dim,
                "hidden_dim": model.hidden_dim,
                "n_layers": model.n_layers,
                "dropout": model.dropout,
                **(
                    {
                        "endpoint_hidden_dim": model.endpoint_hidden_dim,
                        "endpoint_pairs": model.endpoint_pairs,
                    }
                    if isinstance(model, RelationalSwapMLP)
                    else {}
                ),
            },
            "feature_fields": feature_fields,
            "feature_mean": feature_mean,
            "feature_std": feature_std,
            "train_args": train_args,
            "history": history,
        },
        path,
    )


def load_checkpoint(path: str | Path, device: torch.device) -> tuple[torch.nn.Module, list[str], np.ndarray, np.ndarray, dict[str, Any]]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    cfg = checkpoint["model_config"]
    if checkpoint.get("model_type", "mlp") == "relational":
        model = RelationalSwapMLP(
            input_dim=int(cfg["input_dim"]),
            endpoint_pairs=cfg["endpoint_pairs"],
            hidden_dim=int(cfg["hidden_dim"]),
            endpoint_hidden_dim=int(cfg["endpoint_hidden_dim"]),
            n_layers=int(cfg["n_layers"]),
            dropout=float(cfg["dropout"]),
        ).to(device)
    else:
        model = SwapMLP(
            input_dim=int(cfg["input_dim"]),
            hidden_dim=int(cfg["hidden_dim"]),
            n_layers=int(cfg["n_layers"]),
            dropout=float(cfg["dropout"]),
        ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return (
        model,
        list(checkpoint["feature_fields"]),
        np.asarray(checkpoint["feature_mean"], dtype=np.float32),
        np.asarray(checkpoint["feature_std"], dtype=np.float32),
        checkpoint,
    )


def train_selector(
    dataset_path: str | Path,
    out_path: str | Path,
    best_out_path: str | Path | None,
    epochs: int,
    batch_size: int,
    hidden_dim: int,
    n_layers: int,
    dropout: float,
    lr: float,
    val_fraction: float,
    seed: int,
    device: torch.device,
    train_args: dict[str, Any],
    feature_fields: list[str] | None = None,
    exclude_feature_fields: list[str] | None = None,
    architecture: str = "mlp",
    split_by_instance: bool = False,
) -> list[dict[str, Any]]:
    groups, feature_fields = load_groups(
        dataset_path,
        feature_fields=feature_fields,
        exclude_feature_fields=exclude_feature_fields,
    )
    if architecture not in {"mlp", "relational"}:
        raise ValueError(f"unsupported architecture: {architecture}")
    train_groups, val_groups = (
        split_groups_by_instance(groups, val_fraction, seed)
        if split_by_instance
        else split_groups(groups, val_fraction, seed)
    )
    if architecture == "relational":
        endpoint_pairs = relational_endpoint_pairs(feature_fields)
        mean, std = relational_feature_normalizer(train_groups, endpoint_pairs)
        model = RelationalSwapMLP(
            len(feature_fields),
            endpoint_pairs=endpoint_pairs,
            hidden_dim=hidden_dim,
            endpoint_hidden_dim=max(int(hidden_dim) // 2, 1),
            n_layers=n_layers,
            dropout=dropout,
        ).to(device)
    else:
        mean, std = feature_normalizer(train_groups)
        model = SwapMLP(
            len(feature_fields), hidden_dim=hidden_dim,
            n_layers=n_layers, dropout=dropout,
        ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    history: list[dict[str, Any]] = []
    best_metric = float("inf")
    best_epoch = 0
    for epoch in range(int(epochs)):
        model.train()
        idx = np.arange(len(train_groups))
        rng.shuffle(idx)
        total = 0.0
        count = 0
        for start in range(0, len(idx), int(batch_size)):
            batch = idx[start : start + int(batch_size)]
            losses = [group_loss(model, train_groups[int(i)], mean, std, device) for i in batch]
            loss = torch.stack(losses).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss.detach().cpu()) * len(batch)
            count += len(batch)
        train_loss = total / max(count, 1)
        val_loss = evaluate_loss(model, val_groups, mean, std, device)
        metric = train_loss if val_loss is None else val_loss
        row = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "n_train_groups": len(train_groups),
            "n_val_groups": len(val_groups),
        }
        history.append(row)
        if best_out_path is not None and metric < best_metric:
            best_metric = float(metric)
            best_epoch = epoch + 1
            save_checkpoint(
                best_out_path,
                model,
                optimizer,
                epoch + 1,
                feature_fields,
                mean,
                std,
                {**train_args, "best_metric": best_metric, "best_epoch": best_epoch},
                history,
            )
        print(
            f"epoch={epoch + 1} train_loss={train_loss:.4f} "
            f"val_loss={'none' if val_loss is None else f'{val_loss:.4f}'} "
            f"best={'none' if best_out_path is None else f'{best_metric:.4f}@{best_epoch}'}"
        )
    save_checkpoint(out_path, model, optimizer, int(epochs), feature_fields, mean, std, train_args, history)
    return history



