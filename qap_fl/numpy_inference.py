from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch.nn as nn


@dataclass(frozen=True)
class NumpyMLP:
    """CPU inference for a trained sequential MLP with folded normalization."""

    weights: tuple[np.ndarray, ...]
    biases: tuple[np.ndarray, ...]

    @classmethod
    def from_torch(
        cls,
        model: nn.Module,
        feature_mean: np.ndarray,
        feature_std: np.ndarray,
    ) -> NumpyMLP:
        sequential = getattr(model, "net", None)
        if not isinstance(sequential, nn.Sequential):
            raise TypeError("NumpyMLP requires a model with an nn.Sequential 'net'")
        linear_layers: list[nn.Linear] = []
        for layer in sequential:
            if isinstance(layer, nn.Linear):
                linear_layers.append(layer)
            elif not isinstance(layer, (nn.ReLU, nn.Dropout)):
                raise TypeError(f"unsupported MLP layer: {type(layer).__name__}")
        if not linear_layers:
            raise ValueError("model contains no linear layers")

        weights = [
            np.ascontiguousarray(layer.weight.detach().cpu().numpy(), dtype=np.float32)
            for layer in linear_layers
        ]
        biases = [
            np.ascontiguousarray(layer.bias.detach().cpu().numpy(), dtype=np.float32)
            for layer in linear_layers
        ]
        mean = np.asarray(feature_mean, dtype=np.float32)
        std = np.asarray(feature_std, dtype=np.float32)
        if mean.shape != (weights[0].shape[1],) or std.shape != mean.shape:
            raise ValueError("normalization statistics do not match model input width")
        if np.any(std == 0.0):
            raise ValueError("feature_std must be nonzero")

        first_weight = weights[0]
        weights[0] = np.ascontiguousarray(first_weight / std[None, :], dtype=np.float32)
        biases[0] = np.ascontiguousarray(
            biases[0] - first_weight @ (mean / std), dtype=np.float32
        )
        return cls(weights=tuple(weights), biases=tuple(biases))

    def __call__(self, raw_features: np.ndarray) -> np.ndarray:
        values = np.ascontiguousarray(raw_features, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != self.weights[0].shape[1]:
            raise ValueError("raw_features have the wrong shape")
        for index, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            values = values @ weight.T + bias
            if index + 1 < len(self.weights):
                np.maximum(values, 0.0, out=values)
        return values.reshape(-1).astype(np.float64, copy=False)


__all__ = ["NumpyMLP"]
