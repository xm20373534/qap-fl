from __future__ import annotations

import torch
import torch.nn as nn


class SwapMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, n_layers: int = 3, dropout: float = 0.0) -> None:
        super().__init__()
        if n_layers < 1:
            raise ValueError("n_layers must be at least 1.")
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.n_layers = int(n_layers)
        self.dropout = float(dropout)
        layers: list[nn.Module] = []
        dim = self.input_dim
        for _ in range(self.n_layers - 1):
            layers.append(nn.Linear(dim, self.hidden_dim))
            layers.append(nn.ReLU())
            if self.dropout > 0:
                layers.append(nn.Dropout(self.dropout))
            dim = self.hidden_dim
        layers.append(nn.Linear(dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim == 1:
            features = features.unsqueeze(0)
        if features.ndim != 2:
            raise ValueError("features must have shape [num_pairs, input_dim].")
        if features.shape[-1] != self.input_dim:
            raise ValueError(f"expected input_dim={self.input_dim}, got {features.shape[-1]}.")
        return self.net(features).squeeze(-1)


class SetContextSwapMLP(nn.Module):
    """Score every swap after pooling the complete candidate set context."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        embedding_dim: int = 64,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.embedding_dim = int(embedding_dim)
        self.dropout = float(dropout)
        self.token_encoder = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout) if self.dropout > 0 else nn.Identity(),
            nn.Linear(self.hidden_dim, self.embedding_dim),
            nn.ReLU(),
        )
        # Local token, mean/max pool, and local deviation from the pool mean.
        self.score_head = nn.Sequential(
            nn.Linear(4 * self.embedding_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout) if self.dropout > 0 else nn.Identity(),
            nn.Linear(self.hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.shape[-1] != self.input_dim:
            raise ValueError(f"expected features [num_pairs, {self.input_dim}].")
        if features.shape[0] == 0:
            return features.new_zeros((0,))
        encoded = self.token_encoder(features)
        mean = encoded.mean(dim=0, keepdim=True)
        maximum = encoded.max(dim=0, keepdim=True).values
        context = torch.cat(
            [
                encoded,
                mean.expand_as(encoded),
                maximum.expand_as(encoded),
                encoded - mean,
            ],
            dim=-1,
        )
        return self.score_head(context).squeeze(-1)


class EdgeGatedMessagePassing(nn.Module):
    def __init__(self, embedding_dim: int, edge_dim: int) -> None:
        super().__init__()
        self.source_gate = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.target_gate = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.edge_gate = nn.Linear(edge_dim, embedding_dim)
        self.value = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.edge_value = nn.Linear(edge_dim, embedding_dim, bias=False)
        self.output = nn.Linear(embedding_dim, embedding_dim)
        self.norm = nn.LayerNorm(embedding_dim)

    def forward(self, nodes: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
        if edges.shape[:2] != (len(nodes), len(nodes)):
            raise ValueError("edges must have shape [n, n, edge_dim]")
        source = self.source_gate(nodes)[:, None, :]
        target = self.target_gate(nodes)[None, :, :]
        gate = torch.sigmoid(source + target + self.edge_gate(edges))
        values = self.value(nodes)[None, :, :] + self.edge_value(edges)
        diagonal = torch.eye(len(nodes), device=nodes.device, dtype=torch.bool)[:, :, None]
        gate = gate.masked_fill(diagonal, 0.0)
        aggregate = (gate * values).sum(dim=1) / gate.sum(dim=1).clamp_min(1e-6)
        return self.norm(nodes + self.output(aggregate)).relu()


class QAPRelationalSwapScorer(nn.Module):
    """Condition swap scores on the full assignment-induced interaction graph."""

    def __init__(
        self,
        node_input_dim: int,
        edge_input_dim: int,
        pair_input_dim: int,
        hidden_dim: int = 128,
        embedding_dim: int = 64,
        message_layers: int = 2,
    ) -> None:
        super().__init__()
        self.node_input_dim = int(node_input_dim)
        self.edge_input_dim = int(edge_input_dim)
        self.pair_input_dim = int(pair_input_dim)
        self.hidden_dim = int(hidden_dim)
        self.embedding_dim = int(embedding_dim)
        self.message_layers = int(message_layers)
        self.node_encoder = nn.Sequential(
            nn.Linear(self.node_input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.embedding_dim),
            nn.ReLU(),
        )
        self.layers = nn.ModuleList(
            EdgeGatedMessagePassing(self.embedding_dim, self.edge_input_dim)
            for _ in range(self.message_layers)
        )
        self.pair_encoder = nn.Sequential(
            nn.Linear(self.pair_input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.embedding_dim),
            nn.ReLU(),
        )
        local_dim = 4 * self.embedding_dim
        self.score_head = nn.Sequential(
            nn.Linear(3 * local_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1),
        )

    def forward(
        self,
        node_features: torch.Tensor,
        edge_features: torch.Tensor,
        pairs: torch.Tensor,
        pair_features: torch.Tensor,
    ) -> torch.Tensor:
        if node_features.ndim != 2 or node_features.shape[1] != self.node_input_dim:
            raise ValueError("invalid node feature shape")
        if edge_features.shape != (len(node_features), len(node_features), self.edge_input_dim):
            raise ValueError("invalid edge feature shape")
        if pairs.ndim != 2 or pairs.shape[1] != 2 or len(pairs) != len(pair_features):
            raise ValueError("invalid swap pair shape")
        nodes = self.node_encoder(node_features)
        for layer in self.layers:
            nodes = layer(nodes, edge_features)
        left = nodes.index_select(0, pairs[:, 0])
        right = nodes.index_select(0, pairs[:, 1])
        pair_tokens = self.pair_encoder(pair_features)
        local = torch.cat([left + right, torch.abs(left - right), left * right, pair_tokens], dim=-1)
        mean = local.mean(dim=0, keepdim=True).expand_as(local)
        maximum = local.max(dim=0, keepdim=True).values.expand_as(local)
        return self.score_head(torch.cat([local, mean, maximum], dim=-1)).squeeze(-1)


class RelationalSwapMLP(nn.Module):
    """Score a swap with shared endpoint encoding and symmetric aggregation."""

    def __init__(
        self,
        input_dim: int,
        endpoint_pairs: list[list[int]],
        hidden_dim: int = 128,
        endpoint_hidden_dim: int = 64,
        n_layers: int = 3,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if n_layers < 2:
            raise ValueError("relational n_layers must be at least 2.")
        pairs = torch.as_tensor(endpoint_pairs, dtype=torch.long)
        if pairs.ndim != 2 or pairs.shape[1] != 2 or len(pairs) == 0:
            raise ValueError("endpoint_pairs must have shape [endpoint_dim, 2].")
        used = set(int(x) for x in pairs.flatten().tolist())
        if min(used) < 0 or max(used) >= int(input_dim):
            raise ValueError("endpoint_pairs contain an out-of-range feature index.")
        context_indices = [index for index in range(int(input_dim)) if index not in used]
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.endpoint_hidden_dim = int(endpoint_hidden_dim)
        self.n_layers = int(n_layers)
        self.dropout = float(dropout)
        self.endpoint_pairs = [[int(x) for x in row] for row in endpoint_pairs]
        self.register_buffer("left_indices", pairs[:, 0])
        self.register_buffer("right_indices", pairs[:, 1])
        self.register_buffer("context_indices", torch.as_tensor(context_indices, dtype=torch.long))

        endpoint_layers: list[nn.Module] = [
            nn.Linear(len(endpoint_pairs), self.endpoint_hidden_dim),
            nn.ReLU(),
        ]
        if self.dropout > 0:
            endpoint_layers.append(nn.Dropout(self.dropout))
        self.endpoint_encoder = nn.Sequential(*endpoint_layers)

        pair_width = 2 * self.endpoint_hidden_dim + len(context_indices)
        head: list[nn.Module] = []
        for _ in range(self.n_layers - 2):
            head.append(nn.Linear(pair_width, self.hidden_dim))
            head.append(nn.ReLU())
            if self.dropout > 0:
                head.append(nn.Dropout(self.dropout))
            pair_width = self.hidden_dim
        head.append(nn.Linear(pair_width, 1))
        self.pair_head = nn.Sequential(*head)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim == 1:
            features = features.unsqueeze(0)
        if features.ndim != 2 or features.shape[-1] != self.input_dim:
            raise ValueError(f"expected features [num_pairs, {self.input_dim}].")
        left = self.endpoint_encoder(features.index_select(1, self.left_indices))
        right = self.endpoint_encoder(features.index_select(1, self.right_indices))
        symmetric = [left + right, torch.abs(left - right)]
        if self.context_indices.numel():
            symmetric.append(features.index_select(1, self.context_indices))
        return self.pair_head(torch.cat(symmetric, dim=1)).squeeze(-1)
