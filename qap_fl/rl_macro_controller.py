"""Small on-policy actor-critic controller for SMDP macro actions."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn


class _ActorCritic(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 32, n_actions: int = 2) -> None:
        super().__init__()
        self.body = nn.Sequential(nn.Linear(input_dim, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh())
        self.actor = nn.Linear(hidden, n_actions)
        self.critic = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.body(x)
        return self.actor(h), self.critic(h).squeeze(-1)


@dataclass
class SMDPActorCritic:
    """Online one-step actor-critic over macro-action transitions.

    The controller keeps its parameters between runs for train-instance
    adaptation. ``training=False`` freezes updates for held-out instances.
    """

    hidden: int = 32
    learning_rate: float = 1e-3
    gamma: float = 0.90
    clip_epsilon: float = 0.20
    entropy_weight: float = 0.01
    actions: tuple[str, ...] = ("official_perturb", "tabu_burst")
    training: bool = True

    def __post_init__(self) -> None:
        self.model: _ActorCritic | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.feature_names: tuple[str, ...] | None = None
        self.n_updates = 0
        self.n_switches = 0
        self.n_exploration = 0
        self._reset_episode()

    def _reset_episode(self) -> None:
        self._previous_state: np.ndarray | None = None
        self._previous_action: int | None = None
        self._previous_logprob: torch.Tensor | None = None
        self._previous_value: torch.Tensor | None = None
        self._previous_best = 0.0
        self._previous_time = 0.0
        self._cost_scale = 1.0

    def _vector(self, features: dict[str, float] | None) -> np.ndarray:
        values = dict(features or {})
        if self.feature_names is None:
            self.feature_names = tuple(sorted(values))
        return np.asarray([float(values.get(name, 0.0)) for name in self.feature_names], dtype=np.float32)

    def _ensure_model(self, input_dim: int) -> None:
        if self.model is None:
            self.model = _ActorCritic(input_dim, self.hidden, len(self.actions))
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def start_run(self, *, initial_cost: float, best_cost: float, started_at: float, **_: object) -> None:
        self._reset_episode()
        self._cost_scale = max(abs(float(initial_cost)), 1.0)
        self._previous_best = float(best_cost)
        self._previous_time = float(started_at)

    def _update(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        if not self.training or self.model is None or self.optimizer is None:
            return
        current = torch.from_numpy(state)
        nxt = torch.from_numpy(next_state)
        logits, value = self.model(current)
        with torch.no_grad():
            _, next_value = self.model(nxt)
        logprob = torch.log_softmax(logits, dim=-1)[int(action)]
        target = torch.tensor(float(reward), dtype=torch.float32) + (0.0 if done else self.gamma * next_value)
        advantage = (target - value).detach()
        policy_loss = -logprob * advantage
        value_loss = 0.5 * (value - target.detach()) ** 2
        entropy = -(torch.softmax(logits, dim=-1) * torch.log_softmax(logits, dim=-1)).sum()
        loss = policy_loss + value_loss - self.entropy_weight * entropy
        self.optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0); self.optimizer.step()
        self.n_updates += 1

    def select_action(self, *, best_cost: float, now: float, state_features: dict[str, float] | None = None, **_: object) -> str:
        state = self._vector(state_features)
        self._ensure_model(len(state))
        if self._previous_state is not None and self._previous_action is not None:
            elapsed = max(float(now) - self._previous_time, 1e-3)
            gain = max(self._previous_best - float(best_cost), 0.0) / self._cost_scale
            reward = min(gain / elapsed, 10.0)
            self._update(self._previous_state, self._previous_action, reward, state, False)
        assert self.model is not None
        with torch.no_grad():
            logits, value = self.model(torch.from_numpy(state))
            distribution = torch.distributions.Categorical(logits=logits)
            action = int(distribution.sample().item()) if self.training else int(torch.argmax(logits).item())
            logprob = distribution.log_prob(torch.tensor(action))
        if self._previous_action is not None and action != self._previous_action:
            self.n_switches += 1
        if self.training:
            self.n_exploration += int(action != int(torch.argmax(logits).item()))
        self._previous_state = state
        self._previous_action = action
        self._previous_logprob = logprob
        self._previous_value = value
        self._previous_best = float(best_cost)
        self._previous_time = float(now)
        return self.actions[action]

    def finish_run(self, *, best_cost: float, now: float, **_: object) -> None:
        if self._previous_state is not None and self._previous_action is not None:
            elapsed = max(float(now) - self._previous_time, 1e-3)
            gain = max(self._previous_best - float(best_cost), 0.0) / self._cost_scale
            self._update(self._previous_state, self._previous_action, min(gain / elapsed, 10.0), self._previous_state, True)

    def freeze(self) -> None:
        self.training = False

    def unfreeze(self) -> None:
        self.training = True


__all__ = ["SMDPActorCritic"]
