"""Low-overhead online macro-action control with delayed window rewards."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .macro_state_features import decayed_macro_history_features


EPS = 1e-12


@dataclass
class WindowedPrimalBandit:
    """Choose one macro action for a window of local-optimum closures.

    Each time segment has independent UCB statistics.  A window reward is the
    time-average area between its starting incumbent and the incumbent curve,
    normalized by the run's initial cost.  The statistic therefore rewards
    improvements that occur early without requiring an optimum value.
    """

    window_size: int = 8
    n_segments: int = 4
    exploration: float = 0.5
    actions: tuple[str, ...] = ("official_perturb", "tabu_burst")
    counts: np.ndarray = field(init=False, repr=False)
    reward_sums: np.ndarray = field(init=False, repr=False)
    reward_squares: np.ndarray = field(init=False, repr=False)
    completed_rewards: list[float] = field(init=False, default_factory=list)
    completed_endpoint_gains: list[float] = field(init=False, default_factory=list)
    action_history: list[str] = field(init=False, default_factory=list)
    n_updates: int = field(init=False, default=0)
    n_switches: int = field(init=False, default=0)
    history_decay: float = 0.85
    history_limit: int = 10

    def __post_init__(self) -> None:
        if int(self.window_size) <= 0:
            raise ValueError("window_size must be positive")
        if int(self.n_segments) <= 0:
            raise ValueError("n_segments must be positive")
        if len(self.actions) < 2:
            raise ValueError("at least two actions are required")
        if float(self.exploration) < 0.0:
            raise ValueError("exploration must be non-negative")
        if not 0.0 < float(self.history_decay) <= 1.0:
            raise ValueError("history_decay must be in (0, 1]")
        if int(self.history_limit) <= 0:
            raise ValueError("history_limit must be positive")
        shape = (int(self.n_segments), len(self.actions))
        self.counts = np.zeros(shape, dtype=np.int64)
        self.reward_sums = np.zeros(shape, dtype=np.float64)
        self.reward_squares = np.zeros(shape, dtype=np.float64)
        self._reset_run_state()

    def _reset_run_state(self) -> None:
        self.counts.fill(0)
        self.reward_sums.fill(0.0)
        self.reward_squares.fill(0.0)
        self.completed_rewards = []
        self.completed_endpoint_gains = []
        self.action_history = []
        self.n_updates = 0
        self.n_switches = 0
        self.recent_utilities: list[tuple[str, float]] = []
        self._started_at: float | None = None
        self._time_budget_sec: float | None = None
        self._cost_scale = 1.0
        self._current_action_index: int | None = None
        self._current_segment = 0
        self._window_start_time = 0.0
        self._window_start_best = 0.0
        self._last_observation_time = 0.0
        self._last_best = 0.0
        self._window_area = 0.0
        self._window_closures = 0

    def start_run(
        self,
        *,
        initial_cost: float,
        best_cost: float,
        started_at: float,
        max_time_sec: float | None,
    ) -> None:
        self._reset_run_state()
        self._started_at = float(started_at)
        self._time_budget_sec = None if max_time_sec is None else float(max_time_sec)
        self._cost_scale = max(abs(float(initial_cost)), 1.0)
        self._window_start_time = float(started_at)
        self._window_start_best = float(best_cost)
        self._last_observation_time = float(started_at)
        self._last_best = float(best_cost)

    def _segment_index(self, elapsed_fraction: float | None) -> int:
        if elapsed_fraction is None:
            return 0
        clipped = min(max(float(elapsed_fraction), 0.0), 1.0 - EPS)
        return min(int(clipped * int(self.n_segments)), int(self.n_segments) - 1)

    def _observe(self, *, best_cost: float, now: float) -> None:
        if self._current_action_index is None:
            self._last_observation_time = float(now)
            self._last_best = float(best_cost)
            return
        duration = max(float(now) - self._last_observation_time, 0.0)
        previous_gain = max(
            (self._window_start_best - self._last_best) / self._cost_scale,
            0.0,
        )
        current_gain = max(
            (self._window_start_best - float(best_cost)) / self._cost_scale,
            0.0,
        )
        self._window_area += 0.5 * (previous_gain + current_gain) * duration
        self._last_observation_time = float(now)
        self._last_best = float(best_cost)
        self._window_closures += 1

    def _finish_window(self, *, now: float, best_cost: float) -> None:
        if self._current_action_index is None or self._window_closures <= 0:
            return
        duration = max(float(now) - self._window_start_time, EPS)
        reward = self._window_area / duration
        endpoint_gain = max(
            (self._window_start_best - float(best_cost)) / self._cost_scale,
            0.0,
        )
        segment = int(self._current_segment)
        action = int(self._current_action_index)
        self.counts[segment, action] += 1
        self.reward_sums[segment, action] += reward
        self.reward_squares[segment, action] += reward * reward
        self.completed_rewards.append(float(reward))
        self.completed_endpoint_gains.append(float(endpoint_gain))
        self.n_updates += 1
        self.recent_utilities.append((self.actions[action], float(reward)))
        del self.recent_utilities[:-int(self.history_limit)]

    def _start_window(
        self,
        *,
        action_index: int,
        segment: int,
        best_cost: float,
        now: float,
    ) -> None:
        previous = self._current_action_index
        self._current_action_index = int(action_index)
        self._current_segment = int(segment)
        self._window_start_time = float(now)
        self._window_start_best = float(best_cost)
        self._last_observation_time = float(now)
        self._last_best = float(best_cost)
        self._window_area = 0.0
        self._window_closures = 0
        if previous is not None and previous != self._current_action_index:
            self.n_switches += 1
        self.action_history.append(self.actions[self._current_action_index])

    def _choose_action(self, segment: int) -> int:
        counts = self.counts[int(segment)]
        untried = np.flatnonzero(counts == 0)
        if len(untried):
            # Rotate bootstrap order across time segments to reduce stage bias.
            preferred = int(segment) % len(self.actions)
            if preferred in untried:
                return preferred
            return int(untried[0])

        means = self.reward_sums[int(segment)] / counts
        observed = []
        for action_index in range(len(self.actions)):
            count = int(counts[action_index])
            mean = float(means[action_index])
            variance = max(
                float(self.reward_squares[int(segment), action_index]) / count - mean * mean,
                0.0,
            )
            observed.extend([mean - np.sqrt(variance), mean + np.sqrt(variance)])
        reward_range = max(float(max(observed) - min(observed)), 1e-6)
        total = int(np.sum(counts))
        bonus = (
            float(self.exploration)
            * reward_range
            * np.sqrt(np.log(float(total) + 1.0) / counts.astype(np.float64))
        )
        scores = means + bonus
        return int(np.argmax(scores))

    def select_action(
        self,
        *,
        best_cost: float,
        now: float,
        elapsed_fraction: float | None = None,
        **_: object,
    ) -> str:
        if self._started_at is None:
            raise RuntimeError("start_run must be called before select_action")
        segment = self._segment_index(elapsed_fraction)
        if self._current_action_index is None:
            action_index = self._choose_action(segment)
            self._start_window(
                action_index=action_index,
                segment=segment,
                best_cost=best_cost,
                now=now,
            )
        else:
            self._observe(best_cost=best_cost, now=now)
            if self._window_closures >= int(self.window_size) or segment != self._current_segment:
                self._finish_window(now=now, best_cost=best_cost)
                action_index = self._choose_action(segment)
                self._start_window(
                    action_index=action_index,
                    segment=segment,
                    best_cost=best_cost,
                    now=now,
                )
        return self.actions[int(self._current_action_index)]

    def history_features(self) -> dict[str, float]:
        """Return the deployable decayed utility history for contextual users."""

        return decayed_macro_history_features(
            self.recent_utilities,
            self.actions,
            decay=float(self.history_decay),
            max_events=int(self.history_limit),
        )

    def finish_run(
        self,
        *,
        best_cost: float,
        now: float,
        **_: object,
    ) -> None:
        if self._started_at is None or self._current_action_index is None:
            return
        self._observe(best_cost=best_cost, now=now)
        self._finish_window(now=now, best_cost=best_cost)


class ContextualMacroBandit(WindowedPrimalBandit):
    """A small action-conditional linear-UCB extension.

    The delayed window reward and bootstrap behavior remain identical to
    :class:`WindowedPrimalBandit`; only the action score uses the current
    deployment-visible state.  This keeps the protocol auditable and avoids
    introducing a neural policy before the enriched trajectory data exist.
    """

    _CONTEXT_KEYS = (
        "elapsed_fraction",
        "state_hamming_elite_min",
        "state_hamming_elite_mean",
        "state_hamming_elite_max",
        "state_hamming_elite_q25",
        "state_hamming_elite_q75",
        "state_hamming_elite_count",
        "state_hamming_incumbent",
        "state_basin_depth_min",
        "state_basin_depth_mean",
        "state_basin_depth_max",
        "state_basin_depth_q25",
        "state_basin_depth_q75",
        "state_basin_depth_count",
        "state_basin_depth_decayed",
        "state_basin_depth_last",
        "state_macro_history_count",
        "state_macro_history_decayed_mean",
        "state_macro_history_last_utility",
        "state_macro_history_official_perturb_utility",
        "state_macro_history_official_perturb_count",
        "state_macro_history_tabu_burst_utility",
        "state_macro_history_tabu_burst_count",
    )

    def __init__(
        self,
        *args: object,
        context_ridge: float = 1.0,
        context_exploration: float = 0.25,
        **kwargs: object,
    ) -> None:
        super().__init__(*args, **kwargs)
        if float(context_ridge) <= 0.0:
            raise ValueError("context_ridge must be positive")
        if float(context_exploration) < 0.0:
            raise ValueError("context_exploration must be non-negative")
        self.context_ridge = float(context_ridge)
        self.context_exploration = float(context_exploration)
        self._context_dim = len(self._CONTEXT_KEYS) + 1
        self._context_A = np.tile(
            np.eye(self._context_dim, dtype=np.float64)[None, :, :],
            (len(self.actions), 1, 1),
        ) * self.context_ridge
        self._context_b = np.zeros((len(self.actions), self._context_dim), dtype=np.float64)
        self._context_A_inv = np.tile(
            np.eye(self._context_dim, dtype=np.float64)[None, :, :],
            (len(self.actions), 1, 1),
        ) / self.context_ridge
        self._window_context = np.zeros(self._context_dim, dtype=np.float64)
        self._pending_context = np.zeros(self._context_dim, dtype=np.float64)

    def start_run(self, **kwargs: object) -> None:
        super().start_run(**kwargs)
        self._context_A[:] = np.eye(self._context_dim, dtype=np.float64) * self.context_ridge
        self._context_b.fill(0.0)
        self._context_A_inv[:] = (
            np.eye(self._context_dim, dtype=np.float64) / self.context_ridge
        )
        self._window_context.fill(0.0)
        self._pending_context.fill(0.0)

    def _context_vector(
        self,
        state_features: dict[str, float] | None,
        elapsed_fraction: float | None,
    ) -> np.ndarray:
        state = state_features or {}
        values = [float(state.get(key, 0.0)) for key in self._CONTEXT_KEYS]
        values.append(1.0)
        if elapsed_fraction is not None:
            values[0] = min(max(float(elapsed_fraction), 0.0), 1.0)
        return np.asarray(values, dtype=np.float64)

    def _choose_action(self, segment: int) -> int:
        counts = self.counts[int(segment)]
        untried = np.flatnonzero(counts == 0)
        if len(untried):
            return super()._choose_action(segment)
        x = self._pending_context
        scores = []
        for action_index in range(len(self.actions)):
            inverse = self._context_A_inv[action_index]
            theta = inverse @ self._context_b[action_index]
            mean = float(x @ theta)
            uncertainty = float(np.sqrt(max(x @ inverse @ x, 0.0)))
            scores.append(mean + self.context_exploration * uncertainty)
        return int(np.argmax(scores))

    def _start_window(self, **kwargs: object) -> None:
        self._window_context = self._pending_context.copy()
        super()._start_window(**kwargs)

    def _finish_window(self, *, now: float, best_cost: float) -> None:
        action = self._current_action_index
        context = self._window_context.copy()
        before = len(self.completed_rewards)
        super()._finish_window(now=now, best_cost=best_cost)
        if action is None or len(self.completed_rewards) == before:
            return
        reward = float(self.completed_rewards[-1])
        decay = float(self.history_decay)
        self._context_A[int(action)] *= decay
        self._context_b[int(action)] *= decay
        self._context_A[int(action)] += np.outer(context, context)
        self._context_b[int(action)] += context * reward
        inverse = self._context_A_inv[int(action)]
        inverse_context = inverse @ context
        denominator = decay + float(context @ inverse_context)
        self._context_A_inv[int(action)] = (
            inverse - np.outer(inverse_context, inverse_context) / denominator
        ) / decay

    def select_action(
        self,
        *,
        elapsed_fraction: float | None = None,
        state_features: dict[str, float] | None = None,
        **kwargs: object,
    ) -> str:
        history = self.history_features()
        merged = dict(history)
        if state_features:
            merged.update(state_features)
        self._pending_context = self._context_vector(merged, elapsed_fraction)
        return super().select_action(
            elapsed_fraction=elapsed_fraction,
            state_features=merged,
            **kwargs,
        )


__all__ = ["ContextualMacroBandit", "WindowedPrimalBandit"]
