from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .qap import QAPInstance, compute_cost, is_valid_perm


EPS = 1e-9
CONSTRUCTION_FEATURE_NAMES = [
    "depth_fraction",
    "remaining_fraction",
    "partial_cost",
    "facility_abs_strength",
    "facility_signed_strength",
    "facility_mean",
    "facility_std",
    "facility_nonzero_fraction",
    "facility_strength_rank",
    "location_strength",
    "location_mean",
    "location_std",
    "location_nonzero_fraction",
    "location_centrality_rank",
    "strength_centrality_product",
    "strength_rank_match",
    "assigned_increment",
    "assigned_increment_mean",
    "assigned_increment_min",
    "assigned_increment_max",
    "remaining_flow_sum",
    "remaining_flow_mean",
    "remaining_flow_std",
    "remaining_distance_sum",
    "remaining_distance_mean",
    "remaining_distance_std",
    "remaining_rearrangement_low",
    "remaining_rearrangement_high",
    "instance_log_n",
]


def _average_fractional_rank(values: np.ndarray, *, descending: bool) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(-values if descending else values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks / max(len(values) - 1, 1)


@dataclass(frozen=True)
class ConstructionCache:
    F: np.ndarray
    D: np.ndarray
    order: np.ndarray
    flow_scale: float
    distance_scale: float
    objective_scale: float
    facility_abs_strength: np.ndarray
    facility_signed_strength: np.ndarray
    facility_mean: np.ndarray
    facility_std: np.ndarray
    facility_nonzero: np.ndarray
    facility_rank: np.ndarray
    location_strength: np.ndarray
    location_mean: np.ndarray
    location_std: np.ndarray
    location_nonzero: np.ndarray
    location_rank: np.ndarray


def canonical_node_colors(matrix: np.ndarray) -> np.ndarray:
    """Return relabeling-invariant colors for weighted directed matrix nodes."""
    matrix = np.asarray(matrix, dtype=np.float64)
    n = len(matrix)
    signatures = [
        (
            float(matrix[index, index]),
            tuple(sorted(float(value) for value in matrix[index])),
            tuple(sorted(float(value) for value in matrix[:, index])),
        )
        for index in range(n)
    ]

    def ranks(values) -> np.ndarray:
        unique = {value: rank for rank, value in enumerate(sorted(set(values)))}
        return np.asarray([unique[value] for value in values], dtype=np.int64)

    colors = ranks(signatures)
    for _ in range(n):
        refined = [
            (
                int(colors[index]),
                tuple(sorted(
                    (float(matrix[index, other]), float(matrix[other, index]), int(colors[other]))
                    for other in range(n)
                )),
            )
            for index in range(n)
        ]
        updated = ranks(refined)
        if np.array_equal(updated, colors):
            break
        colors = updated
    return colors


def canonical_node_keys(matrix: np.ndarray, colors: np.ndarray) -> list[tuple]:
    """Build relabeling-invariant weighted-neighborhood keys after refinement."""
    matrix = np.asarray(matrix, dtype=np.float64)
    colors = np.asarray(colors, dtype=np.int64)
    return [
        (
            int(colors[index]),
            tuple(sorted((float(matrix[index, other]), int(colors[other])) for other in range(len(matrix)))),
            tuple(sorted((float(matrix[other, index]), int(colors[other])) for other in range(len(matrix)))),
        )
        for index in range(len(matrix))
    ]


def build_construction_cache(instance: QAPInstance) -> ConstructionCache:
    flow = np.asarray(instance.F, dtype=np.float64)
    distance = np.asarray(instance.D, dtype=np.float64)
    n = instance.n
    flow_scale = float(np.mean(np.abs(flow))) + EPS
    distance_scale = float(np.mean(np.abs(distance))) + EPS
    objective_scale = flow_scale * distance_scale * max(n * n, 1)
    facility_abs_strength = (np.sum(np.abs(flow), axis=1) + np.sum(np.abs(flow), axis=0)) / (2.0 * n * flow_scale)
    facility_signed_strength = (np.sum(flow, axis=1) + np.sum(flow, axis=0)) / (2.0 * n * flow_scale)
    location_strength = (np.sum(np.abs(distance), axis=1) + np.sum(np.abs(distance), axis=0)) / (2.0 * n * distance_scale)
    order = np.lexsort((np.arange(n), -facility_abs_strength)).astype(np.int64)
    return ConstructionCache(
        F=flow,
        D=distance,
        order=order,
        flow_scale=flow_scale,
        distance_scale=distance_scale,
        objective_scale=objective_scale,
        facility_abs_strength=facility_abs_strength,
        facility_signed_strength=facility_signed_strength,
        facility_mean=flow.mean(axis=1) / flow_scale,
        facility_std=flow.std(axis=1) / flow_scale,
        facility_nonzero=np.count_nonzero(flow, axis=1) / max(n, 1),
        facility_rank=_average_fractional_rank(facility_abs_strength, descending=True),
        location_strength=location_strength,
        location_mean=distance.mean(axis=1) / distance_scale,
        location_std=distance.std(axis=1) / distance_scale,
        location_nonzero=np.count_nonzero(distance, axis=1) / max(n, 1),
        location_rank=_average_fractional_rank(location_strength, descending=False),
    )


def build_canonical_construction_cache(instance: QAPInstance) -> tuple[ConstructionCache, np.ndarray, list[tuple]]:
    """Use structural colors instead of raw labels to resolve construction ties."""
    cache = build_construction_cache(instance)
    facility_colors = canonical_node_colors(cache.F)
    location_colors = canonical_node_colors(cache.D)
    facility_keys = canonical_node_keys(cache.F, facility_colors)
    location_keys = canonical_node_keys(cache.D, location_colors)
    order = np.asarray(
        sorted(range(instance.n), key=lambda index: (-cache.facility_abs_strength[index], facility_keys[index])),
        dtype=np.int64,
    )
    return replace(cache, order=order), location_colors, location_keys


def assignment_from_prefix(cache: ConstructionCache, prefix: tuple[int, ...]) -> np.ndarray:
    assignment = np.full(len(cache.order), -1, dtype=np.int64)
    if prefix:
        assignment[cache.order[: len(prefix)]] = np.asarray(prefix, dtype=np.int64)
    return assignment


def partial_cost(cache: ConstructionCache, prefix: tuple[int, ...]) -> float:
    if not prefix:
        return 0.0
    facilities = cache.order[: len(prefix)]
    locations = np.asarray(prefix, dtype=np.int64)
    return float(np.sum(cache.F[np.ix_(facilities, facilities)] * cache.D[np.ix_(locations, locations)]))


def build_action_features(
    cache: ConstructionCache,
    prefix: tuple[int, ...],
    candidate_locations: np.ndarray | list[int] | tuple[int, ...],
) -> np.ndarray:
    candidates = np.asarray(candidate_locations, dtype=np.int64)
    n = len(cache.order)
    depth = len(prefix)
    if depth >= n:
        raise ValueError("complete prefix has no construction action")
    if len(set(prefix)) != depth or any(location < 0 or location >= n for location in prefix):
        raise ValueError("prefix must contain unique valid locations")
    available = set(range(n)) - set(prefix)
    if any(int(location) not in available for location in candidates):
        raise ValueError("candidate location is not available")
    facility = int(cache.order[depth])
    assigned_facilities = cache.order[:depth]
    assigned_locations = np.asarray(prefix, dtype=np.int64)
    remaining_facilities = cache.order[depth + 1 :]
    rows = np.zeros((len(candidates), len(CONSTRUCTION_FEATURE_NAMES)), dtype=np.float32)
    base_partial = partial_cost(cache, prefix) / cache.objective_scale
    for row_index, location_raw in enumerate(candidates):
        location = int(location_raw)
        if depth:
            terms = (
                cache.F[facility, assigned_facilities] * cache.D[location, assigned_locations]
                + cache.F[assigned_facilities, facility] * cache.D[assigned_locations, location]
            )
            assigned_increment = float(cache.F[facility, facility] * cache.D[location, location] + terms.sum())
            normalized_terms = terms / cache.objective_scale
        else:
            assigned_increment = float(cache.F[facility, facility] * cache.D[location, location])
            normalized_terms = np.asarray([], dtype=np.float64)
        remaining_locations = np.asarray(sorted(available - {location}), dtype=np.int64)
        if len(remaining_facilities):
            flow_profile = 0.5 * (
                np.abs(cache.F[facility, remaining_facilities])
                + np.abs(cache.F[remaining_facilities, facility])
            ) / cache.flow_scale
            distance_profile = 0.5 * (
                np.abs(cache.D[location, remaining_locations])
                + np.abs(cache.D[remaining_locations, location])
            ) / cache.distance_scale
            flow_sorted = np.sort(flow_profile)[::-1]
            distance_sorted = np.sort(distance_profile)
            rearrangement_low = float(np.mean(flow_sorted * distance_sorted))
            rearrangement_high = float(np.mean(flow_sorted * distance_sorted[::-1]))
        else:
            flow_profile = distance_profile = np.asarray([], dtype=np.float64)
            rearrangement_low = rearrangement_high = 0.0
        rows[row_index] = np.asarray([
            depth / max(n, 1),
            (n - depth) / max(n, 1),
            base_partial,
            cache.facility_abs_strength[facility],
            cache.facility_signed_strength[facility],
            cache.facility_mean[facility],
            cache.facility_std[facility],
            cache.facility_nonzero[facility],
            cache.facility_rank[facility],
            cache.location_strength[location],
            cache.location_mean[location],
            cache.location_std[location],
            cache.location_nonzero[location],
            cache.location_rank[location],
            cache.facility_abs_strength[facility] * cache.location_strength[location],
            abs(cache.facility_rank[facility] - cache.location_rank[location]),
            assigned_increment / cache.objective_scale,
            float(normalized_terms.mean()) if len(normalized_terms) else 0.0,
            float(normalized_terms.min()) if len(normalized_terms) else 0.0,
            float(normalized_terms.max()) if len(normalized_terms) else 0.0,
            float(flow_profile.sum()) / max(n, 1),
            float(flow_profile.mean()) if len(flow_profile) else 0.0,
            float(flow_profile.std()) if len(flow_profile) else 0.0,
            float(distance_profile.sum()) / max(n, 1),
            float(distance_profile.mean()) if len(distance_profile) else 0.0,
            float(distance_profile.std()) if len(distance_profile) else 0.0,
            rearrangement_low,
            rearrangement_high,
            np.log1p(n),
        ], dtype=np.float32)
    return rows


def complete_assignment(cache: ConstructionCache, prefix: tuple[int, ...]) -> np.ndarray:
    if len(prefix) != len(cache.order):
        raise ValueError("prefix is not complete")
    assignment = assignment_from_prefix(cache, prefix)
    if not is_valid_perm(assignment, len(cache.order)):
        raise RuntimeError("construction produced an invalid permutation")
    return assignment


def incremental_greedy(instance: QAPInstance) -> np.ndarray:
    cache = build_construction_cache(instance)
    prefix: tuple[int, ...] = ()
    for _ in range(instance.n):
        available = sorted(set(range(instance.n)) - set(prefix))
        features = build_action_features(cache, prefix, available)
        chosen = int(np.argmin(features[:, CONSTRUCTION_FEATURE_NAMES.index("assigned_increment")]))
        prefix += (int(available[chosen]),)
    return complete_assignment(cache, prefix)


def beam_construct(instance: QAPInstance, score_actions, width: int) -> list[np.ndarray]:
    cache = build_construction_cache(instance)
    beam: list[tuple[tuple[int, ...], float]] = [((), 0.0)]
    for _ in range(instance.n):
        expanded: list[tuple[tuple[int, ...], float]] = []
        for prefix, _ in beam:
            available = np.asarray(sorted(set(range(instance.n)) - set(prefix)), dtype=np.int64)
            features = build_action_features(cache, prefix, available)
            predicted_costs = np.asarray(score_actions(features), dtype=np.float64)
            if predicted_costs.shape != (len(available),):
                raise ValueError("score_actions must return one scalar per action")
            expanded.extend((prefix + (int(location),), float(score)) for location, score in zip(available, predicted_costs))
        expanded.sort(key=lambda item: (item[1], item[0]))
        beam = expanded[: min(int(width), len(expanded))]
    permutations = [complete_assignment(cache, prefix) for prefix, _ in beam]
    permutations.sort(key=lambda perm: (compute_cost(perm, instance.F, instance.D), tuple(perm.tolist())))
    return permutations


def partial_cost_beam_construct(instance: QAPInstance, width: int) -> list[np.ndarray]:
    """Beam baseline ranked only by the exact cost accumulated by the prefix."""
    cache = build_construction_cache(instance)
    beam: list[tuple[tuple[int, ...], float]] = [((), 0.0)]
    for _ in range(instance.n):
        expanded: list[tuple[tuple[int, ...], float]] = []
        for prefix, _ in beam:
            available = sorted(set(range(instance.n)) - set(prefix))
            prefix_cost = partial_cost(cache, prefix)
            facilities = cache.order[: len(prefix)]
            facility = int(cache.order[len(prefix)])
            for location in available:
                increment = float(cache.F[facility, facility] * cache.D[location, location])
                if len(prefix):
                    old_locations = np.asarray(prefix, dtype=np.int64)
                    increment += float(
                        np.sum(
                            cache.F[facility, facilities] * cache.D[location, old_locations]
                            + cache.F[facilities, facility] * cache.D[old_locations, location]
                        )
                    )
                expanded.append((prefix + (int(location),), prefix_cost + increment))
        expanded.sort(key=lambda item: (item[1], item[0]))
        beam = expanded[: min(int(width), len(expanded))]
    permutations = [complete_assignment(cache, prefix) for prefix, _ in beam]
    permutations.sort(key=lambda perm: (compute_cost(perm, instance.F, instance.D), tuple(perm.tolist())))
    return permutations


def _canonical_prefix_key(prefix: tuple[int, ...], location_colors: np.ndarray) -> tuple[int, ...]:
    return tuple(int(location_colors[location]) for location in prefix)


def beam_construct_canonical(instance: QAPInstance, score_actions, width: int) -> list[np.ndarray]:
    """Beam construction with relabeling-invariant structural tie-breaking."""
    cache, location_colors, location_keys = build_canonical_construction_cache(instance)
    beam: list[tuple[tuple[int, ...], float]] = [((), 0.0)]
    for _ in range(instance.n):
        expanded: list[tuple[tuple[int, ...], float]] = []
        for prefix, _ in beam:
            available = np.asarray(
                sorted(
                    set(range(instance.n)) - set(prefix),
                    key=lambda location: (location_keys[location], int(location)),
                ),
                dtype=np.int64,
            )
            features = build_action_features(cache, prefix, available)
            predicted_costs = np.asarray(score_actions(features), dtype=np.float64)
            if predicted_costs.shape != (len(available),):
                raise ValueError("score_actions must return one scalar per action")
            expanded.extend(
                (prefix + (int(location),), float(score))
                for location, score in zip(available, predicted_costs)
            )
        expanded.sort(key=lambda item: (item[1], tuple(location_keys[location] for location in item[0])))
        beam = expanded[: min(int(width), len(expanded))]
    permutations = [complete_assignment(cache, prefix) for prefix, _ in beam]
    permutations.sort(key=lambda perm: compute_cost(perm, instance.F, instance.D))
    return permutations


def partial_cost_beam_construct_canonical(instance: QAPInstance, width: int) -> list[np.ndarray]:
    """Partial-cost beam using the same structural canonicalization."""
    cache, location_colors, location_keys = build_canonical_construction_cache(instance)
    beam: list[tuple[tuple[int, ...], float]] = [((), 0.0)]
    for _ in range(instance.n):
        expanded: list[tuple[tuple[int, ...], float]] = []
        for prefix, _ in beam:
            available = sorted(
                set(range(instance.n)) - set(prefix),
                key=lambda location: (location_keys[location], int(location)),
            )
            prefix_cost = partial_cost(cache, prefix)
            facilities = cache.order[: len(prefix)]
            facility = int(cache.order[len(prefix)])
            for location in available:
                increment = float(cache.F[facility, facility] * cache.D[location, location])
                if len(prefix):
                    old_locations = np.asarray(prefix, dtype=np.int64)
                    increment += float(np.sum(
                        cache.F[facility, facilities] * cache.D[location, old_locations]
                        + cache.F[facilities, facility] * cache.D[old_locations, location]
                    ))
                expanded.append((prefix + (int(location),), prefix_cost + increment))
        expanded.sort(key=lambda item: (item[1], tuple(location_keys[location] for location in item[0])))
        beam = expanded[: min(int(width), len(expanded))]
    permutations = [complete_assignment(cache, prefix) for prefix, _ in beam]
    permutations.sort(key=lambda perm: compute_cost(perm, instance.F, instance.D))
    return permutations


def parent_balanced_beam_construct_canonical(instance: QAPInstance, score_actions, width: int) -> list[np.ndarray]:
    """Branch at the root, then retain one locally best child per beam parent."""
    cache, _, location_keys = build_canonical_construction_cache(instance)
    beam: list[tuple[int, ...]] = [()]
    for depth in range(instance.n):
        next_beam: list[tuple[int, ...]] = []
        for prefix in beam:
            available = np.asarray(
                sorted(
                    set(range(instance.n)) - set(prefix),
                    key=lambda location: (location_keys[location], int(location)),
                ),
                dtype=np.int64,
            )
            features = build_action_features(cache, prefix, available)
            scores = np.asarray(score_actions(features), dtype=np.float64)
            if scores.shape != (len(available),):
                raise ValueError("score_actions must return one scalar per action")
            ranked = sorted(
                zip(available.tolist(), scores.tolist()),
                key=lambda item: (float(item[1]), location_keys[int(item[0])]),
            )
            quota = min(int(width), len(ranked)) if depth == 0 else 1
            next_beam.extend(prefix + (int(location),) for location, _ in ranked[:quota])
        beam = next_beam
    permutations = [complete_assignment(cache, prefix) for prefix in beam]
    permutations.sort(key=lambda perm: compute_cost(perm, instance.F, instance.D))
    return permutations


def parent_balanced_partial_cost_beam_construct_canonical(instance: QAPInstance, width: int) -> list[np.ndarray]:
    """Partial-cost control for one-child-per-parent construction."""
    cache, _, location_keys = build_canonical_construction_cache(instance)
    beam: list[tuple[int, ...]] = [()]
    for depth in range(instance.n):
        next_beam: list[tuple[int, ...]] = []
        for prefix in beam:
            available = sorted(
                set(range(instance.n)) - set(prefix),
                key=lambda location: (location_keys[location], int(location)),
            )
            ranked = sorted(
                (prefix + (int(location),) for location in available),
                key=lambda child: (partial_cost(cache, child), prefix_key_for_locations(child, location_keys)),
            )
            quota = min(int(width), len(ranked)) if depth == 0 else 1
            next_beam.extend(ranked[:quota])
        beam = next_beam
    permutations = [complete_assignment(cache, prefix) for prefix in beam]
    permutations.sort(key=lambda perm: compute_cost(perm, instance.F, instance.D))
    return permutations


def mlp_prefilter_partial_beam_construct_canonical(
    instance: QAPInstance, score_actions, width: int, per_parent: int = 2,
) -> list[np.ndarray]:
    """Use MLP only as a per-parent prefilter, then compare parents by partial cost."""
    if int(per_parent) < 1:
        raise ValueError("per_parent must be positive")
    cache, _, location_keys = build_canonical_construction_cache(instance)
    beam: list[tuple[int, ...]] = [()]
    for depth in range(instance.n):
        expanded: list[tuple[tuple[int, ...], float]] = []
        for prefix in beam:
            available = np.asarray(
                sorted(
                    set(range(instance.n)) - set(prefix),
                    key=lambda location: (location_keys[location], int(location)),
                ),
                dtype=np.int64,
            )
            features = build_action_features(cache, prefix, available)
            mlp_scores = np.asarray(score_actions(features), dtype=np.float64)
            ranked = sorted(
                zip(available.tolist(), mlp_scores.tolist()),
                key=lambda item: (float(item[1]), location_keys[int(item[0])]),
            )
            quota = int(width) if depth == 0 else min(int(per_parent), len(ranked))
            for location, _ in ranked[:quota]:
                child = prefix + (int(location),)
                expanded.append((child, partial_cost(cache, child)))
        expanded.sort(key=lambda item: (item[1], prefix_key_for_locations(item[0], location_keys)))
        beam = [prefix for prefix, _ in expanded[: min(int(width), len(expanded))]]
    permutations = [complete_assignment(cache, prefix) for prefix in beam]
    permutations.sort(key=lambda perm: compute_cost(perm, instance.F, instance.D))
    return permutations


def prefix_key_for_locations(prefix: tuple[int, ...], location_keys: list[tuple]) -> tuple:
    return tuple(location_keys[location] for location in prefix)
