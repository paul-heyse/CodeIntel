"""Weight semantics helpers for rustworkx graph algorithms."""

from __future__ import annotations

import math
from enum import Enum

from codeintel.build.graphs.rx.normalize import NanPolicy, edge_weight_from_payload

DEFAULT_WEIGHT_EPSILON = 1e-12


class WeightSemantics(Enum):
    """Interpretation of edge weights for algorithms."""

    STRENGTH = "strength"
    COST = "cost"


def _validate_positive(value: float, *, label: str) -> float:
    if not math.isfinite(value) or value <= 0:
        message = f"{label} must be a positive finite number"
        raise ValueError(message)
    return value


def strength_to_cost(strength: float, *, epsilon: float = DEFAULT_WEIGHT_EPSILON) -> float:
    """Convert a strength weight into a cost suitable for shortest paths.

    Returns
    -------
    float
        Cost weight suitable for shortest path algorithms.
    """
    _validate_positive(strength, label="Edge strength")
    return 1.0 / max(strength, epsilon)


def cost_to_strength(cost: float, *, epsilon: float = DEFAULT_WEIGHT_EPSILON) -> float:
    """Convert a cost weight into a strength suitable for centrality scores.

    Returns
    -------
    float
        Strength weight suitable for centrality algorithms.
    """
    _validate_positive(cost, label="Edge cost")
    return 1.0 / max(cost, epsilon)


def edge_strength_from_payload(
    payload: object | None,
    *,
    nan_policy: NanPolicy,
    semantics: WeightSemantics,
    epsilon: float = DEFAULT_WEIGHT_EPSILON,
) -> float:
    """Return an edge strength weight from a payload and semantics.

    Returns
    -------
    float
        Edge strength weight for algorithm inputs.
    """
    weight = edge_weight_from_payload(payload, nan_policy=nan_policy)
    if semantics is WeightSemantics.STRENGTH:
        _validate_positive(weight, label="Edge strength")
        return weight
    return cost_to_strength(weight, epsilon=epsilon)


def edge_cost_from_payload(
    payload: object | None,
    *,
    nan_policy: NanPolicy,
    semantics: WeightSemantics,
    epsilon: float = DEFAULT_WEIGHT_EPSILON,
) -> float:
    """Return an edge cost weight from a payload and semantics.

    Returns
    -------
    float
        Edge cost weight for algorithm inputs.
    """
    weight = edge_weight_from_payload(payload, nan_policy=nan_policy)
    if semantics is WeightSemantics.COST:
        _validate_positive(weight, label="Edge cost")
        return weight
    return strength_to_cost(weight, epsilon=epsilon)


__all__ = [
    "DEFAULT_WEIGHT_EPSILON",
    "WeightSemantics",
    "cost_to_strength",
    "edge_cost_from_payload",
    "edge_strength_from_payload",
    "strength_to_cost",
]
