"""Policy objects for rustworkx graph construction and normalization."""

from __future__ import annotations

import operator
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from codeintel.build.graphs.rx.normalize import NanPolicy, edge_weight_from_payload

if TYPE_CHECKING:
    from codeintel.build.graphs.engine.protocol import GraphKind


@dataclass(frozen=True, slots=True)
class GraphWeightPolicy:
    """Configuration for combining edge weights."""

    name: str
    default_weight: float = 1.0
    combine: Callable[[float, float], float] = operator.add
    nan_policy: NanPolicy = "keep"

    def normalize_weight(self, value: object | None) -> float:
        """Coerce input values into numeric weights.

        Returns
        -------
        float
            Normalized numeric edge weight.
        """
        if value is None:
            return self.default_weight
        return edge_weight_from_payload(value, nan_policy=self.nan_policy)

    def combine_weights(self, current: float, increment: float) -> float:
        """Combine an existing weight with an increment.

        Returns
        -------
        float
            Combined edge weight.
        """
        return self.combine(current, increment)


@dataclass(frozen=True, slots=True)
class GraphNumericPolicy:
    """Numeric normalization policy for algorithm outputs."""

    nan_policy: NanPolicy = "keep"
    clustering_abs_tol: float = 1e-9
    clustering_rel_tol: float = 1e-6
    constraint_abs_tol: float = 1e-9
    constraint_rel_tol: float = 1e-6
    effective_abs_tol: float = 1e-9
    effective_rel_tol: float = 1e-6
    harmonic_abs_tol: float = 1e-9
    harmonic_rel_tol: float = 1e-6
    projection_abs_tol: float = 1e-12
    projection_rel_tol: float = 1e-9
    bipartite_abs_tol: float = 1e-12
    bipartite_rel_tol: float = 1e-9
    dijkstra_abs_tol: float = 1e-12
    dijkstra_rel_tol: float = 1e-9


DEFAULT_WEIGHT_POLICY = GraphWeightPolicy(name="sum")
DEFAULT_NUMERIC_POLICY = GraphNumericPolicy()

GRAPH_KIND_WEIGHT_POLICIES: dict[str, GraphWeightPolicy] = {
    "CALL_GRAPH": DEFAULT_WEIGHT_POLICY,
    "IMPORT_GRAPH": DEFAULT_WEIGHT_POLICY,
    "CFG_GRAPH": DEFAULT_WEIGHT_POLICY,
    "SYMBOL_MODULE_GRAPH": DEFAULT_WEIGHT_POLICY,
    "SYMBOL_FUNCTION_GRAPH": DEFAULT_WEIGHT_POLICY,
    "CONFIG_MODULE_BIPARTITE": DEFAULT_WEIGHT_POLICY,
}


def weight_policy_for_name(name: str | None) -> GraphWeightPolicy | None:
    """Resolve a weight policy by name when registered.

    Returns
    -------
    GraphWeightPolicy | None
        Matching policy when found.
    """
    if name is None:
        return None
    for policy in GRAPH_KIND_WEIGHT_POLICIES.values():
        if policy.name == name:
            return policy
    if DEFAULT_WEIGHT_POLICY.name == name:
        return DEFAULT_WEIGHT_POLICY
    return None


def _graph_kind_tokens(kind: GraphKind) -> list[str]:
    name = getattr(kind, "name", None)
    if isinstance(name, str) and name:
        return name.split("|")
    raw = str(kind)
    if "." in raw:
        raw = raw.split(".", 1)[1]
    return raw.split("|")


def weight_policy_for_kind(kind: GraphKind) -> GraphWeightPolicy:
    """Return the weight policy for a specific graph kind.

    Returns
    -------
    GraphWeightPolicy
        Weight policy for the requested graph kind.
    """
    name = getattr(kind, "name", None)
    if isinstance(name, str):
        policy = GRAPH_KIND_WEIGHT_POLICIES.get(name)
        if policy is not None:
            return policy
    for token in _graph_kind_tokens(kind):
        policy = GRAPH_KIND_WEIGHT_POLICIES.get(token)
        if policy is not None:
            return policy
    return DEFAULT_WEIGHT_POLICY


__all__ = [
    "DEFAULT_NUMERIC_POLICY",
    "DEFAULT_WEIGHT_POLICY",
    "GRAPH_KIND_WEIGHT_POLICIES",
    "GraphNumericPolicy",
    "GraphWeightPolicy",
    "weight_policy_for_kind",
    "weight_policy_for_name",
]
