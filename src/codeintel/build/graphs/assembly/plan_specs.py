"""Reusable plan specs for graph assembly."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from codeintel.build.tabular.expr_vocab import E, Expression
from codeintel.build.tabular.plan_ops import HashJoinSpec, JoinType
from codeintel.core.columnar.ordering import SortKey


@dataclass(frozen=True, slots=True)
class GraphJoinSpec:
    """Convenience wrapper for hash join specs in graph pipelines."""

    left_keys: tuple[str, ...]
    right_keys: tuple[str, ...]
    how: JoinType = "inner"
    left_output: tuple[str, ...] | None = None
    right_output: tuple[str, ...] | None = None
    output_suffix_for_left: str | None = None
    output_suffix_for_right: str | None = None
    filter_expression: Expression | None = None

    def to_hash_join_spec(self) -> HashJoinSpec:
        """Return the HashJoinSpec for this graph join configuration.

        Returns
        -------
        HashJoinSpec
            Hash join spec derived from this configuration.
        """
        return HashJoinSpec(
            left_keys=self.left_keys,
            right_keys=self.right_keys,
            how=self.how,
            left_output=self.left_output,
            right_output=self.right_output,
            output_suffix_for_left=self.output_suffix_for_left,
            output_suffix_for_right=self.output_suffix_for_right,
            filter_expression=self.filter_expression,
        )


def graph_join_spec(spec: GraphJoinSpec) -> HashJoinSpec:
    """Create a HashJoinSpec for graph plan pipelines.

    Returns
    -------
    HashJoinSpec
        Hash join spec for graph assembly.
    """
    return spec.to_hash_join_spec()


def projection_for_columns(columns: Sequence[str]) -> Mapping[str, Expression]:
    """Return a projection mapping for the provided column names.

    Returns
    -------
    Mapping[str, Expression]
        Projection mapping for the selected columns.
    """
    return {name: E.field(name) for name in columns}


def edge_projection(
    *,
    src: str,
    dst: str,
    weight: str | None = None,
    extras: Sequence[str] = (),
) -> Mapping[str, Expression]:
    """Return a projection mapping for edge tables.

    Returns
    -------
    Mapping[str, Expression]
        Projection mapping for edge columns.
    """
    columns = [src, dst]
    if weight is not None:
        columns.append(weight)
    for extra in extras:
        if extra not in columns:
            columns.append(extra)
    return projection_for_columns(columns)


def node_projection(
    *,
    node_id: str,
    attrs: Sequence[str] = (),
) -> Mapping[str, Expression]:
    """Return a projection mapping for node tables.

    Returns
    -------
    Mapping[str, Expression]
        Projection mapping for node columns.
    """
    columns = [node_id]
    for attr in attrs:
        if attr not in columns:
            columns.append(attr)
    return projection_for_columns(columns)


def ordering_keys(keys: Sequence[str]) -> tuple[SortKey, ...]:
    """Return ascending sort keys for the provided column names.

    Returns
    -------
    tuple[SortKey, ...]
        Sort keys for ascending order.
    """
    return tuple((key, "ascending") for key in keys)


__all__ = [
    "GraphJoinSpec",
    "edge_projection",
    "graph_join_spec",
    "node_projection",
    "ordering_keys",
    "projection_for_columns",
]
