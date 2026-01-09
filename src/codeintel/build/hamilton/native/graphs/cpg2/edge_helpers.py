"""Shared helpers for building CPG edge tables."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal

import pyarrow as pa

from codeintel.build.hamilton.native.graphs.cpg.constants import (
    CPG_EDGES_TABLE_KEY,
    CPG_TARGET_NAME,
)
from codeintel.build.tabular.explode_ops import ExplodeSpec, explode_edges
from codeintel.build.tabular.expr_vocab import E
from codeintel.build.tabular.finalize_ops import FinalizeSpec, finalize_table
from codeintel.build.tabular.kernels import stable_sort_indices
from codeintel.build.tabular.plan_ops import Plan, materialize_plan
from codeintel.core.columnar.rows import empty_table_for_table

SortDirection = Literal["ascending", "descending"]
SortKey = tuple[str, SortDirection]

_CPG_EDGE_FIELDS: tuple[str, ...] = (
    "repo",
    "commit",
    "src_cpg_node_id",
    "dst_cpg_node_id",
    "edge_kind",
    "edge_layer",
    "rel_path",
    "ordinal",
    "extras",
    "extras_kv",
)

_DEFAULT_SORT_KEYS: tuple[SortKey, ...] = (
    ("repo", "ascending"),
    ("commit", "ascending"),
    ("src_cpg_node_id", "ascending"),
    ("dst_cpg_node_id", "ascending"),
    ("edge_kind", "ascending"),
    ("ordinal", "ascending"),
)


def finalize_cpg_edge_rows(
    edge_rows: Sequence[Mapping[str, object]],
    *,
    sort_keys: Sequence[SortKey] | None = None,
    error_context_cols: Sequence[str] = ("repo", "commit", "edge_kind"),
) -> pa.Table:
    """Explode + finalize CPG edge rows into a contract-aligned table.

    Parameters
    ----------
    edge_rows
        Edge rows to materialize as graph.cpg_edges.
    sort_keys
        Optional stable sort keys for deterministic ordering.
    error_context_cols
        Parent columns to include in explode error reports.

    Returns
    -------
    pyarrow.Table
        Finalized edge table aligned to the contract schema.
    """
    if not edge_rows:
        return empty_table_for_table(CPG_EDGES_TABLE_KEY)

    parent_rows: list[dict[str, object]] = []
    for index, row in enumerate(edge_rows):
        edge = dict(row)
        context = {name: edge.get(name) for name in error_context_cols}
        parent_rows.append({"edge_id": index, **context, "edge": [edge]})

    parent_table = pa.Table.from_pylist(parent_rows)
    exploded = explode_edges(
        parent_table,
        spec=ExplodeSpec(
            src_col="edge_id",
            dst_list_col="edge",
            error_context_cols=error_context_cols,
        ),
    )
    if exploded.good.num_rows == 0:
        return empty_table_for_table(CPG_EDGES_TABLE_KEY)

    projection = {name: E.field(("edge", name)) for name in _CPG_EDGE_FIELDS}
    plan = Plan.table(exploded.good).project(projection)
    edges = materialize_plan(plan, use_threads=True)

    resolved_sort_keys = sort_keys if sort_keys is not None else _DEFAULT_SORT_KEYS
    if resolved_sort_keys:
        edges = edges.take(stable_sort_indices(edges, sort_keys=resolved_sort_keys))

    result = finalize_table(
        edges,
        spec=FinalizeSpec(
            table_key=CPG_EDGES_TABLE_KEY,
            mode="strict",
            target_name=CPG_TARGET_NAME,
        ),
    )
    return result.good


__all__ = ["finalize_cpg_edge_rows"]
