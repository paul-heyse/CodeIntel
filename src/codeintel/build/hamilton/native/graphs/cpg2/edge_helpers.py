"""Shared helpers for building CPG edge tables."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Literal

import pyarrow as pa

from codeintel.build.hamilton.native.graphs.cpg.constants import (
    CPG_EDGES_TABLE_KEY,
)
from codeintel.build.tabular.finalize_ops import finalize_reader, finalize_spec_for_table
from codeintel.core.columnar.arrowdsl import ExecutionPlan, project_struct_fields
from codeintel.core.columnar.execution_context import resolve_execution_context
from codeintel.core.columnar.plan_builder import TablePlanOptions, build_table_plan
from codeintel.core.columnar.plan_kernels import ExplodeSpec, explode_edges_for_join
from codeintel.core.columnar.plan_ops import Plan
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
    ("edge_layer", "ascending"),
    ("ordinal", "ascending"),
)
_INTERNAL_PLAN_TABLE_KEY = "internal.plan_materialize"

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

    column_names = ("edge_id", *error_context_cols, "edge")
    parent_data: dict[str, list[object]] = {name: [] for name in column_names}
    for index, row in enumerate(edge_rows):
        edge = dict(row)
        parent_data["edge_id"].append(index)
        for name in error_context_cols:
            parent_data[name].append(edge.get(name))
        parent_data["edge"].append([edge])

    parent_table = pa.Table.from_pydict(parent_data)
    exploded = explode_edges_for_join(
        parent_table,
        spec=ExplodeSpec(
            src_col="edge_id",
            dst_list_col="edge",
            error_context_cols=error_context_cols,
        ),
        allowed_columns=("edge",),
    )
    if exploded.good.num_rows == 0:
        return empty_table_for_table(CPG_EDGES_TABLE_KEY)

    projection = project_struct_fields("edge", _CPG_EDGE_FIELDS)
    plan = build_table_plan(
        table=exploded.good,
        options=TablePlanOptions(projection=projection),
    )
    resolved_sort_keys = sort_keys if sort_keys is not None else _DEFAULT_SORT_KEYS
    ordered = plan.order_by(sort_keys=list(resolved_sort_keys))
    return _plan_to_table(ordered, use_threads=True)


def _plan_to_table(plan: Plan, *, use_threads: bool) -> pa.Table:
    execution_ctx = resolve_execution_context(None)
    if not use_threads:
        execution_ctx = replace(execution_ctx, use_threads=False)
    reader = ExecutionPlan.from_plan(plan).to_reader(ctx=execution_ctx)
    result = finalize_reader(
        reader,
        spec=finalize_spec_for_table(
            _INTERNAL_PLAN_TABLE_KEY,
            mode="tolerant",
            ordering=plan.ordering,
        ),
    )
    return result.good


__all__ = ["finalize_cpg_edge_rows"]
