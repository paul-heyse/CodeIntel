"""Shared helpers for CFG and DFG analytics.

This module consolidates common utility functions used by both cfg_core.py
and dfg_core.py to eliminate code duplication.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import pyarrow as pa

from codeintel.build.analytics.utilities.snapshot import SnapshotContext, snapshot_plan
from codeintel.build.graphs.rx.algos import GraphInput, ensure_store
from codeintel.build.graphs.rx.iterators import iter_edge_id_weights
from codeintel.build.tabular.arrow_ops import iter_rows
from codeintel.build.tabular.compute_helpers import safe_filter_expr
from codeintel.build.tabular.compute_masks import equal_expr, is_in_expr, is_valid_expr
from codeintel.build.tabular.expr_vocab import E, Expression
from codeintel.core.columnar.arrowdsl import ExecutionPlan, PipelineRunOptions, run_pipeline
from codeintel.core.columnar.conversion import empty_table_from_schema
from codeintel.core.columnar.finalize_ops import finalize_spec_for_table
from codeintel.core.columnar.plan_builder import TablePlanOptions, build_table_plan
from codeintel.core.columnar.plan_kernels import GroupedRollupSpec, grouped_rollup_table
from codeintel.core.columnar.plan_ops import Plan
from codeintel.core.data_models.ids import normalize_decimal_id

if TYPE_CHECKING:
    from codeintel.core.columnar.execution_context import ExecutionContext


def degree_dict(
    graph: GraphInput,
    *,
    direction: str,
    weight: str | None = None,
) -> dict[int, int]:
    """Materialize degree counts into a concrete mapping for type safety.

    Parameters
    ----------
    graph
        The directed graph to compute degrees for.
    direction
        Either "in" for in-degree or "out" for out-degree.
    weight
        Optional edge weight attribute name.

    Returns
    -------
    dict[int, int]
        Mapping of node -> degree.
    """
    store = ensure_store(graph, weight=weight)
    degree_map: dict[int, int] = {int(str(node)): 0 for node in store.node_ids()}
    for src_id, dst_id, weight_val in iter_edge_id_weights(store):
        weight_int = int(weight_val)
        if direction == "in":
            degree_map[int(str(dst_id))] = degree_map.get(int(str(dst_id)), 0) + weight_int
        else:
            degree_map[int(str(src_id))] = degree_map.get(int(str(src_id)), 0) + weight_int
    return degree_map


CFG_BLOCKS_TABLE_KEY = "graph.cfg_blocks"
CFG_EDGES_TABLE_KEY = "graph.cfg_edges"
DFG_EDGES_TABLE_KEY = "graph.dfg_edges"
CORE_GOIDS_TABLE_KEY = "core.goids"
CORE_MODULES_TABLE_KEY = "core.modules"


def parse_block_idx(block_id: str | int | None) -> int | None:
    """Extract the integer block index from a block identifier.

    Parameters
    ----------
    block_id
        Block identifier string in the form "block<N>" or an integer.

    Returns
    -------
    int | None
        Parsed block index when available.
    """
    if block_id is None:
        return None
    block_text = str(block_id)
    if "block" not in block_text:
        return None
    try:
        return int(block_text.rsplit("block", 1)[-1])
    except ValueError:
        return None


def _combine_expr(
    current: Expression | None,
    next_expr: Expression,
) -> Expression:
    if current is None:
        return next_expr
    return cast("Expression", current & next_expr)


@dataclass(frozen=True, slots=True)
class PrefilterRequest:
    """Prefilter settings for table expressions."""

    repo: str | None = None
    commit: str | None = None
    kinds: Sequence[str] | None = None
    require_valid: Sequence[str] = ()


def prefilter_table(
    table: pa.Table,
    *,
    table_key: str,
    request: PrefilterRequest,
) -> pa.Table:
    """Prefilter a table using compute expressions when columns exist.

    Returns
    -------
    pyarrow.Table
        Filtered table when expressions apply.
    """
    column_names = set(table.column_names)
    expr: Expression | None = None
    if request.repo is not None and "repo" in column_names:
        expr = _combine_expr(expr, equal_expr("repo", request.repo))
    if request.commit is not None and "commit" in column_names:
        expr = _combine_expr(expr, equal_expr("commit", request.commit))
    if request.kinds and "kind" in column_names:
        expr = _combine_expr(expr, is_in_expr("kind", value_set=request.kinds))
    for name in request.require_valid:
        if name in column_names:
            expr = _combine_expr(expr, is_valid_expr(name))
    if expr is None:
        return table
    try:
        return _materialize_plan(
            build_table_plan(
                table=table,
                options=TablePlanOptions(filter_expr=expr),
            ),
            table_key=table_key,
            ctx=None,
        )
    except (pa.ArrowInvalid, pa.ArrowNotImplementedError, pa.ArrowTypeError, TypeError, ValueError):
        return safe_filter_expr(table, expr)


def cfg_blocks_rowset(
    table: pa.Table,
    *,
    repo: str | None = None,
    commit: str | None = None,
    ctx: ExecutionContext | None = None,
) -> pa.Table:
    """Build a grouped rowset of CFG blocks by function.

    Returns
    -------
    pyarrow.Table
        Aggregated CFG block rows keyed by function.
    """
    required = (
        "function_goid_h128",
        "block_idx",
        "kind",
        "in_degree",
        "out_degree",
    )
    if not set(required).issubset(table.column_names):
        return pa.Table.from_pylist([])
    plan = snapshot_plan(
        table,
        columns=required,
        context=SnapshotContext(
            repo=repo,
            commit=commit,
            ctx=ctx,
            table_key=CFG_BLOCKS_TABLE_KEY,
        ),
    )
    plan = plan.filter(
        E.and_(
            E.is_valid("function_goid_h128"),
            E.is_valid("block_idx"),
        )
    )
    filtered = _materialize_plan(plan, table_key=CFG_BLOCKS_TABLE_KEY, ctx=ctx)
    return grouped_rollup_table(
        filtered,
        spec=GroupedRollupSpec(
            keys=("function_goid_h128",),
            aggregates=[
                ("block_idx", "list", None, "block_idx"),
                ("kind", "list", None, "kind"),
                ("in_degree", "list", None, "in_degree"),
                ("out_degree", "list", None, "out_degree"),
            ],
            pre_sort_keys=(("function_goid_h128", "ascending"), ("block_idx", "ascending")),
        ),
        ctx=ctx,
    )


def cfg_edges_rowset(
    table: pa.Table,
    *,
    repo: str | None = None,
    commit: str | None = None,
    ctx: ExecutionContext | None = None,
) -> pa.Table:
    """Build a grouped rowset of CFG edges by function.

    Returns
    -------
    pyarrow.Table
        Aggregated CFG edge rows keyed by function.
    """
    required = ("function_goid_h128", "src_block_id", "dst_block_id", "edge_kind")
    if not set(required).issubset(table.column_names):
        return pa.Table.from_pylist([])
    plan = snapshot_plan(
        table,
        columns=required,
        context=SnapshotContext(
            repo=repo,
            commit=commit,
            ctx=ctx,
            table_key=CFG_EDGES_TABLE_KEY,
        ),
    )
    plan = plan.filter(
        E.and_(
            E.is_valid("function_goid_h128"),
            E.is_valid("src_block_id"),
            E.is_valid("dst_block_id"),
        )
    )
    filtered = _materialize_plan(plan, table_key=CFG_EDGES_TABLE_KEY, ctx=ctx)
    return grouped_rollup_table(
        filtered,
        spec=GroupedRollupSpec(
            keys=("function_goid_h128",),
            aggregates=[
                ("src_block_id", "list", None, "src_block_id"),
                ("dst_block_id", "list", None, "dst_block_id"),
                ("edge_kind", "list", None, "edge_kind"),
            ],
            pre_sort_keys=(
                ("function_goid_h128", "ascending"),
                ("src_block_id", "ascending"),
                ("dst_block_id", "ascending"),
                ("edge_kind", "ascending"),
            ),
        ),
        ctx=ctx,
    )


def dfg_edges_rowset(
    table: pa.Table,
    *,
    repo: str | None = None,
    commit: str | None = None,
    ctx: ExecutionContext | None = None,
) -> pa.Table:
    """Build a grouped rowset of DFG edges by function.

    Returns
    -------
    pyarrow.Table
        Aggregated DFG edge rows keyed by function.
    """
    required = (
        "function_goid_h128",
        "src_block_id",
        "dst_block_id",
        "src_var",
        "dst_var",
        "via_phi",
        "use_kind",
    )
    if not set(required).issubset(table.column_names):
        return pa.Table.from_pylist([])
    plan = snapshot_plan(
        table,
        columns=required,
        context=SnapshotContext(
            repo=repo,
            commit=commit,
            ctx=ctx,
            table_key=DFG_EDGES_TABLE_KEY,
        ),
    )
    plan = plan.filter(
        E.and_(
            E.is_valid("function_goid_h128"),
            E.is_valid("src_block_id"),
            E.is_valid("dst_block_id"),
            E.is_valid("src_var"),
            E.is_valid("dst_var"),
        )
    )
    filtered = _materialize_plan(plan, table_key=DFG_EDGES_TABLE_KEY, ctx=ctx)
    return grouped_rollup_table(
        filtered,
        spec=GroupedRollupSpec(
            keys=("function_goid_h128",),
            aggregates=[
                ("src_block_id", "list", None, "src_block_id"),
                ("dst_block_id", "list", None, "dst_block_id"),
                ("src_var", "list", None, "src_var"),
                ("dst_var", "list", None, "dst_var"),
                ("via_phi", "list", None, "via_phi"),
                ("use_kind", "list", None, "use_kind"),
            ],
            pre_sort_keys=(
                ("function_goid_h128", "ascending"),
                ("src_block_id", "ascending"),
                ("dst_block_id", "ascending"),
                ("src_var", "ascending"),
                ("dst_var", "ascending"),
                ("use_kind", "ascending"),
                ("via_phi", "ascending"),
            ),
        ),
        ctx=ctx,
    )


def load_function_metadata(
    goids_frame: pa.Table,
    modules_frame: pa.Table,
    *,
    repo: str,
    commit: str,
    ctx: ExecutionContext | None = None,
) -> dict[int, tuple[str, str | None, str | None]]:
    """Load function metadata keyed by GOID from tabular frames.

    Parameters
    ----------
    goids_frame
        Frame containing ``core.goids`` rows.
    modules_frame
        Frame containing ``core.modules`` rows.
    repo
        Repository identifier.
    commit
        Commit identifier.
    ctx
        Optional execution context for determinism and profiling.

    Returns
    -------
    dict[int, tuple[str, str | None, str | None]]
        Mapping of GOID -> (rel_path, module, qualname).
    """
    module_by_path: dict[str, str] = {}
    filtered_modules = _module_metadata_table(
        modules_frame,
        repo=repo,
        commit=commit,
        ctx=ctx,
    )
    for row in iter_rows(filtered_modules, ("path", "module")):
        path = row.get("path")
        module = row.get("module")
        if isinstance(path, str) and isinstance(module, str):
            module_by_path[path] = module

    metadata: dict[int, tuple[str, str | None, str | None]] = {}
    filtered_goids = _goid_metadata_table(goids_frame, repo=repo, commit=commit, ctx=ctx)
    for row in iter_rows(filtered_goids, ("goid_h128", "rel_path", "qualname")):
        goid = normalize_decimal_id(row.get("goid_h128"))
        if goid is None:
            continue
        rel_path = row.get("rel_path")
        if not isinstance(rel_path, str):
            continue
        qualname = row.get("qualname")
        module = module_by_path.get(rel_path)
        metadata[int(goid)] = (
            rel_path,
            module,
            qualname if isinstance(qualname, str) else None,
        )
    return metadata


def _module_metadata_table(
    table: pa.Table,
    *,
    repo: str,
    commit: str,
    ctx: ExecutionContext | None,
) -> pa.Table:
    if "path" not in table.column_names or "module" not in table.column_names:
        return empty_table_from_schema(table.schema)
    plan = snapshot_plan(
        table,
        context=SnapshotContext(
            repo=repo,
            commit=commit,
            ctx=ctx,
            table_key=CORE_MODULES_TABLE_KEY,
        ),
    )
    plan = plan.filter(E.and_(E.is_valid("path"), E.is_valid("module")))
    plan = plan.project({"path": E.field("path"), "module": E.field("module")})
    filtered = _materialize_plan(plan, table_key=CORE_MODULES_TABLE_KEY, ctx=ctx)
    return grouped_rollup_table(
        filtered,
        spec=GroupedRollupSpec(
            keys=("path",),
            aggregates=(("module", "min", None, "module"),),
        ),
        ctx=ctx,
    )


def _goid_metadata_table(
    table: pa.Table,
    *,
    repo: str,
    commit: str,
    ctx: ExecutionContext | None,
) -> pa.Table:
    required = {"goid_h128", "rel_path"}
    if not required.issubset(table.column_names):
        return empty_table_from_schema(table.schema)
    plan = snapshot_plan(
        table,
        context=SnapshotContext(
            repo=repo,
            commit=commit,
            ctx=ctx,
            table_key=CORE_GOIDS_TABLE_KEY,
        ),
    )
    filters: list[Expression] = []
    if "kind" in table.column_names:
        filters.append(E.in_("kind", ["function", "method"]))
    filters.append(E.is_valid("goid_h128"))
    filters.append(E.is_valid("rel_path"))
    plan = plan.filter(E.and_(*filters))
    project = {
        "goid_h128": E.field("goid_h128"),
        "rel_path": E.field("rel_path"),
    }
    if "qualname" in table.column_names:
        project["qualname"] = E.field("qualname")
    else:
        project["qualname"] = E.scalar(None)
    plan = plan.project(project)
    filtered = _materialize_plan(plan, table_key=CORE_GOIDS_TABLE_KEY, ctx=ctx)
    return grouped_rollup_table(
        filtered,
        spec=GroupedRollupSpec(
            keys=("goid_h128",),
            aggregates=(
                ("rel_path", "min", None, "rel_path"),
                ("qualname", "min", None, "qualname"),
            ),
        ),
        ctx=ctx,
    )


def _materialize_plan(
    plan: Plan,
    *,
    table_key: str,
    ctx: ExecutionContext | None,
) -> pa.Table:
    result = run_pipeline(
        plan=ExecutionPlan.from_plan(plan),
        finalize=finalize_spec_for_table(table_key, mode="tolerant"),
        options=PipelineRunOptions(ctx=ctx),
    )
    return result.good


__all__ = [
    "cfg_blocks_rowset",
    "cfg_edges_rowset",
    "degree_dict",
    "dfg_edges_rowset",
    "load_function_metadata",
    "parse_block_idx",
    "prefilter_table",
]
