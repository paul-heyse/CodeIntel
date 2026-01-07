"""Shared helpers for CFG and DFG analytics.

This module consolidates common utility functions used by both cfg_core.py
and dfg_core.py to eliminate code duplication.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, cast

import pyarrow as pa
import pyarrow.compute as pc

from codeintel.build.tabular.arrow_ops import iter_rows
from codeintel.build.tabular.compute_helpers import safe_filter
from codeintel.build.tabular.compute_masks import equal_expr, is_in_expr, is_valid_expr
from codeintel.core.data_models.ids import normalize_decimal_id

if TYPE_CHECKING:
    from collections.abc import Iterable

    import networkx as nx
    from pyarrow.compute import Expression as ComputeExpression
else:
    ComputeExpression = object

_EXPR_TYPE = getattr(pc, "Expression", None)


def degree_dict(
    graph: nx.DiGraph,
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
    raw_pairs = (
        graph.in_degree(weight=weight) if direction == "in" else graph.out_degree(weight=weight)
    )
    pairs = cast("Iterable[tuple[int, int | float]]", raw_pairs)
    return {int(node): int(deg) for node, deg in pairs}


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
    current: ComputeExpression | None,
    next_expr: ComputeExpression,
) -> ComputeExpression:
    if current is None:
        return next_expr
    return cast("ComputeExpression", current & next_expr)


def prefilter_table(
    table: pa.Table,
    *,
    repo: str | None = None,
    commit: str | None = None,
    kinds: Sequence[str] | None = None,
    require_valid: Sequence[str] = (),
) -> pa.Table:
    """Prefilter a table using compute expressions when columns exist.

    Returns
    -------
    pyarrow.Table
        Filtered table when expressions apply.
    """
    column_names = set(table.column_names)
    expr: ComputeExpression | None = None
    if repo is not None and "repo" in column_names:
        expr = _combine_expr(expr, equal_expr("repo", repo))
    if commit is not None and "commit" in column_names:
        expr = _combine_expr(expr, equal_expr("commit", commit))
    if kinds and "kind" in column_names:
        expr = _combine_expr(expr, is_in_expr("kind", value_set=kinds))
    for name in require_valid:
        if name in column_names:
            expr = _combine_expr(expr, is_valid_expr(name))
    if expr is None or _EXPR_TYPE is None:
        return table
    return safe_filter(table, expr)


def load_function_metadata(
    goids_frame: pa.Table,
    modules_frame: pa.Table,
    *,
    repo: str,
    commit: str,
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

    Returns
    -------
    dict[int, tuple[str, str | None, str | None]]
        Mapping of GOID -> (rel_path, module, qualname).
    """
    module_by_path: dict[str, str] = {}
    filtered_modules = prefilter_table(
        modules_frame,
        repo=repo,
        commit=commit,
        require_valid=("path", "module"),
    )
    for row in iter_rows(filtered_modules):
        path = row.get("path")
        module = row.get("module")
        if isinstance(path, str) and isinstance(module, str):
            module_by_path[path] = module

    metadata: dict[int, tuple[str, str | None, str | None]] = {}
    filtered_goids = prefilter_table(
        goids_frame,
        repo=repo,
        commit=commit,
        kinds=("function", "method"),
        require_valid=("goid_h128", "rel_path"),
    )
    for row in iter_rows(filtered_goids):
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


__all__ = [
    "degree_dict",
    "load_function_metadata",
    "parse_block_idx",
    "prefilter_table",
]
