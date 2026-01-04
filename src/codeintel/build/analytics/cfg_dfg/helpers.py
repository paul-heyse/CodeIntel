"""Shared helpers for CFG and DFG analytics.

This module consolidates common utility functions used by both cfg_core.py
and dfg_core.py to eliminate code duplication.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import pyarrow as pa

from codeintel.core.data_models.ids import normalize_decimal_id

if TYPE_CHECKING:
    from collections.abc import Iterable

    import networkx as nx


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
    for row in modules_frame.to_pylist():
        path = row.get("path")
        module = row.get("module")
        if isinstance(path, str) and isinstance(module, str):
            module_by_path[path] = module

    metadata: dict[int, tuple[str, str | None, str | None]] = {}
    for row in goids_frame.to_pylist():
        if row.get("repo") != repo or row.get("commit") != commit:
            continue
        if row.get("kind") not in {"function", "method"}:
            continue
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
]
