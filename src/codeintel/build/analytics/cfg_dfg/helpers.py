"""Shared helpers for CFG and DFG analytics.

This module consolidates common utility functions used by both cfg_core.py
and dfg_core.py to eliminate code duplication.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from codeintel.core.data_models.ids import normalize_decimal_id
from codeintel.storage.constants import DEFAULT_ARROW_BATCH_SIZE
from codeintel.storage.query_results import (
    coerce_optional_str,
    coerce_str,
    iter_tuples_from_arrow_reader,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

    import networkx as nx

    from codeintel.storage.gateway import StorageGateway


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
    gateway: StorageGateway, repo: str, commit: str
) -> dict[int, tuple[str, str | None, str | None]]:
    """Load function metadata keyed by GOID.

    Parameters
    ----------
    gateway
        Storage gateway for database access.
    repo
        Repository identifier.
    commit
        Commit identifier.

    Returns
    -------
    dict[int, tuple[str, str | None, str | None]]
        Mapping of GOID -> (rel_path, module, qualname).
    """
    reader = gateway.execute(
        """
        SELECT g.goid_h128,
               g.rel_path,
               m.module,
               g.qualname
        FROM core.goids g
        LEFT JOIN core.modules m
          ON m.path = g.rel_path
        WHERE g.repo = ? AND g.commit = ?
          AND g.kind IN ('function', 'method')
        """,
        [repo, commit],
    ).fetch_record_batch(DEFAULT_ARROW_BATCH_SIZE)
    result: dict[int, tuple[str, str | None, str | None]] = {}
    for goid_raw, rel_path, module, qualname in iter_tuples_from_arrow_reader(reader):
        goid = normalize_decimal_id(goid_raw)
        if goid is None:
            continue
        rel_path_text = coerce_str(rel_path, ctx="function_metadata.rel_path")
        module_text = coerce_optional_str(module, ctx="function_metadata.module")
        qualname_text = coerce_optional_str(qualname, ctx="function_metadata.qualname")
        result[goid] = (rel_path_text, module_text, qualname_text)
    return result


__all__ = [
    "degree_dict",
    "load_function_metadata",
    "parse_block_idx",
]
