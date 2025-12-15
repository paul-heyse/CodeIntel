"""Call graph edge deduplication utilities.

This module provides utilities for deduplicating call graph edges.
These functions are used by the callgraph builder and collection modules.

Note
----
The persistence functions (persist_call_graph_edges, persist_call_graph_nodes)
have been removed. Use ``ctx.write_table()`` in Hamilton plugins instead.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

    from codeintel.core.schemas.generated_types import CallGraphEdgeRow


def default_edge_key(row: CallGraphEdgeRow) -> tuple[object, ...]:
    """Build a dedupe key for call graph edges including repo/commit scope.

    Parameters
    ----------
    row
        Edge row to generate key for.

    Returns
    -------
    tuple[object, ...]
        Immutable key for deduplication.
    """
    return (
        row["repo"],
        row["commit"],
        row["caller_goid_h128"],
        row["callee_goid_h128"],
        row["callsite_path"],
        row["callsite_line"],
        row["callsite_col"],
    )


def dedupe_edge_rows(
    edges: list[CallGraphEdgeRow],
    key_fn: Callable[[CallGraphEdgeRow], tuple[object, ...]] | None = None,
) -> list[CallGraphEdgeRow]:
    """Remove duplicate edges using the provided key builder.

    Parameters
    ----------
    edges
        List of edges to deduplicate.
    key_fn
        Optional custom key builder function. Defaults to default_edge_key.

    Returns
    -------
    list[CallGraphEdgeRow]
        Unique edges preserving original order.
    """
    key_builder = key_fn or default_edge_key
    seen: set[tuple[object, ...]] = set()
    unique_edges: list[CallGraphEdgeRow] = []
    for row in edges:
        key = key_builder(row)
        if key in seen:
            continue
        seen.add(key)
        unique_edges.append(row)
    return unique_edges


__all__ = [
    "dedupe_edge_rows",
    "default_edge_key",
]
