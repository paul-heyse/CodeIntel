"""Persistence adapter for call graph edges.

This module provides utilities for deduplicating and persisting
call graph edges to the storage layer.
"""

from __future__ import annotations

from collections.abc import Callable

from codeintel.config.datasets import CallGraphEdgeRow, call_graph_edge_to_tuple
from codeintel.ingestion.common import run_batch
from codeintel.storage.gateway import StorageGateway


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


def persist_call_graph_edges(
    gateway: StorageGateway,
    edges: list[CallGraphEdgeRow],
    repo: str,
    commit: str,
) -> None:
    """Persist call graph edges after deduplication.

    Parameters
    ----------
    gateway
        Storage gateway for database access.
    edges
        List of edges to persist.
    repo
        Repository identifier.
    commit
        Commit identifier.
    """
    run_batch(
        gateway,
        "graph.call_graph_edges",
        [call_graph_edge_to_tuple(edge) for edge in edges],
        delete_params=[repo, commit],
        scope="call_graph_edges",
    )


__all__ = [
    "dedupe_edge_rows",
    "default_edge_key",
    "persist_call_graph_edges",
]
