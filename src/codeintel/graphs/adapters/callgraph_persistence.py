"""Persistence adapter for call graph edges.

This module provides utilities for deduplicating and persisting
call graph edges to the storage layer.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pandas as pd

from codeintel.config.datasets import (
    call_graph_edge_to_tuple,
    call_graph_node_to_tuple,
    dict_to_call_graph_edge,
    dict_to_call_graph_node,
)
from codeintel.ingestion.adapters import IngestStorageService
from codeintel.storage.pandera_schemas import validate_dataset_df

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from codeintel.config.datasets import (
        CallGraphEdgeRow,
    )
    from codeintel.storage.gateway import StorageGateway

log = logging.getLogger(__name__)


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


def _validate_rows(table_key: str, rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    """
    Validate rows using Pandera schema and convert to dict format.

    Parameters
    ----------
    table_key
        Table key for schema lookup.
    rows
        Rows to validate.

    Returns
    -------
    list[dict[str, object]]
        Validated rows as dictionaries.
    """
    if not rows:
        return []
    df = pd.DataFrame(rows)
    validated = validate_dataset_df(table_key, df)
    return validated.where(pd.notna(validated), None).to_dict(orient="records")


def persist_call_graph_edges(
    gateway: StorageGateway,
    edges: list[CallGraphEdgeRow],
    repo: str,
    commit: str,
    *,
    validate: bool = True,
) -> int:
    """
    Persist call graph edges after deduplication and validation.

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
    validate
        Whether to validate rows with Pandera schema.

    Returns
    -------
    int
        Number of edges persisted.
    """
    if not edges:
        return 0

    if validate:
        validated = _validate_rows("graph.call_graph_edges", list(edges))
        edges_to_persist = [
            {**e, "evidence_json": e.get("evidence_json") or "{}"} for e in validated
        ]
    else:
        edges_to_persist = list(edges)

    storage_service = IngestStorageService.from_gateway(gateway)
    storage_service.run_batch(
        "graph.call_graph_edges",
        [call_graph_edge_to_tuple(dict_to_call_graph_edge(e)) for e in edges_to_persist],
        delete_params=[repo, commit],
        scope="call_graph_edges",
    )
    return len(edges_to_persist)


def persist_call_graph_nodes(
    gateway: StorageGateway,
    nodes: Sequence[Mapping[str, object]],
    repo: str,
    commit: str,
    *,
    validate: bool = True,
) -> int:
    """
    Persist call graph nodes with optional validation.

    Parameters
    ----------
    gateway
        Storage gateway for database access.
    nodes
        List of nodes to persist.
    repo
        Repository identifier.
    commit
        Commit identifier.
    validate
        Whether to validate rows with Pandera schema.

    Returns
    -------
    int
        Number of nodes persisted.
    """
    if not nodes:
        return 0

    if validate:
        validated = _validate_rows("graph.call_graph_nodes", nodes)
        nodes_to_persist = validated
    else:
        nodes_to_persist = list(nodes)

    storage_service = IngestStorageService.from_gateway(gateway)
    storage_service.run_batch(
        "graph.call_graph_nodes",
        [call_graph_node_to_tuple(dict_to_call_graph_node(n)) for n in nodes_to_persist],
        delete_params=[repo, commit],
        scope="call_graph_nodes",
    )
    return len(nodes_to_persist)


__all__ = [
    "dedupe_edge_rows",
    "default_edge_key",
    "persist_call_graph_edges",
    "persist_call_graph_nodes",
]
