"""ID conversion and normalization utilities.

This module provides functions for converting and normalizing node
identifiers across different representations (Decimal, int, str).
"""

from __future__ import annotations

import logging
from decimal import Decimal

import networkx as nx

log = logging.getLogger(__name__)


def to_decimal_id(value: int | str | Decimal | None) -> Decimal | None:
    """Coerce identifiers to Decimal for DuckDB writes.

    Parameters
    ----------
    value
        Identifier value to normalize.

    Returns
    -------
    Decimal | None
        Decimal-backed identifier or None when no value is provided.
    """
    if value is None:
        return None
    return Decimal(int(value))


def normalize_decimal_id(value: object) -> int | None:
    """Normalize DuckDB DECIMAL identifiers to integers.

    Parameters
    ----------
    value
        Raw identifier value sourced from DuckDB.

    Returns
    -------
    int | None
        Parsed integer when coercion succeeds, otherwise None.
    """
    result: int | None
    if value is None:
        result = None
    elif isinstance(value, int):
        result = value
    elif isinstance(value, Decimal):
        result = int(value)
    elif isinstance(value, (bytes, bytearray)):
        try:
            result = int(value.decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            result = None
    else:
        try:
            result = int(str(value))
        except (TypeError, ValueError):
            result = None
    return result


def normalize_node_id(node: Decimal | float | str | None) -> int | str | None:
    """Normalize graph node identifiers for consistent dictionary keys.

    Parameters
    ----------
    node
        Node identifier to normalize.

    Returns
    -------
    int | str | None
        Integer for numeric nodes (including Decimal or digit-only strings),
        otherwise stringified value; None is preserved.
    """
    result: int | str | None
    if node is None:
        result = None
    elif isinstance(node, Decimal):
        result = int(node)
    elif isinstance(node, (int, float)):
        try:
            result = int(node)
        except (TypeError, ValueError):
            result = None
    elif isinstance(node, str) and node.isdigit():
        result = int(node)
    else:
        result = str(node)
    return result


def safe_float(value: float | Decimal | str | None) -> float | None:
    """Coerce a value to float when possible.

    Parameters
    ----------
    value
        Input value to convert via float(). None returns None.

    Returns
    -------
    float | None
        Converted float when coercion succeeds, otherwise None.
    """
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def log_empty_graph(name: str, graph: nx.Graph) -> None:
    """Emit a debug log when a graph has no nodes.

    Parameters
    ----------
    name
        Graph name for the log message.
    graph
        Graph to check.
    """
    if graph.number_of_nodes() == 0:
        log.debug("Graph %s is empty; metrics will be zeroed", name)


def log_projection_skipped(label: str, reason: str, *, nodes: int, graph_nodes: int) -> None:
    """Log when a projection cannot be computed.

    This is informational, not an error - empty partitions are valid when
    no cross-partition relationships exist (e.g., config files that don't
    reference modules).

    Parameters
    ----------
    label
        Label for the projection.
    reason
        Reason the projection was skipped.
    nodes
        Number of nodes in the partition.
    graph_nodes
        Number of nodes in the bipartite graph.
    """
    log.info(
        "Skipping %s projection: %s (partition_size=%d graph_nodes=%d)",
        label,
        reason,
        nodes,
        graph_nodes,
    )


__all__ = [
    "log_empty_graph",
    "log_projection_skipped",
    "normalize_decimal_id",
    "normalize_node_id",
    "safe_float",
    "to_decimal_id",
]
