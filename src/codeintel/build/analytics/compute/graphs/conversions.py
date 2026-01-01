"""ID conversion and normalization utilities.

This module provides functions for converting and normalizing node
identifiers across different representations (Decimal, int, str).

Note
----
For the canonical ``normalize_decimal_id`` function, import directly
from ``codeintel.core.data_models.ids``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from decimal import Decimal

    import networkx as nx

log = logging.getLogger(__name__)


def safe_float(value: float | str | Decimal | None) -> float | None:
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
    "safe_float",
]
