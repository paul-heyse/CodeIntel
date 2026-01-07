"""Conversion and logging utilities for graph analytics."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from codeintel.build.graphs.rx.algos import GraphInput, graph_node_count

if TYPE_CHECKING:
    from decimal import Decimal

log = logging.getLogger(__name__)


def safe_float(value: float | str | Decimal | None) -> float | None:
    """Coerce a value to float when possible.

    Returns
    -------
    float | None
        Float value when coercion succeeds, otherwise None.
    """
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def log_empty_graph(name: str, graph: GraphInput) -> None:
    """Emit a debug log when a graph has no nodes."""
    if graph_node_count(graph) == 0:
        log.debug("Graph %s is empty; metrics will be zeroed", name)


def log_projection_skipped(label: str, reason: str, *, nodes: int, graph_nodes: int) -> None:
    """Log when a projection cannot be computed."""
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
