"""Unified registry for serving datasets and operations.

.. deprecated::
    This module is a compatibility shim. All functionality has been moved to
    ``codeintel.serving.operations.catalog``. Import from there directly.

This module re-exports dataflow graph building and dataset metadata functions
from the canonical ``operations.catalog`` module for backward compatibility.
"""

from __future__ import annotations

import warnings

# Re-export all symbols from the canonical location
from codeintel.serving.operations.catalog import (
    DatasetMeta,
    build_dataset_meta,
    build_serving_dataflow_graph,
    get_registry_operation,
    iter_graph_nodes,
    iter_operation_dataset_edges,
    iter_operation_graph_edges,
    iter_operation_nodes,
    iter_registry_operations,
)


def _emit_deprecation_warning() -> None:
    """Emit a deprecation warning for this module."""
    warnings.warn(
        "codeintel.serving.registry is deprecated. "
        "Import from codeintel.serving.operations.catalog instead.",
        DeprecationWarning,
        stacklevel=3,
    )


__all__ = [
    "DatasetMeta",
    "build_dataset_meta",
    "build_serving_dataflow_graph",
    "get_registry_operation",
    "iter_graph_nodes",
    "iter_operation_dataset_edges",
    "iter_operation_graph_edges",
    "iter_operation_nodes",
    "iter_registry_operations",
]
