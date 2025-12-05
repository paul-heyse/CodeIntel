"""Unified registry for serving datasets and operations.

.. deprecated:: 0.2.0
    This module will be removed in version 1.0.0.
    Import from ``codeintel.serving.operations.catalog`` instead.

This module re-exports dataflow graph building and dataset metadata functions
from the canonical ``operations.catalog`` module for backward compatibility.
All new code should import directly from ``codeintel.serving.operations.catalog``.

Migration
---------
Replace::

    from codeintel.serving.registry import build_serving_dataflow_graph

With::

    from codeintel.serving.operations.catalog import build_serving_dataflow_graph
"""

from __future__ import annotations

import warnings as _warnings

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

# Emit deprecation warning on first import (after imports to satisfy E402)
_warnings.warn(
    "codeintel.serving.registry is deprecated and will be removed in v1.0.0. "
    "Import from codeintel.serving.operations.catalog instead.",
    DeprecationWarning,
    stacklevel=2,
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
