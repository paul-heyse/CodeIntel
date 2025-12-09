"""Central operation registration.

This module imports all operation modules to trigger registration
of their OperationSpecs with the global registry.
"""

from __future__ import annotations

# Import modules to trigger registration
# These imports are used for their side effects (registering operations)
from codeintel.cli.operations import (
    build_operations,
    dataset_operations,
    docs_operations,
    graph_operations,
    history_operations,
    ide_operations,
    op_operations,
    storage_operations,
    subsystem_operations,
)

__all__ = [
    "build_operations",
    "dataset_operations",
    "docs_operations",
    "graph_operations",
    "history_operations",
    "ide_operations",
    "op_operations",
    "storage_operations",
    "subsystem_operations",
]
