"""Unified context protocols for CodeIntel pipelines.

This package provides protocol interfaces for execution contexts,
enabling type-safe access to runtime information across different
subsystems (analytics, graphs, ingestion).

Key Components
--------------
ExecutionContextProtocol
    Base protocol for all execution contexts.
StorageContextProtocol
    Protocol for contexts with storage gateway access.
SnapshotContextProtocol
    Protocol for contexts with snapshot reference.
ConfigContextProtocol
    Protocol for contexts with configuration access.
ResourceContextProtocol
    Protocol for contexts with resource registry access.
"""

from __future__ import annotations

from codeintel.core.context.protocol import (
    ConfigContextProtocol,
    ExecutionContextProtocol,
    ResourceContextProtocol,
    SnapshotContextProtocol,
    StorageContextProtocol,
)

__all__ = [
    "ConfigContextProtocol",
    "ExecutionContextProtocol",
    "ResourceContextProtocol",
    "SnapshotContextProtocol",
    "StorageContextProtocol",
]
