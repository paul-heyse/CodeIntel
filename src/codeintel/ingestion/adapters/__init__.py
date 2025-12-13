"""Adapter implementations for ingestion port protocols.

This package provides concrete implementations of the port protocols,
connecting the pure domain logic to real infrastructure:

- DuckDBStorageAdapter: DuckDB-specific storage operations
- ToolRunnerAdapter: External tool execution via ToolService
- BuildToolAdapter: Bridge from build protocols to ingestion ports
- FilesystemDiscoveryAdapter: File system module discovery
- HashChangeDetectionAdapter: Blake2b hash-based change detection
"""

from __future__ import annotations

from codeintel.ingestion.adapters.build_tool_adapter import BuildToolAdapter
from codeintel.ingestion.adapters.duckdb_storage import (
    DuckDBStorageAdapter,
)
from codeintel.ingestion.adapters.filesystem_discovery import FilesystemDiscoveryAdapter
from codeintel.ingestion.adapters.hash_change_detection import HashChangeDetectionAdapter
from codeintel.ingestion.adapters.tool_runner import ToolRunnerAdapter

__all__ = [
    "BuildToolAdapter",
    "DuckDBStorageAdapter",
    "FilesystemDiscoveryAdapter",
    "HashChangeDetectionAdapter",
    "ToolRunnerAdapter",
]
