"""Adapter implementations for ingestion port protocols.

This package provides concrete implementations of the port protocols defined in
``codeintel.ingestion.ports``, connecting the pure domain logic to real
infrastructure (DuckDB, file system, external tools).

Port/Adapter Pattern
--------------------
These adapters follow the hexagonal (ports and adapters) architecture pattern:

- **Ports** (in ``ingestion/ports/``) define abstract interfaces for I/O
  operations needed by ingestion logic.
- **Adapters** (this package) implement those interfaces with concrete
  infrastructure.

This separation allows the ingestion domain logic to remain pure and testable
while infrastructure concerns are isolated to these adapter implementations.

Available Adapters
------------------
DuckDBStorageAdapter
    Implements ``IngestStoragePort`` using DuckDB via ``StorageGateway``.
    Routes writes/deletes through ``DuckDBPolicyBackend`` and reads through
    the gateway/ibis connection.

FilesystemDiscoveryAdapter
    Implements ``ModuleDiscoveryPort`` using file system scanning via
    ``SourceScanner``. Discovers Python modules by scanning directories
    with configurable scan profiles.

BuildToolAdapter
    Bridge from build protocols to ingestion ports. Connects Hamilton build
    context to ingestion storage and tool operations.

HashChangeDetectionAdapter
    Implements ``ChangeDetectionPort`` using Blake2b hash-based change
    detection. Computes file digests to detect modified sources.

ToolRunnerAdapter
    Implements ``ToolPort`` using ``ToolService`` for external tool
    execution (SCIP indexers, coverage tools, etc.).

When to Use Adapters vs Direct Gateway Access
---------------------------------------------
**Use adapters when:**

- You need the port interface for dependency injection or testing
- You're writing code that should be infrastructure-agnostic
- You want to leverage adapter-specific validation or transformation

**Use ``ctx.gateway`` directly when:**

- You need low-level database operations not covered by port interfaces
- You're in a Hamilton plugin and want simple row writes via ``ctx.write_table()``
- Performance is critical and you need direct ibis/SQL access

See Also
--------
- ``codeintel.ingestion.ports`` : Port protocol definitions
- ``codeintel.storage.gateway.StorageGateway`` : Direct database access
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
