"""Adapter implementations for ingestion port protocols.

This package provides concrete implementations of the port protocols defined in
``codeintel.ingestion.ports``, connecting the pure domain logic to real
infrastructure (file system, external tools, dataset snapshots).

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
HashChangeDetectionAdapter
    Implements ``ChangeDetectionPort`` using Blake2b hash-based change
    detection with optional dataset snapshot lookups.

FilesystemDiscoveryAdapter
    Implements ``ModuleDiscoveryPort`` using file system scanning via
    ``SourceScanner``. Discovers Python modules by scanning directories
    with configurable scan profiles.

HashChangeDetectionAdapter
    Implements ``ChangeDetectionPort`` using Blake2b hash-based change
    detection. Computes file digests to detect modified sources.

See Also
--------
- ``codeintel.ingestion.ports`` : Port protocol definitions
"""

from __future__ import annotations

from codeintel.ingestion.adapters.filesystem_discovery import FilesystemDiscoveryAdapter
from codeintel.ingestion.adapters.hash_change_detection import HashChangeDetectionAdapter

__all__ = [
    "FilesystemDiscoveryAdapter",
    "HashChangeDetectionAdapter",
]
