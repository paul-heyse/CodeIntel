"""Port protocols for ingestion boundaries.

This package defines the port protocols that abstract I/O boundaries in the
ingestion system. Ports enable dependency injection and make domain logic
testable in isolation.

Port Categories
---------------
- Tools: Executing external analysis tools
- Discovery: Finding and reading source modules
- ChangeDetection: Computing file changes between snapshots
"""

from __future__ import annotations

from codeintel.ingestion.ports.change_detection import (
    ChangeDetectionPort,
    ChangeRequest,
    ChangeSet,
    FileDigest,
)
from codeintel.ingestion.ports.discovery import (
    ModuleDiscoveryPort,
    ModuleRecord,
)
from codeintel.ingestion.ports.tools import (
    DiagnosticResult,
    IngestToolPort,
    ScipResult,
    TestResult,
)

__all__ = [
    "ChangeDetectionPort",
    "ChangeRequest",
    "ChangeSet",
    "DiagnosticResult",
    "FileDigest",
    "IngestToolPort",
    "ModuleDiscoveryPort",
    "ModuleRecord",
    "ScipResult",
    "TestResult",
]
