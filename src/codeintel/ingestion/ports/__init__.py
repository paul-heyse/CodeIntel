"""Port protocols for ingestion boundaries.

This package defines the port protocols that abstract I/O boundaries in the
ingestion system. Ports enable dependency injection and make domain logic
testable in isolation.

Port Categories
---------------
- Storage: Persisting data to database tables
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
from codeintel.ingestion.ports.storage import (
    BatchResult,
    IngestStoragePort,
    QueryResult,
)
from codeintel.ingestion.ports.tools import (
    CoverageFileData,
    CoverageResult,
    DiagnosticResult,
    IngestToolPort,
    ScipResult,
    TestResult,
)

__all__ = [
    "BatchResult",
    "ChangeDetectionPort",
    "ChangeRequest",
    "ChangeSet",
    "CoverageFileData",
    "CoverageResult",
    "DiagnosticResult",
    "FileDigest",
    "IngestStoragePort",
    "IngestToolPort",
    "ModuleDiscoveryPort",
    "ModuleRecord",
    "QueryResult",
    "ScipResult",
    "TestResult",
]
