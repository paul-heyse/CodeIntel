"""Ingestion stages that parse repositories into normalized DuckDB tables for analytics.

Architecture Overview
---------------------
The ingestion system follows a port-adapter pattern for clean separation of concerns:

**Ports** (interfaces):
- `IngestStoragePort`: Database operations (write, delete, query)
- `IngestToolPort`: External tool execution (pyright, scip, pytest)
- `ModuleDiscoveryPort`: Source file enumeration
- `ChangeDetectionPort`: Incremental change tracking

**Adapters** (implementations):
- `DuckDBStorageAdapter`: DuckDB-specific storage operations
- `FilesystemDiscoveryAdapter`: File system module discovery
- `HashChangeDetectionAdapter`: Blake2b hash-based change detection

**Compute** (pure domain logic):
- Ingestion compute steps live under `codeintel.ingestion.compute` and are
  intended for internal use by Hamilton native targets.

"""

from __future__ import annotations

from codeintel.core.concurrency import (
    WorkerConfig,
    create_executor,
    executor_factory,
    resolve_worker_count,
    worker_pool,
)
from codeintel.core.paths import (
    ensure_repo_root,
    repo_relpath,
)
from codeintel.ingestion.adapters import (
    DuckDBStorageAdapter,
    FilesystemDiscoveryAdapter,
    HashChangeDetectionAdapter,
)
from codeintel.ingestion.infrastructure import normalize_rel_path
from codeintel.ingestion.infrastructure.scanning import (
    ScanProfile,
    default_code_profile,
    default_config_profile,
    profile_from_env,
)
from codeintel.ingestion.ports import (
    BatchResult,
    ChangeDetectionPort,
    ChangeRequest,
    ChangeSet,
    CoverageFileData,
    CoverageResult,
    DiagnosticResult,
    FileDigest,
    IngestStoragePort,
    IngestToolPort,
    ModuleDiscoveryPort,
    ModuleRecord,
    QueryResult,
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
    "DuckDBStorageAdapter",
    "FileDigest",
    "FilesystemDiscoveryAdapter",
    "HashChangeDetectionAdapter",
    "IngestStoragePort",
    "IngestToolPort",
    "ModuleDiscoveryPort",
    "ModuleRecord",
    "QueryResult",
    "ScanProfile",
    "ScipResult",
    "TestResult",
    "WorkerConfig",
    "create_executor",
    "default_code_profile",
    "default_config_profile",
    "ensure_repo_root",
    "executor_factory",
    "normalize_rel_path",
    "profile_from_env",
    "repo_relpath",
    "resolve_worker_count",
    "worker_pool",
]
