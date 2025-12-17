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
- `ToolRunnerAdapter`: External tool execution via ToolService
- `FilesystemDiscoveryAdapter`: File system module discovery
- `HashChangeDetectionAdapter`: Blake2b hash-based change detection
- `BuildToolAdapter`: Adapter bridging build system protocols to IngestToolPort

**Compute** (pure domain logic):
- `AstExtractStep`: Python AST extraction
- `CstExtractStep`: LibCST concrete syntax tree extraction
- `DocstringsExtractStep`: Docstring parsing and persistence
- `TypingIngestStep`: Type annotation analysis
- `CoverageIngestStep`: Coverage data ingestion
- `TestsIngestStep`: Test results ingestion
- `ScipIngestStep`: SCIP symbol indexing
- `ConfigIngestStep`: Configuration file flattening
- `RepoScanStep`: Repository scanning and module discovery

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
    BuildToolAdapter,
    DuckDBStorageAdapter,
    FilesystemDiscoveryAdapter,
    HashChangeDetectionAdapter,
    ToolRunnerAdapter,
)
from codeintel.ingestion.compute import (
    AstExtractStep,
    ConfigIngestStep,
    CoverageIngestStep,
    CstExtractStep,
    DocstringsExtractStep,
    RepoScanStep,
    ScipIngestResult,
    ScipIngestStep,
    StepResult,
    TestsIngestStep,
    TypingIngestStep,
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
from codeintel.ingestion.tracker import (
    ChangeTracker,
    ChangeTrackerDatasetView,
)

__all__ = [
    "AstExtractStep",
    "BatchResult",
    "BuildToolAdapter",
    "ChangeDetectionPort",
    "ChangeRequest",
    "ChangeSet",
    "ChangeTracker",
    "ChangeTrackerDatasetView",
    "ConfigIngestStep",
    "CoverageFileData",
    "CoverageIngestStep",
    "CoverageResult",
    "CstExtractStep",
    "DiagnosticResult",
    "DocstringsExtractStep",
    "DuckDBStorageAdapter",
    "FileDigest",
    "FilesystemDiscoveryAdapter",
    "HashChangeDetectionAdapter",
    "IngestStoragePort",
    "IngestToolPort",
    "ModuleDiscoveryPort",
    "ModuleRecord",
    "QueryResult",
    "RepoScanStep",
    "ScanProfile",
    "ScipIngestResult",
    "ScipIngestStep",
    "ScipResult",
    "StepResult",
    "TestResult",
    "TestsIngestStep",
    "ToolRunnerAdapter",
    "TypingIngestStep",
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
