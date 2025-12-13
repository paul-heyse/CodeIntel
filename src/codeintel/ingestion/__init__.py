"""Ingestion stages that parse repositories into normalized DuckDB tables for analytics.

This module provides the plugin-based ingestion architecture for CodeIntel.

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

**Plugins** (TargetPlugin implementations):
- `RepoScanPlugin`, `AstExtractPlugin`, etc.: Build system integrated plugins

**Validation** (contract validation):
- `IngestContractValidator`: Contract validation runner
- `IngestContractSpec`: Contract specification
- `ContractValidationResult`: Validation results
- `run_ingest_validations`: High-level validation entry point

Builtin Plugins
---------------
The following plugins are registered by default:
- `repo_scan` - Scans repository modules and builds change-tracker state
- `scip_ingest` - Runs scip-python for symbol indexing
- `cst_extract` - Parses CST via LibCST
- `ast_extract` - Parses Python AST for nodes and metrics
- `typing_ingest` - Computes typedness and static diagnostics
- `coverage_ingest` - Loads coverage.py data
- `tests_ingest` - Ingests pytest JSON reports
- `docstrings_ingest` - Extracts and persists docstrings
- `config_ingest` - Flattens configuration files
"""

from __future__ import annotations

# Note: Plugin imports are deferred to avoid circular imports.
# Use `from codeintel.build.plugins.ingestion import FooPlugin` directly.
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
from codeintel.ingestion.infrastructure.paths import (
    ensure_repo_root,
    normalize_rel_path,
    relpath_to_module,
    repo_relpath,
)
from codeintel.ingestion.infrastructure.scanning import (
    ScanProfile,
    default_code_profile,
    default_config_profile,
    profile_from_env,
)
from codeintel.ingestion.infrastructure.workers import (
    AST_WORKER_CONFIG,
    CST_WORKER_CONFIG,
    WorkerConfig,
    create_executor,
    executor_factory,
    resolve_worker_count,
    worker_pool,
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
    IncrementalIngestOps,
    IncrementalIngestPolicy,
    SupportsFullRebuild,
    run_incremental_ingest,
)
from codeintel.ingestion.validation import (
    ColumnConstraint,
    ContractValidationResult,
    ContractViolation,
    ForeignKeyConstraint,
    IngestContractSpec,
    IngestContractValidator,
    IngestValidationOptions,
    foreign_key_contract,
    not_null_contract,
    run_ingest_validations,
)

__all__ = [
    "AST_WORKER_CONFIG",
    "CST_WORKER_CONFIG",
    # Plugin classes moved to codeintel.build.plugins.ingestion to break circular imports.
    # Import directly: from codeintel.build.plugins.ingestion import AstExtractPlugin
    "AstExtractStep",
    "BatchResult",
    "BuildToolAdapter",
    "ChangeDetectionPort",
    "ChangeRequest",
    "ChangeSet",
    "ChangeTracker",
    "ChangeTrackerDatasetView",
    "ColumnConstraint",
    "ConfigIngestStep",
    "ContractValidationResult",
    "ContractViolation",
    "CoverageFileData",
    "CoverageIngestStep",
    "CoverageResult",
    "CstExtractStep",
    "DiagnosticResult",
    "DocstringsExtractStep",
    "DuckDBStorageAdapter",
    "FileDigest",
    "FilesystemDiscoveryAdapter",
    "ForeignKeyConstraint",
    "HashChangeDetectionAdapter",
    "IncrementalIngestOps",
    "IncrementalIngestPolicy",
    "IngestContractSpec",
    "IngestContractValidator",
    "IngestStoragePort",
    "IngestToolPort",
    "IngestValidationOptions",
    "ModuleDiscoveryPort",
    "ModuleRecord",
    "QueryResult",
    "RepoScanStep",
    "ScanProfile",
    "ScipIngestResult",
    "ScipIngestStep",
    "ScipResult",
    "StepResult",
    "SupportsFullRebuild",
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
    "foreign_key_contract",
    "normalize_rel_path",
    "not_null_contract",
    "profile_from_env",
    "relpath_to_module",
    "repo_relpath",
    "resolve_worker_count",
    "run_incremental_ingest",
    "run_ingest_validations",
    "worker_pool",
]
