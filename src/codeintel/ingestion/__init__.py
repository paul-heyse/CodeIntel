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

**Compute** (pure domain logic - analogous to graphs/compute/):
- `AstExtractStep`: Python AST extraction
- `CstExtractStep`: LibCST concrete syntax tree extraction
- `DocstringsExtractStep`: Docstring parsing and persistence
- `TypingIngestStep`: Type annotation analysis
- `CoverageIngestStep`: Coverage data ingestion
- `TestsIngestStep`: Test results ingestion
- `ScipIngestStep`: SCIP symbol indexing
- `ConfigIngestStep`: Configuration file flattening
- `RepoScanStep`: Repository scanning and module discovery

**Core** (plugin infrastructure):
- `BaseIngestPlugin`: Abstract base for all plugins
- `TableWriterIngestPlugin`: Base for plugins that write tables
- `ConfiguredIngestPlugin`: Base for plugins with typed configuration
- `IngestExecutionContext`: Execution context for plugins

**Plugins** (orchestration layer):
- `IngestPluginProtocol`: Plugin interface
- `IngestPluginResult`: Execution result
- `RepoScanPlugin`, `AstExtractPlugin`, etc.: Class-based plugins

**Recipes** (composition):
- `IngestRecipe`: Declarative recipe definition
- `RecipeExecutor`: Recipe execution with stage orchestration

**Runtime** (execution infrastructure - analogous to graphs/runtime/):
- `IngestExecutorConfig`: Plugin execution configuration
- `IngestRuntimeTelemetry`: Telemetry collection and reporting
- `PluginExecutionPlan`: Execution planning and ordering
- `execute_plugin_batch`: Batch plugin execution

**Validation** (contract validation - analogous to graphs/validation/):
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

from codeintel.ingestion.adapters import (
    DuckDBStorageAdapter,
    FilesystemDiscoveryAdapter,
    HashChangeDetectionAdapter,
    ToolRunnerAdapter,
)

# Compute layer exports (pure domain logic - renamed from steps for alignment with graphs/compute)
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

# Core plugin infrastructure exports
from codeintel.ingestion.core import (
    BaseIngestPlugin,
    ConfiguredIngestPlugin,
    ConfiguredTableWriterPlugin,
    IngestExecutionContext,
    ResourceNotFoundError,
    TableWriterIngestPlugin,
    ToolDependentIngestPlugin,
    TrackerRequiringPlugin,
    ValidationResult,
)
from codeintel.ingestion.infrastructure.paths import (
    ensure_repo_root,
    normalize_rel_path,
    relpath_to_module,
    repo_relpath,
)

# Worker infrastructure exports
from codeintel.ingestion.infrastructure.workers import (
    AST_WORKER_CONFIG,
    CST_WORKER_CONFIG,
    WorkerConfig,
    create_executor,
    executor_factory,
    resolve_worker_count,
    worker_pool,
)

# Plugin architecture exports
from codeintel.ingestion.plugins import (
    DEFAULT_CONTEXT_MAPPINGS,
    DEFAULT_INGEST_PLUGINS,
    AstExtractPlugin,
    ColumnConstraint,
    ConfigFactory,
    ConfigIngestPlugin,
    ConfigMapping,
    ContractValidationResult,
    ContractViolation,
    CoverageIngestPlugin,
    CstExtractPlugin,
    DocstringsIngestPlugin,
    ForeignKeyConstraint,
    IngestContractSpec,
    IngestContractValidator,
    IngestIsolationKind,
    IngestPluginMetadata,
    IngestPluginPlan,
    IngestPluginProtocol,
    IngestPluginRegistry,
    IngestPluginResult,
    IngestPluginSkip,
    IngestResourceHints,
    IngestRuntimeScratch,
    IngestSeverity,
    IngestStage,
    PlanOptions,
    RepoScanPlugin,
    ScipIngestPlugin,
    TestsIngestPlugin,
    TypingIngestPlugin,
    foreign_key_contract,
    get_config_fields,
    get_ingest_registry,
    infer_config_mapping,
    list_ingest_plugins,
    not_null_contract,
    plan_ingest_plugins,
    register_class_based_plugins,
    register_ingest_plugin,
    row_count_contract,
)

# Port-Adapter architecture exports
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

# Recipe architecture exports
from codeintel.ingestion.recipes import (
    BUILTIN_RECIPES,
    IngestRecipe,
    RecipeExecutionResult,
    RecipeOptions,
    RecipeStage,
    RecipeStageResult,
    execute_recipe,
    get_builtin_recipe,
)

# Runtime infrastructure exports (analogous to graphs/runtime/)
from codeintel.ingestion.runtime import (
    IngestExecutorConfig,
    IngestPlanContext,
    IngestPluginSpan,
    IngestRunReport,
    IngestRuntimeTelemetry,
    PluginExecutionPlan,
    PluginExecutionRecord,
    PluginExecutionSettings,
    PluginFatalError,
    execute_plugin,
    execute_plugin_batch,
    get_ingest_telemetry,
    resolve_plugin_order,
)

# Tracker domain service exports (renamed from change_tracker for alignment with graphs/catalog)
from codeintel.ingestion.tracker import (
    ChangeTracker,
    ChangeTrackerDatasetView,
    IncrementalIngestOps,
    IncrementalIngestPolicy,
    SupportsFullRebuild,
    run_incremental_ingest,
)

# Validation framework exports (additional exports from validation/)
from codeintel.ingestion.validation import (
    IngestValidationOptions,
    run_ingest_validations,
)

__all__ = [
    "AST_WORKER_CONFIG",
    "BUILTIN_RECIPES",
    "CST_WORKER_CONFIG",
    "DEFAULT_CONTEXT_MAPPINGS",
    "DEFAULT_INGEST_PLUGINS",
    "AstExtractPlugin",
    "AstExtractStep",
    "BaseIngestPlugin",
    "BatchResult",
    "ChangeDetectionPort",
    "ChangeRequest",
    "ChangeSet",
    "ChangeTracker",
    "ChangeTrackerDatasetView",
    "ColumnConstraint",
    "ConfigFactory",
    "ConfigIngestPlugin",
    "ConfigIngestStep",
    "ConfigMapping",
    "ConfiguredIngestPlugin",
    "ConfiguredTableWriterPlugin",
    "ContractValidationResult",
    "ContractViolation",
    "CoverageFileData",
    "CoverageIngestPlugin",
    "CoverageIngestStep",
    "CoverageResult",
    "CstExtractPlugin",
    "CstExtractStep",
    "DiagnosticResult",
    "DocstringsExtractStep",
    "DocstringsIngestPlugin",
    "DuckDBStorageAdapter",
    "FileDigest",
    "FilesystemDiscoveryAdapter",
    "ForeignKeyConstraint",
    "HashChangeDetectionAdapter",
    "IncrementalIngestOps",
    "IncrementalIngestPolicy",
    "IngestContractSpec",
    "IngestContractValidator",
    "IngestExecutionContext",
    "IngestExecutorConfig",
    "IngestIsolationKind",
    "IngestPlanContext",
    "IngestPluginMetadata",
    "IngestPluginPlan",
    "IngestPluginProtocol",
    "IngestPluginRegistry",
    "IngestPluginResult",
    "IngestPluginSkip",
    "IngestPluginSpan",
    "IngestRecipe",
    "IngestResourceHints",
    "IngestRunReport",
    "IngestRuntimeScratch",
    "IngestRuntimeTelemetry",
    "IngestSeverity",
    "IngestStage",
    "IngestStoragePort",
    "IngestToolPort",
    "IngestValidationOptions",
    "ModuleDiscoveryPort",
    "ModuleRecord",
    "PlanOptions",
    "PluginExecutionPlan",
    "PluginExecutionRecord",
    "PluginExecutionSettings",
    "PluginFatalError",
    "QueryResult",
    "RecipeExecutionResult",
    "RecipeOptions",
    "RecipeStage",
    "RecipeStageResult",
    "RepoScanPlugin",
    "RepoScanStep",
    "ResourceNotFoundError",
    "ScipIngestPlugin",
    "ScipIngestResult",
    "ScipIngestStep",
    "ScipResult",
    "StepResult",
    "SupportsFullRebuild",
    "TableWriterIngestPlugin",
    "TestResult",
    "TestsIngestPlugin",
    "TestsIngestStep",
    "ToolDependentIngestPlugin",
    "ToolRunnerAdapter",
    "TrackerRequiringPlugin",
    "TypingIngestPlugin",
    "TypingIngestStep",
    "ValidationResult",
    "WorkerConfig",
    "create_executor",
    "ensure_repo_root",
    "execute_plugin",
    "execute_plugin_batch",
    "execute_recipe",
    "executor_factory",
    "foreign_key_contract",
    "get_builtin_recipe",
    "get_config_fields",
    "get_ingest_registry",
    "get_ingest_telemetry",
    "infer_config_mapping",
    "list_ingest_plugins",
    "normalize_rel_path",
    "not_null_contract",
    "plan_ingest_plugins",
    "register_class_based_plugins",
    "register_ingest_plugin",
    "relpath_to_module",
    "repo_relpath",
    "resolve_plugin_order",
    "resolve_worker_count",
    "row_count_contract",
    "run_incremental_ingest",
    "run_ingest_validations",
    "worker_pool",
]
