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

**Steps** (pure domain logic):
- `AstExtractStep`: Python AST extraction
- `CstExtractStep`: LibCST concrete syntax tree extraction
- `DocstringsExtractStep`: Docstring parsing and persistence
- `TypingIngestStep`: Type annotation analysis
- `CoverageIngestStep`: Coverage data ingestion
- `TestsIngestStep`: Test results ingestion
- `ScipIngestStep`: SCIP symbol indexing
- `ConfigIngestStep`: Configuration file flattening
- `RepoScanStep`: Repository scanning and change detection

**Plugins** (orchestration layer):
- `IngestPluginProtocol`: Plugin interface
- `IngestPluginContext`: Execution context
- `IngestPluginResult`: Execution result
- `@ingest_plugin`: Decorator for plugin definition

**Recipes** (composition):
- `IngestRecipe`: Declarative recipe definition
- `RecipeExecutor`: Recipe execution with stage orchestration

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

from codeintel.ingestion.adapters import (
    DuckDBStorageAdapter,
    FilesystemDiscoveryAdapter,
    HashChangeDetectionAdapter,
    ToolRunnerAdapter,
)
from codeintel.ingestion.paths import (
    ensure_repo_root,
    normalize_rel_path,
    relpath_to_module,
    repo_relpath,
)

# Pipeline architecture exports
from codeintel.ingestion.pipeline import (
    IngestPipeline,
    PipelineConfig,
    PipelineExecutor,
    PipelineResult,
    SupportsFullRebuild,
    execute_pipeline,
)

# Plugin architecture exports
from codeintel.ingestion.plugins import (
    DEFAULT_CONTEXT_MAPPINGS,
    DEFAULT_INGEST_PLUGINS,
    ClassBasedIngestPlugin,
    ColumnConstraint,
    ConfigFactory,
    ConfigMapping,
    ContractValidationResult,
    ContractViolation,
    ForeignKeyConstraint,
    FunctionalIngestPlugin,
    HarnessConfig,
    HarnessContext,
    IngestContractSpec,
    IngestContractValidator,
    IngestExecutionHarness,
    IngestIsolationKind,
    IngestPluginContext,
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
    foreign_key_contract,
    get_config_fields,
    get_ingest_registry,
    infer_config_mapping,
    ingest_plugin,
    list_ingest_plugins,
    not_null_contract,
    plan_ingest_plugins,
    register_class_plugin,
    register_ingest_plugin,
    row_count_contract,
    with_harness,
)

# Port-Adapter architecture exports
from codeintel.ingestion.ports import (
    BatchResult,
    ChangeDetectionPort,
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
from codeintel.ingestion.steps import (
    AstExtractStep,
    ConfigIngestStep,
    CoverageIngestStep,
    CstExtractStep,
    DocstringsExtractStep,
    RepoScanStep,
    ScipIngestStep,
    StepResult,
    TestsIngestStep,
    TypingIngestStep,
)

# Worker infrastructure exports
from codeintel.ingestion.workers import (
    AST_WORKER_CONFIG,
    CST_WORKER_CONFIG,
    WorkerConfig,
    create_executor,
    executor_factory,
    resolve_worker_count,
    worker_pool,
)

__all__ = [
    "AST_WORKER_CONFIG",
    "BUILTIN_RECIPES",
    "CST_WORKER_CONFIG",
    "DEFAULT_CONTEXT_MAPPINGS",
    "DEFAULT_INGEST_PLUGINS",
    "AstExtractStep",
    "BatchResult",
    "ChangeDetectionPort",
    "ChangeSet",
    "ClassBasedIngestPlugin",
    "ColumnConstraint",
    "ConfigFactory",
    "ConfigIngestStep",
    "ConfigMapping",
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
    "FunctionalIngestPlugin",
    "HarnessConfig",
    "HarnessContext",
    "HashChangeDetectionAdapter",
    "IngestContractSpec",
    "IngestContractValidator",
    "IngestExecutionHarness",
    "IngestIsolationKind",
    "IngestPipeline",
    "IngestPluginContext",
    "IngestPluginMetadata",
    "IngestPluginPlan",
    "IngestPluginProtocol",
    "IngestPluginRegistry",
    "IngestPluginResult",
    "IngestPluginSkip",
    "IngestRecipe",
    "IngestResourceHints",
    "IngestRuntimeScratch",
    "IngestSeverity",
    "IngestStage",
    "IngestStoragePort",
    "IngestToolPort",
    "ModuleDiscoveryPort",
    "ModuleRecord",
    "PipelineConfig",
    "PipelineExecutor",
    "PipelineResult",
    "QueryResult",
    "RecipeExecutionResult",
    "RecipeOptions",
    "RecipeStage",
    "RecipeStageResult",
    "RepoScanStep",
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
    "ensure_repo_root",
    "execute_pipeline",
    "execute_recipe",
    "executor_factory",
    "foreign_key_contract",
    "get_builtin_recipe",
    "get_config_fields",
    "get_ingest_registry",
    "infer_config_mapping",
    "ingest_plugin",
    "list_ingest_plugins",
    "normalize_rel_path",
    "not_null_contract",
    "plan_ingest_plugins",
    "register_class_plugin",
    "register_ingest_plugin",
    "relpath_to_module",
    "repo_relpath",
    "resolve_worker_count",
    "row_count_contract",
    "with_harness",
    "worker_pool",
]
