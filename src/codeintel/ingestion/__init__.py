"""Ingestion stages that parse repositories into normalized DuckDB tables for analytics.

This module provides the plugin-based ingestion architecture for CodeIntel.

Architecture Overview
---------------------
- Plugin protocol: `IngestPluginProtocol`, `IngestPluginContext`, `IngestPluginResult`
- Registry: `get_ingest_registry()`, `register_ingest_plugin()`, `list_ingest_plugins()`
- Decorator: `@ingest_plugin` for declarative plugin definition
- Recipes: `IngestRecipe`, `execute_recipe()`
- Harness: `HarnessConfig`, `IngestExecutionHarness` for reducing plugin boilerplate
- Pipeline: `IngestPipeline`, `PipelineExecutor` for unified incremental/full ingestion

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
    # Worker infrastructure
    "AST_WORKER_CONFIG",
    "BUILTIN_RECIPES",
    "CST_WORKER_CONFIG",
    "DEFAULT_CONTEXT_MAPPINGS",
    "DEFAULT_INGEST_PLUGINS",
    "ClassBasedIngestPlugin",
    "ColumnConstraint",
    "ConfigFactory",
    "ConfigMapping",
    "ContractValidationResult",
    "ContractViolation",
    "ForeignKeyConstraint",
    "FunctionalIngestPlugin",
    "HarnessConfig",
    "HarnessContext",
    "IngestContractSpec",
    "IngestContractValidator",
    "IngestExecutionHarness",
    "IngestIsolationKind",
    # Pipeline architecture
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
    "PipelineConfig",
    "PipelineExecutor",
    "PipelineResult",
    "RecipeExecutionResult",
    "RecipeOptions",
    "RecipeStage",
    "RecipeStageResult",
    "SupportsFullRebuild",
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
