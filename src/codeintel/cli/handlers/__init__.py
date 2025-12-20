"""Unified CLI handlers package.

This package provides business logic handlers for CLI commands.

All handlers use CommandContext (from codeintel.cli.context) for unified
access to runtime, storage, gateway, params, and service layers.

Command[T] Pattern (for commands with parameters):
    Commands like jobs, health, plugins, graphs use `Command[T]` base class
    with an `execute(ctx: CommandContext)` method.

Handler Function Pattern (for complex operations):
    Handler functions receive CommandContext and perform business logic
    for domains like build, datasets, storage, subsystem, docs, etc.

Components:
    - handlers._utilities: Shared utilities (gateway opening, logging)
    - handlers.<domain>: Domain-specific handler functions

Examples
--------
>>> from codeintel.cli.context import CommandContext
>>> from codeintel.cli.handlers.build import build_status_handler
>>> from codeintel.cli.execution.bootstrap import bootstrap_cli
>>> bootstrap_cli(verbosity=1)
<codeintel.cli.config.model.CliConfig object at ...>
"""

from __future__ import annotations

from codeintel.cli.context import CommandContext
from codeintel.cli.execution.bootstrap import bootstrap_cli
from codeintel.cli.handlers._utilities import (
    get_handler_logger,
    open_handler_gateway,
)
from codeintel.cli.handlers.build import (
    BuildHistoryResult,
    BuildRunResult,
    BuildStatusResult,
    build_history_handler,
    build_run_handler,
    build_status_handler,
)
from codeintel.cli.handlers.datasets import (
    DatasetDiffResult,
    DatasetLintResult,
    DatasetListResult,
    DatasetSnapshotResult,
    datasets_diff_handler,
    datasets_lint_handler,
    datasets_list_handler,
    datasets_snapshot_handler,
)
from codeintel.cli.handlers.docs import (
    DocsExportResult,
    DocsValidateResult,
    ExportMode,
    docs_export_handler,
    docs_validate_handler,
)
from codeintel.cli.handlers.graphs import (
    GraphPlanResult,
    GraphPlanStage,
    GraphTargetInfo,
    GraphTargetsResult,
    graph_targets_list_handler,
    graph_targets_plan_handler,
)
from codeintel.cli.handlers.health import (
    HealthCheckResult,
    health_check_handler,
    is_health_check_passing,
)
from codeintel.cli.handlers.history import (
    HistoryTimeseriesResult,
    history_timeseries_handler,
)
from codeintel.cli.handlers.jobs import (
    JobCancelResult,
    JobOutputResult,
    JobsCleanupResult,
    JobsListResult,
    JobStatusResult,
    jobs_cancel_handler,
    jobs_cleanup_handler,
    jobs_list_handler,
    jobs_output_handler,
    jobs_status_handler,
)
from codeintel.cli.handlers.ops import (
    DatasetDescribeResult,
    DatasetVerifyResult,
    ServeStartResult,
    dataset_describe_handler,
    dataset_list_handler,
    dataset_verify_handler,
    serve_http_handler,
    serve_mcp_handler,
)
from codeintel.cli.handlers.plugins import (
    PluginInfoResult,
    PluginNewResult,
    PluginPathsResult,
    PluginsDiscoverResult,
    PluginsListResult,
    PluginTestResult,
    PluginValidateResult,
    plugins_discover_handler,
    plugins_info_handler,
    plugins_list_handler,
    plugins_new_handler,
    plugins_paths_handler,
    plugins_test_handler,
    plugins_validate_handler,
)
from codeintel.cli.handlers.storage import (
    GenerateMacrosResult,
    ProfileStorageResult,
    StorageDatabaseExportResult,
    StorageDatabaseImportResult,
    ValidateMacrosResult,
    export_database_handler,
    generate_macros_handler,
    import_database_handler,
    profile_storage_handler,
    validate_macros_handler,
)

__all__ = [
    "BuildHistoryResult",
    "BuildRunResult",
    "BuildStatusResult",
    "CommandContext",
    "DatasetDescribeResult",
    "DatasetDiffResult",
    "DatasetLintResult",
    "DatasetListResult",
    "DatasetSnapshotResult",
    "DatasetVerifyResult",
    "DocsExportResult",
    "DocsValidateResult",
    "ExportMode",
    "GenerateMacrosResult",
    "GraphPlanResult",
    "GraphPlanStage",
    "GraphTargetInfo",
    "GraphTargetsResult",
    "HealthCheckResult",
    "HistoryTimeseriesResult",
    "JobCancelResult",
    "JobOutputResult",
    "JobStatusResult",
    "JobsCleanupResult",
    "JobsListResult",
    "PluginInfoResult",
    "PluginNewResult",
    "PluginPathsResult",
    "PluginTestResult",
    "PluginValidateResult",
    "PluginsDiscoverResult",
    "PluginsListResult",
    "ProfileStorageResult",
    "ServeStartResult",
    "StorageDatabaseExportResult",
    "StorageDatabaseImportResult",
    "ValidateMacrosResult",
    "bootstrap_cli",
    "build_history_handler",
    "build_run_handler",
    "build_status_handler",
    "dataset_describe_handler",
    "dataset_list_handler",
    "dataset_verify_handler",
    "datasets_diff_handler",
    "datasets_lint_handler",
    "datasets_list_handler",
    "datasets_snapshot_handler",
    "docs_export_handler",
    "docs_validate_handler",
    "export_database_handler",
    "generate_macros_handler",
    "get_handler_logger",
    "graph_targets_list_handler",
    "graph_targets_plan_handler",
    "health_check_handler",
    "history_timeseries_handler",
    "import_database_handler",
    "is_health_check_passing",
    "jobs_cancel_handler",
    "jobs_cleanup_handler",
    "jobs_list_handler",
    "jobs_output_handler",
    "jobs_status_handler",
    "open_handler_gateway",
    "plugins_discover_handler",
    "plugins_info_handler",
    "plugins_list_handler",
    "plugins_new_handler",
    "plugins_paths_handler",
    "plugins_test_handler",
    "plugins_validate_handler",
    "profile_storage_handler",
    "serve_http_handler",
    "serve_mcp_handler",
    "validate_macros_handler",
]
