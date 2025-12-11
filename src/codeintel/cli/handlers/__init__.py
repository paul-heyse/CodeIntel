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
>>> bootstrap_cli(verbosity=1)  # doctest: +SKIP
<codeintel.cli.config.model.CliConfig object at ...>
"""

from __future__ import annotations

# Bootstrap for logging/signal setup
from codeintel.cli.execution.bootstrap import bootstrap_cli

# Utilities
from codeintel.cli.handlers._utilities import (
    get_handler_logger,
    open_handler_gateway,
)

# Domain handlers and result types
from codeintel.cli.handlers.build import (
    BuildHistoryResult,
    BuildRunResult,
    BuildStatusResult,
    build_history_handler,
    build_run_handler,
    build_status_handler,
)

# Legacy context types (for backward compatibility during transition)
# New code should use CommandContext from codeintel.cli.context
from codeintel.cli.handlers.context import (
    HandlerContext,
    HandlerContextOptions,
    ParameterError,
    handler_context_manager,
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
    docs_export_handler,
    docs_validate_handler,
)
from codeintel.cli.handlers.graphs import (
    GraphPlanResult,
    GraphPluginsResult,
    graph_plugins_list_handler,
    graph_plugins_plan_handler,
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
from codeintel.cli.handlers.ide import (
    IdeHintsResult,
    ide_hints_handler,
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
    OperationCallResult,
    OperationListResult,
    ServeStartResult,
    dataset_describe_handler,
    dataset_list_handler,
    dataset_verify_handler,
    op_call_handler,
    op_list_handler,
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
    MacroRequirement,
    ProfileStorageResult,
    ValidateMacrosResult,
    generate_macros_handler,
    profile_storage_handler,
    validate_macros_handler,
)
from codeintel.cli.handlers.subsystem import (
    SubsystemCoverageResult,
    SubsystemListResult,
    SubsystemMembershipResult,
    SubsystemProfilesResult,
    SubsystemShowResult,
    subsystem_coverage_handler,
    subsystem_list_handler,
    subsystem_module_memberships_handler,
    subsystem_profiles_handler,
    subsystem_show_handler,
)

__all__ = [
    # Result types
    "BuildHistoryResult",
    "BuildRunResult",
    "BuildStatusResult",
    "DatasetDescribeResult",
    "DatasetDiffResult",
    "DatasetLintResult",
    "DatasetListResult",
    "DatasetSnapshotResult",
    "DatasetVerifyResult",
    "DocsExportResult",
    "DocsValidateResult",
    "GenerateMacrosResult",
    "GraphPlanResult",
    "GraphPluginsResult",
    # Context types
    "HandlerContext",
    "HandlerContextOptions",
    "HealthCheckResult",
    "HistoryTimeseriesResult",
    "IdeHintsResult",
    "JobCancelResult",
    "JobOutputResult",
    "JobStatusResult",
    "JobsCleanupResult",
    "JobsListResult",
    "MacroRequirement",
    "OperationCallResult",
    "OperationListResult",
    "ParameterError",
    "PluginInfoResult",
    "PluginNewResult",
    "PluginPathsResult",
    "PluginTestResult",
    "PluginValidateResult",
    "PluginsDiscoverResult",
    "PluginsListResult",
    "ProfileStorageResult",
    "ServeStartResult",
    "SubsystemCoverageResult",
    "SubsystemListResult",
    "SubsystemMembershipResult",
    "SubsystemProfilesResult",
    "SubsystemShowResult",
    "ValidateMacrosResult",
    "bootstrap_cli",
    # Handler functions
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
    "generate_macros_handler",
    # Utilities
    "get_handler_logger",
    "graph_plugins_list_handler",
    "graph_plugins_plan_handler",
    "handler_context_manager",
    "health_check_handler",
    "history_timeseries_handler",
    "ide_hints_handler",
    "is_health_check_passing",
    "jobs_cancel_handler",
    "jobs_cleanup_handler",
    "jobs_list_handler",
    "jobs_output_handler",
    "jobs_status_handler",
    "op_call_handler",
    "op_list_handler",
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
    "subsystem_coverage_handler",
    "subsystem_list_handler",
    "subsystem_module_memberships_handler",
    "subsystem_profiles_handler",
    "subsystem_show_handler",
    "validate_macros_handler",
]
