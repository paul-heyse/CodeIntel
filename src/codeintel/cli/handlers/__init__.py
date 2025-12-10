"""Unified CLI handlers package.

This package provides:

1. Base utilities (logging, context) in `handlers.base`
2. Protocol and enhanced context in `handlers.protocol`
3. Domain-specific handlers in `handlers.<domain>`

Examples
--------
>>> from codeintel.cli.handlers import setup_logging, build_handler_context
>>> setup_logging(verbosity=1)
>>> ctx = build_handler_context("build.run", {"target": "all"})
"""

from __future__ import annotations

from codeintel.cli.handlers.base import (
    HandlerContext,
    build_handler_context,
    get_handler_logger,
    open_handler_gateway,
    setup_logging,
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
    DatasetsListResult,
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
    DatasetListResult,
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
from codeintel.cli.handlers.protocol import (
    EnhancedHandlerContext,
    HandlerProtocol,
    handler_context,
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
    "BuildHistoryResult",
    "BuildRunResult",
    "BuildStatusResult",
    "DatasetDescribeResult",
    "DatasetDiffResult",
    "DatasetLintResult",
    "DatasetListResult",
    "DatasetSnapshotResult",
    "DatasetVerifyResult",
    "DatasetsListResult",
    "DocsExportResult",
    "DocsValidateResult",
    "EnhancedHandlerContext",
    "GenerateMacrosResult",
    "GraphPlanResult",
    "GraphPluginsResult",
    "HandlerContext",
    "HandlerProtocol",
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
    "build_handler_context",
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
    "get_handler_logger",
    "graph_plugins_list_handler",
    "graph_plugins_plan_handler",
    "handler_context",
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
    "setup_logging",
    "subsystem_coverage_handler",
    "subsystem_list_handler",
    "subsystem_module_memberships_handler",
    "subsystem_profiles_handler",
    "subsystem_show_handler",
    "validate_macros_handler",
]
