"""Test harness infrastructure for plugin and handler testing.

This package provides test harnesses for plugins and CLI handlers with
shared base classes to reduce code duplication.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tests._helpers.harnesses.analytics import (
        AnalyticsPluginHarness,
        data_models_plugin_harness,
        entrypoints_plugin_harness,
        graph_plugin_harness,
        plugin_harness_with_packs,
    )
    from tests._helpers.harnesses.analytics_harness import (
        DEFAULT_ANALYTICS_TARGETS,
        AnalyticsTargetHarness,
    )
    from tests._helpers.harnesses.base import (
        BaseResultAssertions,
        BaseTestHarness,
        ResultLike,
    )
    from tests._helpers.harnesses.cli import (
        CliHandlerHarness,
        cli_handler_harness,
        core_handler_harness,
        graph_handler_harness,
        ops_handler_harness,
        storage_handler_harness,
        subsystem_handler_harness,
    )
    from tests._helpers.harnesses.datasets import DatasetHandlerHarness, dataset_handler_harness
    from tests._helpers.harnesses.docs import DocsHandlerHarness, docs_handler_harness
    from tests._helpers.harnesses.graph_harness import GraphTargetHarness
    from tests._helpers.harnesses.hamilton_build import (
        HamiltonBuildHarness,
        HarnessConfig,
        HarnessOpenOptions,
        RepoStrategy,
        RepoWriter,
    )
    from tests._helpers.harnesses.mcp_app import McpAppHarness
    from tests._helpers.harnesses.plan_status import (
        PlanSummary,
        compute_plan_summary,
        compute_status_summary,
        format_plan_diff,
    )
    from tests._helpers.harnesses.serving_app import ServingAppHarness
    from tests._helpers.harnesses.serving_harness import ServingTargetHarness
    from tests._helpers.harnesses.storage import StorageHandlerHarness, storage_macro_harness

__all__ = [
    "DEFAULT_ANALYTICS_TARGETS",
    "AnalyticsPluginHarness",
    "AnalyticsTargetHarness",
    "BaseResultAssertions",
    "BaseTestHarness",
    "CliHandlerHarness",
    "DatasetHandlerHarness",
    "DocsHandlerHarness",
    "GraphTargetHarness",
    "HamiltonBuildHarness",
    "HarnessConfig",
    "HarnessOpenOptions",
    "McpAppHarness",
    "PlanSummary",
    "RepoStrategy",
    "RepoWriter",
    "ResultLike",
    "ServingAppHarness",
    "ServingTargetHarness",
    "StorageHandlerHarness",
    "cli_handler_harness",
    "compute_plan_summary",
    "compute_status_summary",
    "core_handler_harness",
    "data_models_plugin_harness",
    "dataset_handler_harness",
    "docs_handler_harness",
    "entrypoints_plugin_harness",
    "format_plan_diff",
    "graph_handler_harness",
    "graph_plugin_harness",
    "ops_handler_harness",
    "plugin_harness_with_packs",
    "storage_handler_harness",
    "storage_macro_harness",
    "subsystem_handler_harness",
]

_LAZY_IMPORTS = {
    "DEFAULT_ANALYTICS_TARGETS": "tests._helpers.harnesses.analytics_harness",
    "AnalyticsPluginHarness": "tests._helpers.harnesses.analytics",
    "AnalyticsTargetHarness": "tests._helpers.harnesses.analytics_harness",
    "BaseResultAssertions": "tests._helpers.harnesses.base",
    "BaseTestHarness": "tests._helpers.harnesses.base",
    "CliHandlerHarness": "tests._helpers.harnesses.cli",
    "DatasetHandlerHarness": "tests._helpers.harnesses.datasets",
    "DocsHandlerHarness": "tests._helpers.harnesses.docs",
    "GraphTargetHarness": "tests._helpers.harnesses.graph_harness",
    "HamiltonBuildHarness": "tests._helpers.harnesses.hamilton_build",
    "HarnessConfig": "tests._helpers.harnesses.hamilton_build",
    "HarnessOpenOptions": "tests._helpers.harnesses.hamilton_build",
    "McpAppHarness": "tests._helpers.harnesses.mcp_app",
    "PlanSummary": "tests._helpers.harnesses.plan_status",
    "RepoStrategy": "tests._helpers.harnesses.hamilton_build",
    "RepoWriter": "tests._helpers.harnesses.hamilton_build",
    "ResultLike": "tests._helpers.harnesses.base",
    "ServingAppHarness": "tests._helpers.harnesses.serving_app",
    "ServingTargetHarness": "tests._helpers.harnesses.serving_harness",
    "StorageHandlerHarness": "tests._helpers.harnesses.storage",
    "cli_handler_harness": "tests._helpers.harnesses.cli",
    "compute_plan_summary": "tests._helpers.harnesses.plan_status",
    "compute_status_summary": "tests._helpers.harnesses.plan_status",
    "core_handler_harness": "tests._helpers.harnesses.cli",
    "data_models_plugin_harness": "tests._helpers.harnesses.analytics",
    "dataset_handler_harness": "tests._helpers.harnesses.datasets",
    "docs_handler_harness": "tests._helpers.harnesses.docs",
    "entrypoints_plugin_harness": "tests._helpers.harnesses.analytics",
    "format_plan_diff": "tests._helpers.harnesses.plan_status",
    "graph_handler_harness": "tests._helpers.harnesses.cli",
    "graph_plugin_harness": "tests._helpers.harnesses.analytics",
    "ops_handler_harness": "tests._helpers.harnesses.cli",
    "plugin_harness_with_packs": "tests._helpers.harnesses.analytics",
    "storage_handler_harness": "tests._helpers.harnesses.cli",
    "storage_macro_harness": "tests._helpers.harnesses.storage",
    "subsystem_handler_harness": "tests._helpers.harnesses.cli",
}


def __getattr__(name: str) -> object:
    if name in _LAZY_IMPORTS:
        module_name = _LAZY_IMPORTS[name]
        module = __import__(module_name, fromlist=[name])
        return getattr(module, name)
    message = f"module {__name__} has no attribute {name}"
    raise AttributeError(message)
