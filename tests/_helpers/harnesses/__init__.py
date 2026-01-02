"""Test harness infrastructure for plugin and handler testing.

This package provides test harnesses for plugins and CLI handlers with
shared base classes to reduce code duplication.
"""

from __future__ import annotations

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
from tests._helpers.harnesses.graph_harness import (
    GraphTargetHarness,
)
from tests._helpers.harnesses.hamilton_build import (
    HamiltonBuildHarness,
    HarnessConfig,
    HarnessOpenOptions,
    RepoStrategy,
    RepoWriter,
)
from tests._helpers.harnesses.mcp_app import (
    McpAppHarness,
)
from tests._helpers.harnesses.plan_status import (
    PlanSummary,
    compute_plan_summary,
    compute_status_summary,
    format_plan_diff,
)
from tests._helpers.harnesses.serving_app import (
    ServingAppHarness,
)
from tests._helpers.harnesses.serving_harness import (
    ServingTargetHarness,
)
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
