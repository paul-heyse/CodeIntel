"""Test harness infrastructure for plugin and handler testing.

This package provides test harnesses for plugins and CLI handlers with
shared base classes to reduce code duplication.
"""

from __future__ import annotations

from tests._helpers.harnesses.analytics import (
    AnalyticsPluginHarness,
    coverage_plugin_harness,
    data_models_plugin_harness,
    entrypoints_plugin_harness,
    graph_plugin_harness,
    plugin_harness_with_packs,
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

__all__ = [
    "AnalyticsPluginHarness",
    "BaseResultAssertions",
    "BaseTestHarness",
    "CliHandlerHarness",
    "ResultLike",
    "cli_handler_harness",
    "core_handler_harness",
    "coverage_plugin_harness",
    "data_models_plugin_harness",
    "entrypoints_plugin_harness",
    "graph_handler_harness",
    "graph_plugin_harness",
    "ops_handler_harness",
    "plugin_harness_with_packs",
    "storage_handler_harness",
    "subsystem_handler_harness",
]
