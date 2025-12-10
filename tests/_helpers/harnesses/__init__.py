"""Test harness infrastructure for plugin testing.

This package provides test harnesses for plugins with shared base
classes to reduce code duplication.
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

__all__ = [
    "AnalyticsPluginHarness",
    "BaseResultAssertions",
    "BaseTestHarness",
    "ResultLike",
    "coverage_plugin_harness",
    "data_models_plugin_harness",
    "entrypoints_plugin_harness",
    "graph_plugin_harness",
    "plugin_harness_with_packs",
]
