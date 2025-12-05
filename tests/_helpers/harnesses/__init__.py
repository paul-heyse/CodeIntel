"""Test harness infrastructure for plugin testing.

This package provides fluent test harnesses for analytics and graph
plugins, with shared base classes to reduce code duplication.

Example
-------
>>> from tests._helpers.harnesses import PluginTestHarness, GraphPluginTestHarness
>>> from tests._helpers.harnesses import assert_result
"""

from __future__ import annotations

from tests._helpers.harnesses.analytics import (
    PluginResultAssertions,
    PluginTestHarness,
    ValidationResultAssertions,
    assert_result,
    assert_validation,
)
from tests._helpers.harnesses.base import (
    BaseResultAssertions,
    BaseTestHarness,
    ResultLike,
)
from tests._helpers.harnesses.graphs import (
    GraphPluginTestHarness,
    NewPluginTestHarness,
)

__all__ = [
    "BaseResultAssertions",
    "BaseTestHarness",
    "GraphPluginTestHarness",
    "NewPluginTestHarness",
    "PluginResultAssertions",
    "PluginTestHarness",
    "ResultLike",
    "ValidationResultAssertions",
    "assert_result",
    "assert_validation",
]
