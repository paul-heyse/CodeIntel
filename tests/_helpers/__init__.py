"""Shared test helpers package.

This package provides standardized test infrastructure for both analytics
and ingestion plugins, including:

- `PluginTestHarness`: Fluent test harness for analytics plugins
- `IngestPluginTestHarness`: Fluent test harness for ingestion plugins
- Various assertion helpers and test utilities
"""

from __future__ import annotations

from tests._helpers.ingest_plugin_harness import (
    IngestPluginResultAssertions,
    IngestPluginTestHarness,
    assert_ingest_result,
)
from tests._helpers.plugin_harness import (
    PluginResultAssertions,
    PluginTestHarness,
    ValidationResultAssertions,
    assert_result,
    assert_validation,
)

__all__ = [
    "IngestPluginResultAssertions",
    "IngestPluginTestHarness",
    "PluginResultAssertions",
    "PluginTestHarness",
    "ValidationResultAssertions",
    "assert_ingest_result",
    "assert_result",
    "assert_validation",
]
