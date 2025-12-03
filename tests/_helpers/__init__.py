"""Shared test helpers package.

This package provides standardized test infrastructure for both analytics
and ingestion plugins, including:

- `PluginTestHarness`: Fluent test harness for analytics plugins
- `IngestPluginTestHarness`: Fluent test harness for ingestion plugins
- `TestContext`: Unified test environment for hexagonal architecture
- `TestScenario`: Declarative scenario builder
- Seed packs for composable test data
- Various assertion helpers and test utilities
"""

from __future__ import annotations

from tests._helpers.context import (
    DEFAULT_COMMIT,
    DEFAULT_REPO,
    QueryRow,
    SeedPack,
    TestContext,
    create_test_context,
)
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
from tests._helpers.scenarios import (
    ScenarioConfig,
    TestScenario,
    coverage_context,
    full_context,
    graph_context,
    minimal_context,
)
from tests._helpers.seeds import (
    CORE_PACK,
    COVERAGE_PACK,
    GRAPH_PACK,
    METRICS_PACK,
)

__all__ = [
    "CORE_PACK",
    "COVERAGE_PACK",
    "DEFAULT_COMMIT",
    "DEFAULT_REPO",
    "GRAPH_PACK",
    "METRICS_PACK",
    "IngestPluginResultAssertions",
    "IngestPluginTestHarness",
    "PluginResultAssertions",
    "PluginTestHarness",
    "QueryRow",
    "ScenarioConfig",
    "SeedPack",
    "TestContext",
    "TestScenario",
    "ValidationResultAssertions",
    "assert_ingest_result",
    "assert_result",
    "assert_validation",
    "coverage_context",
    "create_test_context",
    "full_context",
    "graph_context",
    "minimal_context",
]
