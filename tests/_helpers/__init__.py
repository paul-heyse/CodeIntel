"""Shared test helpers package.

This package provides standardized test infrastructure for both analytics
and ingestion plugins, including:

- `PluginTestHarness`: Fluent test harness for analytics plugins
- `IngestPluginTestHarness`: Fluent test harness for ingestion plugins
- `TestContext`: Unified test environment for hexagonal architecture
- `TestScenario`: Declarative scenario builder
- Seed packs for composable test data
- Provisioning utilities for gateway setup
- Various assertion helpers and test utilities
"""

from __future__ import annotations

from tests._helpers.configs.provisioning_config import (
    CallgraphFixtureOptions,
    GatewayOptions,
    GraphMetricsGatewayOptions,
    ProvisionedGateway,
    ProvisioningConfig,
)
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
from tests._helpers.orchestration.provisioning import (
    build_callgraph_fixture_repo,
    docs_views_ready_gateway,
    graph_metrics_ready_gateway,
    provision_docs_export_ready,
    provision_gateway_with_repo,
    provision_graph_ready_repo,
    provisioned_gateway,
)
from tests._helpers.orchestration.seeding import (
    seed_call_graph_scoping,
    seed_docs_export_invalid_profile,
    seed_function_graph_cycle,
    seed_graph_validation_gaps,
    seed_module_graph_inputs,
)
from tests._helpers.orchestration.seeding_docs import (
    seed_docs_export_minimal,
    seed_mcp_backend,
    seed_profile_data,
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
    "CallgraphFixtureOptions",
    "GatewayOptions",
    "GraphMetricsGatewayOptions",
    "IngestPluginResultAssertions",
    "IngestPluginTestHarness",
    "PluginResultAssertions",
    "PluginTestHarness",
    "ProvisionedGateway",
    "ProvisioningConfig",
    "QueryRow",
    "ScenarioConfig",
    "SeedPack",
    "TestContext",
    "TestScenario",
    "ValidationResultAssertions",
    "assert_ingest_result",
    "assert_result",
    "assert_validation",
    "build_callgraph_fixture_repo",
    "coverage_context",
    "create_test_context",
    "docs_views_ready_gateway",
    "full_context",
    "graph_context",
    "graph_metrics_ready_gateway",
    "minimal_context",
    "provision_docs_export_ready",
    "provision_gateway_with_repo",
    "provision_graph_ready_repo",
    "provisioned_gateway",
    "seed_call_graph_scoping",
    "seed_docs_export_invalid_profile",
    "seed_docs_export_minimal",
    "seed_function_graph_cycle",
    "seed_graph_validation_gaps",
    "seed_mcp_backend",
    "seed_module_graph_inputs",
    "seed_profile_data",
]
