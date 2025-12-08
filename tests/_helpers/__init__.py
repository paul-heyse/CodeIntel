"""Shared test helpers package.

This package provides standardized test infrastructure including:

- `TestContext`: Unified test environment for hexagonal architecture
- `TestScenario`: Declarative scenario builder
- `GraphPluginBuilder`: Fluent builder for graph test plugins
- `plugin_registrar`: Context manager for scoped plugin registration
- Seed packs for composable test data
- Provisioning utilities for gateway setup
- Standard NetworkX graph fixtures
- Various assertion helpers and test utilities
"""

from __future__ import annotations

from tests._helpers.build import (
    ManifestParams,
    RecordingExecutor,
    RecordingPlugin,
    RecordingProviders,
    make_build_config,
    make_build_paths,
    make_snapshot,
    sample_build_plan,
    sample_manifest,
    sample_target_graph,
    write_build_config,
)
from tests._helpers.configs.provisioning_config import (
    CallgraphFixtureOptions,
    GatewayOptions,
    GraphMetricsGatewayOptions,
    ProvisionedGateway,
    ProvisioningConfig,
)
from tests._helpers.constants import DEFAULT_COMMIT, DEFAULT_REPO, DEFAULT_RUN_ID
from tests._helpers.context import (
    QueryRow,
    SeedPack,
    TestContext,
    create_test_context,
)
from tests._helpers.coverage import build_fake_coverage, seed_coverage_pack
from tests._helpers.env import build_test_gateway, create_test_env
from tests._helpers.fakes.contexts import (
    ExecutionContextBuilder,
    build_plugin_execution_context,
    build_target_execution_context,
)
from tests._helpers.fakes.graph_plugins import (
    GraphPluginBuilder,
    make_functional_plugin,
    plugin_registrar,
)
from tests._helpers.fakes.httpx_clients import RecordingAsyncClient
from tests._helpers.fakes.ingestion_context import (
    RecordingContext,
    RecordingGateway,
    RecordingResources,
    build_repo_tree,
    make_target_context,
)
from tests._helpers.fakes.networkx_graphs import (
    chain_graph,
    complete_digraph,
    cyclic_graph,
    diamond_graph,
    disconnected_graph,
    star_graph,
)
from tests._helpers.fakes.query_service import FakeQueryService, ModelLike
from tests._helpers.immutability import assert_all_frozen, assert_frozen
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
from tests._helpers.rows import function_meta, function_metrics_row, module_row
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
    DATA_MODELS_PACK,
    FUNCTION_TYPES_PACK,
    GRAPH_PACK,
    METRICS_PACK,
    SUBSYSTEM_ANALYTICS_PACK,
)

__all__ = [
    "CORE_PACK",
    "COVERAGE_PACK",
    "DATA_MODELS_PACK",
    "DEFAULT_COMMIT",
    "DEFAULT_REPO",
    "DEFAULT_RUN_ID",
    "FUNCTION_TYPES_PACK",
    "GRAPH_PACK",
    "METRICS_PACK",
    "SUBSYSTEM_ANALYTICS_PACK",
    "CallgraphFixtureOptions",
    "ExecutionContextBuilder",
    "FakeQueryService",
    "GatewayOptions",
    "GraphMetricsGatewayOptions",
    "GraphPluginBuilder",
    "ManifestParams",
    "ModelLike",
    "ProvisionedGateway",
    "ProvisioningConfig",
    "QueryRow",
    "RecordingAsyncClient",
    "RecordingContext",
    "RecordingExecutor",
    "RecordingGateway",
    "RecordingPlugin",
    "RecordingProviders",
    "RecordingResources",
    "ScenarioConfig",
    "SeedPack",
    "TestContext",
    "TestScenario",
    "assert_all_frozen",
    "assert_frozen",
    "build_callgraph_fixture_repo",
    "build_fake_coverage",
    "build_plugin_execution_context",
    "build_repo_tree",
    "build_target_execution_context",
    "build_test_gateway",
    "chain_graph",
    "complete_digraph",
    "coverage_context",
    "create_test_context",
    "create_test_env",
    "cyclic_graph",
    "diamond_graph",
    "disconnected_graph",
    "docs_views_ready_gateway",
    "full_context",
    "function_meta",
    "function_metrics_row",
    "graph_context",
    "graph_metrics_ready_gateway",
    "make_build_config",
    "make_build_paths",
    "make_functional_plugin",
    "make_snapshot",
    "make_target_context",
    "minimal_context",
    "module_row",
    "plugin_registrar",
    "provision_docs_export_ready",
    "provision_gateway_with_repo",
    "provision_graph_ready_repo",
    "provisioned_gateway",
    "sample_build_plan",
    "sample_manifest",
    "sample_target_graph",
    "seed_call_graph_scoping",
    "seed_coverage_pack",
    "seed_docs_export_invalid_profile",
    "seed_docs_export_minimal",
    "seed_function_graph_cycle",
    "seed_graph_validation_gaps",
    "seed_mcp_backend",
    "seed_module_graph_inputs",
    "seed_profile_data",
    "star_graph",
    "write_build_config",
]
