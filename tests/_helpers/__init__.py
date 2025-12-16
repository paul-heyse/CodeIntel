"""Shared test helpers package.

This package provides standardized test infrastructure including:

Canonical Environment Types
---------------------------
- ``TestContext``: Unified test environment for hexagonal architecture
- ``ExecutionContextBuilder``: Fluent builder for plugin/target execution contexts

Context Creation (require tmp_path for isolation)
-------------------------------------------------
- ``create_test_env(tmp_path, ...)``: Create a TestContext with gateway and snapshot
- ``create_test_context(tmp_path, ...)``: Lower-level TestContext creation

Execution Context Building
--------------------------
- ``ExecutionContextBuilder``: Fluent builder for PluginExecutionContext and
  TargetExecutionContext
- ``build_plugin_execution_context(...)``: Convenience function for plugin contexts
- ``build_target_execution_context(...)``: Convenience function for target contexts

Scenario Building
-----------------
- ``TestScenario``: Declarative scenario builder with seed pack composition
- ``minimal_context``, ``graph_context``, etc.: Convenience factories

Recording/Test Doubles
----------------------
- ``RecordingGateway``: Wraps real gateway, records SQL (canonical implementation
  from ``fakes.contexts``)

Gateway Configuration
---------------------
- ``GatewayOptions``: Canonical gateway configuration dataclass
- ``GatewayFactory``: Fluent factory for creating test gateways
- ``provisioning_gateway_options()``: Factory for provisioning defaults

Additional Utilities
--------------------
- ``build_repo_tree``: Write test files to a temp directory
- Seed packs for composable test data (CORE_PACK, GRAPH_PACK, etc.)
- Standard NetworkX graph fixtures
- Provisioning utilities for gateway setup
- Various assertion helpers and test utilities
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any, cast

from tests._helpers.build import (
    ManifestParams,
    RecordingPlugin,
    RecordingProviders,
    make_build_config,
    make_build_paths,
    make_snapshot,
    sample_manifest,
    sample_target_graph,
    write_build_config,
)
from tests._helpers.catalogs import ensure_catalog_with_goids, seed_goids_from_catalog
from tests._helpers.cli_context import (
    CliTestContext,
    cli_test_context_with_seeds,
    create_cli_test_context,
    make_command_context,
    params,
)
from tests._helpers.configs.provisioning_config import (
    CallgraphFixtureOptions,
    GatewayOptions,
    GraphMetricsGatewayOptions,
    ProvisionedGateway,
    ProvisioningConfig,
    provisioning_gateway_options,
)
from tests._helpers.constants import DEFAULT_COMMIT, DEFAULT_REPO, DEFAULT_RUN_ID
from tests._helpers.context import (
    QueryRow,
    SeedPack,
    TestContext,
    coverage_and_graph_context,
    coverage_ready_context,
    create_test_context,
    graph_ready_context,
)
from tests._helpers.coverage import build_fake_coverage, seed_coverage_pack
from tests._helpers.env import (
    build_test_gateway,
    create_provisioned_test_env,
    create_test_env,
)
from tests._helpers.evidence import build_entrypoint_evidence
from tests._helpers.fakes.contexts import (
    ExecutionContextBuilder,
    RecordingGateway,
    build_plugin_execution_context,
    build_target_execution_context,
)
from tests._helpers.fakes.httpx_clients import RecordingAsyncClient
from tests._helpers.fakes.ingestion_context import build_repo_tree
from tests._helpers.fakes.networkx_graphs import (
    chain_graph,
    complete_digraph,
    cyclic_graph,
    diamond_graph,
    disconnected_graph,
    star_graph,
)
from tests._helpers.gateway import GatewayFactory, analytics_gateway
from tests._helpers.hamilton_execution import (
    HamiltonTestBuilder,
    HamiltonTestContext,
    execute_hamilton_target,
    execute_hamilton_target_async,
)
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
    CLI_CORE_PACK,
    CORE_PACK,
    COVERAGE_PACK,
    DATA_MODELS_PACK,
    FUNCTION_TYPES_PACK,
    GRAPH_HANDLER_PACK,
    GRAPH_PACK,
    METRICS_PACK,
    OPERATION_REGISTRY_PACK,
    STORAGE_PROFILE_PACK,
    SUBSYSTEM_ANALYTICS_PACK,
    SUBSYSTEM_HANDLER_PACK,
)

__all__ = [
    "CLI_CORE_PACK",
    "CORE_PACK",
    "COVERAGE_PACK",
    "DATA_MODELS_PACK",
    "DEFAULT_COMMIT",
    "DEFAULT_REPO",
    "DEFAULT_RUN_ID",
    "FUNCTION_TYPES_PACK",
    "GRAPH_HANDLER_PACK",
    "GRAPH_PACK",
    "METRICS_PACK",
    "OPERATION_REGISTRY_PACK",
    "STORAGE_PROFILE_PACK",
    "SUBSYSTEM_ANALYTICS_PACK",
    "SUBSYSTEM_HANDLER_PACK",
    "CallgraphFixtureOptions",
    "CliTestContext",
    "ExecutionContextBuilder",
    "GatewayFactory",
    "GatewayOptions",
    "GraphMetricsGatewayOptions",
    "HamiltonTestBuilder",
    "HamiltonTestContext",
    "ManifestParams",
    "ProvisionedGateway",
    "ProvisioningConfig",
    "QueryRow",
    "RecordingAsyncClient",
    "RecordingGateway",
    "RecordingPlugin",
    "RecordingProviders",
    "ScenarioConfig",
    "SeedPack",
    "TestContext",
    "TestScenario",
    "analytics_gateway",
    "assert_all_frozen",
    "assert_frozen",
    "build_callgraph_fixture_repo",
    "build_entrypoint_evidence",
    "build_fake_coverage",
    "build_plugin_execution_context",
    "build_repo_tree",
    "build_target_execution_context",
    "build_test_gateway",
    "chain_graph",
    "cli_test_context_with_seeds",
    "complete_digraph",
    "coverage_and_graph_context",
    "coverage_context",
    "coverage_ready_context",
    "create_cli_test_context",
    "create_provisioned_test_env",
    "create_test_context",
    "create_test_env",
    "cyclic_graph",
    "diamond_graph",
    "disconnected_graph",
    "docs_views_ready_gateway",
    "ensure_catalog_with_goids",
    "execute_hamilton_target",
    "execute_hamilton_target_async",
    "full_context",
    "function_meta",
    "function_metrics_row",
    "graph_context",
    "graph_metrics_ready_gateway",
    "graph_ready_context",
    "make_build_config",
    "make_build_paths",
    "make_command_context",
    "make_snapshot",
    "minimal_context",
    "module_row",
    "params",
    "provision_docs_export_ready",
    "provision_gateway_with_repo",
    "provision_graph_ready_repo",
    "provisioned_gateway",
    "provisioning_gateway_options",
    "sample_manifest",
    "sample_target_graph",
    "seed_call_graph_scoping",
    "seed_coverage_pack",
    "seed_docs_export_invalid_profile",
    "seed_docs_export_minimal",
    "seed_function_graph_cycle",
    "seed_goids_from_catalog",
    "seed_graph_validation_gaps",
    "seed_mcp_backend",
    "seed_module_graph_inputs",
    "seed_profile_data",
    "star_graph",
    "write_build_config",
]


if TYPE_CHECKING:
    from tests._helpers.context import TestContext
    from tests._helpers.plugin_harness import PluginHarnessFactory
else:
    PluginHarnessFactory = cast("Any", None)

_LAZY_HELPERS: dict[str, str] = {
    "PluginHarnessFactory": "tests._helpers.plugin_harness",
}


def __getattr__(name: str) -> object:
    if name in _LAZY_HELPERS:
        module_name = _LAZY_HELPERS[name]
        module = import_module(module_name)
        return getattr(module, name)
    msg = f"module {__name__} has no attribute {name}"
    raise AttributeError(msg)
