"""Typed fakes for ingestion and analytics tests.

This package provides fake implementations for testing, organized by domain:
- coverage: Fake coverage data and loaders
- tools: Fake tool runners and services
- storage: Fake storage adapters
- configs: Fake configuration primitives
- utilities: Shared utility functions
- graph_plugins: Graph plugin testing helpers
- networkx_graphs: Standard NetworkX graph fixtures
"""

from __future__ import annotations

from tests._helpers.fakes.configs import (
    DEFAULT_TEST_COMMIT,
    DEFAULT_TEST_REPO,
    DEFAULT_TEST_RUN_ID,
    FakePluginContext,
    TestPluginContext,
    create_test_build_paths,
    create_test_run_context,
    create_test_snapshot,
)
from tests._helpers.fakes.coverage import (
    CoverageLoader,
    FakeCoverage,
    FakeCoverageData,
)
from tests._helpers.fakes.execution_contexts import (
    TestExecutionContextBuilder,
    create_test_execution_context,
)
from tests._helpers.fakes.graph_contexts import (
    GraphExecutorTestEnv,
    GraphPlanningTestEnv,
    GraphTelemetryTestEnv,
    create_graph_executor_env,
    create_graph_gateway,
    create_graph_planning_env,
    create_graph_plugin_context,
    create_graph_snapshot,
    create_graph_telemetry_env,
)
from tests._helpers.fakes.graph_plugins import (
    GraphPluginBuilder,
    make_functional_plugin,
    plugin_registrar,
)
from tests._helpers.fakes.graph_runtimes import (
    MockGraphRuntime,
    create_mock_runtime_all_graphs,
    create_mock_runtime_with_call_graph,
    create_mock_runtime_with_import_graph,
)
from tests._helpers.fakes.logging import CAPTURE_HANDLER_LEVEL, CapturingHandler
from tests._helpers.fakes.networkx_graphs import (
    bipartite_graph,
    chain_graph,
    complete_digraph,
    cyclic_graph,
    diamond_graph,
    disconnected_graph,
    hub_and_spoke_graph,
    layered_graph,
    star_graph,
    tree_graph,
)
from tests._helpers.fakes.plugins import (
    GraphPluginPack,
    GraphPluginPackCounters,
    GraphPluginPackSettings,
    TestGraphPlugin,
    build_graph_plugin_pack,
)
from tests._helpers.fakes.serving import (
    ScopeRecordingQuery,
    ServingScopePack,
    build_serving_scope_pack,
)
from tests._helpers.fakes.storage import FakeIngestStorage
from tests._helpers.fakes.tools import (
    FakeScipResult,
    FakeToolRunner,
    FakeToolService,
    FakeToolServiceConfig,
    write_dummy_scip_files,
)
from tests._helpers.fakes.utilities import utcnow

__all__ = [
    "CAPTURE_HANDLER_LEVEL",
    "DEFAULT_TEST_COMMIT",
    "DEFAULT_TEST_REPO",
    "DEFAULT_TEST_RUN_ID",
    "CapturingHandler",
    "CoverageLoader",
    "FakeCoverage",
    "FakeCoverageData",
    "FakeIngestStorage",
    "FakePluginContext",  # Backward compatibility alias for TestPluginContext
    "FakeScipResult",
    "FakeToolRunner",
    "FakeToolService",
    "FakeToolServiceConfig",
    "GraphExecutorTestEnv",
    "GraphPlanningTestEnv",
    "GraphPluginBuilder",
    "GraphPluginPack",
    "GraphPluginPackCounters",
    "GraphPluginPackSettings",
    "GraphTelemetryTestEnv",
    "MockGraphRuntime",
    "ScopeRecordingQuery",
    "ServingScopePack",
    "TestExecutionContextBuilder",
    "TestGraphPlugin",
    "TestPluginContext",
    "bipartite_graph",
    "build_graph_plugin_pack",
    "build_serving_scope_pack",
    "chain_graph",
    "complete_digraph",
    "create_graph_executor_env",
    "create_graph_gateway",
    "create_graph_planning_env",
    "create_graph_plugin_context",
    "create_graph_snapshot",
    "create_graph_telemetry_env",
    "create_mock_runtime_all_graphs",
    "create_mock_runtime_with_call_graph",
    "create_mock_runtime_with_import_graph",
    "create_test_build_paths",
    "create_test_execution_context",
    "create_test_run_context",
    "create_test_snapshot",
    "cyclic_graph",
    "diamond_graph",
    "disconnected_graph",
    "hub_and_spoke_graph",
    "layered_graph",
    "make_functional_plugin",
    "plugin_registrar",
    "star_graph",
    "tree_graph",
    "utcnow",
    "write_dummy_scip_files",
]
