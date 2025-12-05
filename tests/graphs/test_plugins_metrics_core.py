"""Tests for core graph metrics plugins.

This module tests the core graph metrics plugins from
`codeintel.graphs.plugins.metrics.core`, including:

- Core metrics computation with graph resources
- Core metrics computation with engine fallback
- No engine available failure handling
- Function ext metrics computation
- Module ext metrics computation
- Plugin metadata and factory functions
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import networkx as nx

from codeintel.core.resources import ResourceRegistry
from codeintel.graphs.core.context import GraphPluginExecutionContext, PluginScratch
from codeintel.graphs.engine import GraphKind, NxGraphEngine
from codeintel.graphs.plugins.metrics import (
    core_graph_metrics_plugin,
    function_ext_metrics_plugin,
    get_core_graph_metrics_plugin,
    get_function_ext_metrics_plugin,
    get_module_ext_metrics_plugin,
    module_ext_metrics_plugin,
)
from codeintel.graphs.resources.graphs import GraphResource
from codeintel.graphs.resources.storage import StorageResource
from codeintel.storage.gateway import StorageGateway
from tests._helpers.configs import GraphEngineSeed
from tests._helpers.factories import make_snapshot
from tests._helpers.orchestration import build_seeded_graph_engine

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NODE_COUNT: Final = 5
MODULE_COUNT: Final = 4
CALLER_GOID_1001: Final = 1001
CALLEE_GOID_1002: Final = 1002
CALLEE_GOID_1003: Final = 1003
CALLEE_GOID_1004: Final = 1004
CALLEE_GOID_1005: Final = 1005


# ---------------------------------------------------------------------------
# Test Helpers
# ---------------------------------------------------------------------------


def _make_realistic_call_graph() -> nx.DiGraph:
    """Create a realistic call graph for testing.

    Build a graph representing function call relationships
    with some cycles and varied connectivity.

    Returns
    -------
    nx.DiGraph
        A directed graph representing function calls.
    """
    g = nx.DiGraph()

    # Add edges representing call relationships
    # main -> process -> validate -> helper
    # process -> util
    # helper -> util (creates shared dependency)
    # validate -> process (creates cycle)
    g.add_edges_from(
        [
            (CALLER_GOID_1001, CALLEE_GOID_1002),
            (CALLEE_GOID_1002, CALLEE_GOID_1003),
            (CALLEE_GOID_1003, CALLEE_GOID_1004),
            (CALLEE_GOID_1002, CALLEE_GOID_1005),
            (CALLEE_GOID_1004, CALLEE_GOID_1005),
            (CALLEE_GOID_1003, CALLEE_GOID_1002),  # Cycle
        ]
    )

    return g


def _make_realistic_import_graph() -> nx.DiGraph:
    """Create a realistic import graph for testing.

    Build a graph representing module import relationships.

    Returns
    -------
    nx.DiGraph
        A directed graph representing module imports.
    """
    g = nx.DiGraph()

    # Add edges representing import relationships
    # main -> core -> utils
    # main -> helpers
    # helpers -> utils
    g.add_edges_from(
        [
            ("mypackage.main", "mypackage.core"),
            ("mypackage.core", "mypackage.utils"),
            ("mypackage.main", "mypackage.helpers"),
            ("mypackage.helpers", "mypackage.utils"),
        ]
    )

    return g


def _make_execution_context(
    gateway: StorageGateway,
    tmp_path: Path,
    *,
    seed: GraphEngineSeed | None = None,
    engine: NxGraphEngine | None = None,
    use_resources: bool = True,
) -> GraphPluginExecutionContext:
    """Create an execution context for metrics tests.

    Parameters
    ----------
    gateway
        Storage gateway.
    tmp_path
        Temporary path.
    seed
        Optional seed configuration for graph engine construction.
    engine
        Graph engine to use.
    use_resources
        Whether to register resources.

    Returns
    -------
    GraphPluginExecutionContext
        Configured execution context.
    """
    effective_engine = engine
    if effective_engine is None and seed is not None:
        effective_engine = build_seeded_graph_engine(gateway, seed)

    snapshot = (
        effective_engine.snapshot
        if effective_engine is not None
        else make_snapshot(repo="test/metrics", commit="metrics123", repo_root=tmp_path)
    )
    scratch = PluginScratch()
    resources = ResourceRegistry()

    # Always register storage; optionally graph resource
    resources.register(StorageResource, StorageResource(gateway, tmp_path))
    if use_resources and effective_engine is not None:
        resources.register(GraphResource, GraphResource(effective_engine))

    return GraphPluginExecutionContext(
        snapshot=snapshot,
        resources=resources,
        gateway=gateway,
        scratch=scratch,
        plugin_name="metrics_test",
        run_id="metrics-run-001",
    )


def test_compute_core_graph_metrics_with_engine(
    fresh_gateway: StorageGateway, tmp_path: Path
) -> None:
    """Core metrics computed successfully with engine fallback."""
    call_graph = _make_realistic_call_graph()
    import_graph = _make_realistic_import_graph()
    seed = GraphEngineSeed(
        call_graph=call_graph,
        import_graph=import_graph,
        repo_root=tmp_path,
    )
    engine = build_seeded_graph_engine(fresh_gateway, seed)

    ctx = _make_execution_context(fresh_gateway, tmp_path, engine=engine)

    result = get_core_graph_metrics_plugin().execute(ctx)

    assert result.success
    assert result.row_counts is not None
    assert "analytics.graph_metrics_functions" in result.row_counts
    assert "analytics.graph_metrics_modules" in result.row_counts
    assert result.row_counts["analytics.graph_metrics_functions"] == NODE_COUNT
    assert result.row_counts["analytics.graph_metrics_modules"] == MODULE_COUNT


def test_compute_core_graph_metrics_no_engine_fails(
    fresh_gateway: StorageGateway, tmp_path: Path
) -> None:
    """Core metrics computation fails when no engine available."""
    ctx = _make_execution_context(fresh_gateway, tmp_path, engine=None)

    result = get_core_graph_metrics_plugin().execute(ctx)

    assert not result.success
    assert result.error == "No GraphResource registered in context"


def test_compute_core_graph_metrics_empty_graphs(
    fresh_gateway: StorageGateway, tmp_path: Path
) -> None:
    """Core metrics handles empty graphs gracefully."""
    empty_call = nx.DiGraph()
    empty_import = nx.DiGraph()
    seed = GraphEngineSeed(
        call_graph=empty_call,
        import_graph=empty_import,
        repo_root=tmp_path,
    )
    engine = build_seeded_graph_engine(fresh_gateway, seed)

    ctx = _make_execution_context(fresh_gateway, tmp_path, engine=engine)

    result = get_core_graph_metrics_plugin().execute(ctx)

    assert result.success
    assert result.row_counts is not None
    assert result.row_counts["analytics.graph_metrics_functions"] == 0
    assert result.row_counts["analytics.graph_metrics_modules"] == 0


def test_compute_function_ext_metrics_with_engine(
    fresh_gateway: StorageGateway, tmp_path: Path
) -> None:
    """Function ext metrics computed successfully."""
    call_graph = _make_realistic_call_graph()
    seed = GraphEngineSeed(call_graph=call_graph, repo_root=tmp_path)
    engine = build_seeded_graph_engine(fresh_gateway, seed)

    ctx = _make_execution_context(fresh_gateway, tmp_path, engine=engine)

    result = get_function_ext_metrics_plugin().execute(ctx)

    assert result.success
    assert result.row_counts is not None
    assert "analytics.graph_metrics_functions_ext" in result.row_counts
    assert result.row_counts["analytics.graph_metrics_functions_ext"] == NODE_COUNT


def test_compute_function_ext_metrics_no_engine_fails(
    fresh_gateway: StorageGateway, tmp_path: Path
) -> None:
    """Function ext metrics fails when no engine available."""
    ctx = _make_execution_context(fresh_gateway, tmp_path, engine=None)

    result = get_function_ext_metrics_plugin().execute(ctx)

    assert not result.success
    assert result.error == "No GraphResource registered in context"


def test_compute_module_ext_metrics_with_engine(
    fresh_gateway: StorageGateway, tmp_path: Path
) -> None:
    """Module ext metrics computed successfully."""
    import_graph = _make_realistic_import_graph()
    seed = GraphEngineSeed(import_graph=import_graph, repo_root=tmp_path)
    engine = build_seeded_graph_engine(fresh_gateway, seed)

    ctx = _make_execution_context(fresh_gateway, tmp_path, engine=engine)

    result = get_module_ext_metrics_plugin().execute(ctx)

    assert result.success
    assert result.row_counts is not None
    assert "analytics.graph_metrics_modules_ext" in result.row_counts
    assert result.row_counts["analytics.graph_metrics_modules_ext"] == MODULE_COUNT


def test_compute_module_ext_metrics_no_engine_fails(
    fresh_gateway: StorageGateway, tmp_path: Path
) -> None:
    """Module ext metrics fails when no engine available."""
    ctx = _make_execution_context(fresh_gateway, tmp_path, engine=None)

    result = get_module_ext_metrics_plugin().execute(ctx)

    assert not result.success
    assert result.error == "No GraphResource registered in context"


# ---------------------------------------------------------------------------
# Tests: Plugin instances and metadata
# ---------------------------------------------------------------------------


def test_core_graph_metrics_plugin_metadata() -> None:
    """Core graph metrics plugin has correct metadata."""
    plugin = core_graph_metrics_plugin

    assert plugin.metadata.name == "core_graph_metrics"
    assert plugin.metadata.kind == "metric"
    assert plugin.metadata.stage == "core"
    assert "callgraph_builder" in plugin.metadata.depends_on
    assert "import_graph_builder" in plugin.metadata.depends_on
    assert "core_metrics" in plugin.metadata.provides


def test_function_ext_metrics_plugin_metadata() -> None:
    """Function ext metrics plugin has correct metadata."""
    plugin = function_ext_metrics_plugin

    assert plugin.metadata.name == "graph_metrics_functions_ext"
    assert plugin.metadata.kind == "metric"
    assert plugin.metadata.stage == "core"
    assert "callgraph_builder" in plugin.metadata.depends_on


def test_module_ext_metrics_plugin_metadata() -> None:
    """Module ext metrics plugin has correct metadata."""
    plugin = module_ext_metrics_plugin

    assert plugin.metadata.name == "graph_metrics_modules_ext"
    assert plugin.metadata.kind == "metric"
    assert plugin.metadata.stage == "core"
    assert "import_graph_builder" in plugin.metadata.depends_on


# ---------------------------------------------------------------------------
# Tests: Plugin factory functions
# ---------------------------------------------------------------------------


def test_get_core_graph_metrics_plugin_returns_instance() -> None:
    """Factory returns the core graph metrics plugin instance."""
    plugin = get_core_graph_metrics_plugin()

    assert plugin is core_graph_metrics_plugin
    assert plugin.metadata.name == "core_graph_metrics"


def test_get_function_ext_metrics_plugin_returns_instance() -> None:
    """Factory returns the function ext metrics plugin instance."""
    plugin = get_function_ext_metrics_plugin()

    assert plugin is function_ext_metrics_plugin
    assert plugin.metadata.name == "graph_metrics_functions_ext"


def test_get_module_ext_metrics_plugin_returns_instance() -> None:
    """Factory returns the module ext metrics plugin instance."""
    plugin = get_module_ext_metrics_plugin()

    assert plugin is module_ext_metrics_plugin
    assert plugin.metadata.name == "graph_metrics_modules_ext"


# ---------------------------------------------------------------------------
# Tests: Plugin output tables
# ---------------------------------------------------------------------------


def test_core_graph_metrics_plugin_output_tables() -> None:
    """Core graph metrics plugin declares output tables."""
    plugin = core_graph_metrics_plugin

    assert "analytics.graph_metrics_functions" in plugin.metadata.produces_tables
    assert "analytics.graph_metrics_modules" in plugin.metadata.produces_tables


def test_function_ext_metrics_plugin_output_tables() -> None:
    """Function ext metrics plugin declares output table."""
    plugin = function_ext_metrics_plugin

    assert "analytics.graph_metrics_functions_ext" in plugin.metadata.produces_tables


def test_module_ext_metrics_plugin_output_tables() -> None:
    """Module ext metrics plugin declares output table."""
    plugin = module_ext_metrics_plugin

    assert "analytics.graph_metrics_modules_ext" in plugin.metadata.produces_tables


# ---------------------------------------------------------------------------
# Tests: Plugin required graphs
# ---------------------------------------------------------------------------


def test_core_graph_metrics_requires_graphs() -> None:
    """Core graph metrics plugin requires call and import graphs."""
    plugin = core_graph_metrics_plugin

    assert plugin.metadata.requires_graph_kinds is not None
    # Check that it requires at least call_graph and import_graph
    assert GraphKind.CALL_GRAPH in plugin.metadata.requires_graph_kinds
    assert GraphKind.IMPORT_GRAPH in plugin.metadata.requires_graph_kinds


def test_function_ext_metrics_requires_call_graph() -> None:
    """Function ext metrics plugin requires call graph."""
    plugin = function_ext_metrics_plugin

    assert plugin.metadata.requires_graph_kinds is not None
    assert GraphKind.CALL_GRAPH in plugin.metadata.requires_graph_kinds


def test_module_ext_metrics_requires_import_graph() -> None:
    """Module ext metrics plugin requires import graph."""
    plugin = module_ext_metrics_plugin

    assert plugin.metadata.requires_graph_kinds is not None
    assert GraphKind.IMPORT_GRAPH in plugin.metadata.requires_graph_kinds
