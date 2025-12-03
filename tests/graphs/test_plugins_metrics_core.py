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
from typing import TYPE_CHECKING, Final

import networkx as nx

from codeintel.config.primitives import SnapshotRef
from codeintel.graphs.core.context import GraphExecutionContext, GraphRuntimeScratch
from codeintel.graphs.engine import GraphKind, NxGraphEngine
from codeintel.graphs.plugins.metrics import (
    core_graph_metrics_plugin,
    function_ext_metrics_plugin,
    get_core_graph_metrics_plugin,
    get_function_ext_metrics_plugin,
    get_module_ext_metrics_plugin,
    module_ext_metrics_plugin,
)
from codeintel.graphs.resources.container import ResourceContainer
from codeintel.graphs.resources.graphs import GraphResource
from codeintel.graphs.resources.storage import StorageResource
from codeintel.storage.schemas import apply_all_schemas
from tests._helpers.gateway import open_ingestion_gateway_with_macros

if TYPE_CHECKING:
    from codeintel.storage.gateway import StorageGateway

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NODE_COUNT: Final = 5
MODULE_COUNT: Final = 4


# ---------------------------------------------------------------------------
# Test Helpers
# ---------------------------------------------------------------------------


def _make_gateway() -> StorageGateway:
    """Create a gateway for metrics tests.

    Returns
    -------
    StorageGateway
        Configured gateway.
    """
    gateway = open_ingestion_gateway_with_macros(
        apply_schema=True, ensure_views=True, validate_schema=True
    )
    apply_all_schemas(gateway.con)
    return gateway


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
            (1001, 1002),
            (1002, 1003),
            (1003, 1004),
            (1002, 1005),
            (1004, 1005),
            (1003, 1002),  # Cycle
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


class _MockGraphEngine:
    """Minimal graph engine mock for testing metrics computation.

    This provides the same interface as NxGraphEngine for the
    methods needed by metrics plugins and GraphResource.
    """

    def __init__(
        self,
        call_graph: nx.DiGraph | None = None,
        import_graph: nx.DiGraph | None = None,
        repo: str = "test/metrics",
        commit: str = "metrics123",
    ) -> None:
        """Initialize with graphs.

        Parameters
        ----------
        call_graph
            Call graph to return.
        import_graph
            Import graph to return.
        repo
            Repository identifier.
        commit
            Commit hash.
        """
        self._call_graph = call_graph or nx.DiGraph()
        self._import_graph = import_graph or nx.DiGraph()
        self._repo = repo
        self._commit = commit

    @property
    def repo(self) -> str:
        """Repository identifier."""
        return self._repo

    @property
    def commit(self) -> str:
        """Commit hash."""
        return self._commit

    def clear_cache(self) -> None:
        """Clear cached graphs (no-op for mock)."""

    def call_graph(self) -> nx.DiGraph:
        """Return the call graph.

        Returns
        -------
        nx.DiGraph
            The call graph.
        """
        return self._call_graph

    def import_graph(self) -> nx.DiGraph:
        """Return the import graph.

        Returns
        -------
        nx.DiGraph
            The import graph.
        """
        return self._import_graph


def _make_execution_context(
    gateway: StorageGateway,
    tmp_path: Path,
    *,
    engine: _MockGraphEngine | NxGraphEngine | None = None,
    use_resources: bool = True,
) -> GraphExecutionContext:
    """Create an execution context for metrics tests.

    Parameters
    ----------
    gateway
        Storage gateway.
    tmp_path
        Temporary path.
    engine
        Graph engine to use.
    use_resources
        Whether to register resources.

    Returns
    -------
    GraphExecutionContext
        Configured execution context.
    """
    snapshot = SnapshotRef(repo="test/metrics", commit="metrics123", repo_root=tmp_path)
    scratch = GraphRuntimeScratch()
    resources = ResourceContainer()

    # Always register storage; optionally graph resource
    resources.register(StorageResource(gateway, tmp_path))
    if use_resources and engine is not None:
        # Accept any engine-like object (including _MockGraphEngine for tests)
        resources.register(GraphResource(engine))  # type: ignore[arg-type]

    return GraphExecutionContext(
        snapshot=snapshot,
        resources=resources,
        _gateway=gateway,
        scratch=scratch,
        plugin_name="metrics_test",
        run_id="metrics-run-001",
    )


def test_compute_core_graph_metrics_with_engine(tmp_path: Path) -> None:
    """Core metrics computed successfully with engine fallback."""
    gateway = _make_gateway()
    try:
        call_graph = _make_realistic_call_graph()
        import_graph = _make_realistic_import_graph()
        engine = _MockGraphEngine(call_graph, import_graph)

        ctx = _make_execution_context(gateway, tmp_path, engine=engine)

        result = get_core_graph_metrics_plugin().execute(ctx)

        assert result.success
        assert result.row_counts is not None
        assert "analytics.graph_metrics_functions" in result.row_counts
        assert "analytics.graph_metrics_modules" in result.row_counts
        assert result.row_counts["analytics.graph_metrics_functions"] == NODE_COUNT
        assert result.row_counts["analytics.graph_metrics_modules"] == MODULE_COUNT
    finally:
        gateway.close()


def test_compute_core_graph_metrics_no_engine_fails(tmp_path: Path) -> None:
    """Core metrics computation fails when no engine available."""
    gateway = _make_gateway()
    try:
        ctx = _make_execution_context(gateway, tmp_path, engine=None)

        result = get_core_graph_metrics_plugin().execute(ctx)

        assert not result.success
        assert result.error == "No GraphResource registered in context"
    finally:
        gateway.close()


def test_compute_core_graph_metrics_empty_graphs(tmp_path: Path) -> None:
    """Core metrics handles empty graphs gracefully."""
    gateway = _make_gateway()
    try:
        empty_call = nx.DiGraph()
        empty_import = nx.DiGraph()
        engine = _MockGraphEngine(empty_call, empty_import)

        ctx = _make_execution_context(gateway, tmp_path, engine=engine)

        result = get_core_graph_metrics_plugin().execute(ctx)

        assert result.success
        assert result.row_counts is not None
        assert result.row_counts["analytics.graph_metrics_functions"] == 0
        assert result.row_counts["analytics.graph_metrics_modules"] == 0
    finally:
        gateway.close()


def test_compute_function_ext_metrics_with_engine(tmp_path: Path) -> None:
    """Function ext metrics computed successfully."""
    gateway = _make_gateway()
    try:
        call_graph = _make_realistic_call_graph()
        engine = _MockGraphEngine(call_graph=call_graph)

        ctx = _make_execution_context(gateway, tmp_path, engine=engine)

        result = get_function_ext_metrics_plugin().execute(ctx)

        assert result.success
        assert result.row_counts is not None
        assert "analytics.graph_metrics_functions_ext" in result.row_counts
        assert result.row_counts["analytics.graph_metrics_functions_ext"] == NODE_COUNT
    finally:
        gateway.close()


def test_compute_function_ext_metrics_no_engine_fails(tmp_path: Path) -> None:
    """Function ext metrics fails when no engine available."""
    gateway = _make_gateway()
    try:
        ctx = _make_execution_context(gateway, tmp_path, engine=None)

        result = get_function_ext_metrics_plugin().execute(ctx)

        assert not result.success
        assert result.error == "No GraphResource registered in context"
    finally:
        gateway.close()


def test_compute_module_ext_metrics_with_engine(tmp_path: Path) -> None:
    """Module ext metrics computed successfully."""
    gateway = _make_gateway()
    try:
        import_graph = _make_realistic_import_graph()
        engine = _MockGraphEngine(import_graph=import_graph)

        ctx = _make_execution_context(gateway, tmp_path, engine=engine)

        result = get_module_ext_metrics_plugin().execute(ctx)

        assert result.success
        assert result.row_counts is not None
        assert "analytics.graph_metrics_modules_ext" in result.row_counts
        assert result.row_counts["analytics.graph_metrics_modules_ext"] == MODULE_COUNT
    finally:
        gateway.close()


def test_compute_module_ext_metrics_no_engine_fails(tmp_path: Path) -> None:
    """Module ext metrics fails when no engine available."""
    gateway = _make_gateway()
    try:
        ctx = _make_execution_context(gateway, tmp_path, engine=None)

        result = get_module_ext_metrics_plugin().execute(ctx)

        assert not result.success
        assert result.error == "No GraphResource registered in context"
    finally:
        gateway.close()


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

    assert plugin.metadata.requires_graphs is not None
    # Check that it requires at least call_graph and import_graph
    assert GraphKind.CALL_GRAPH in plugin.metadata.requires_graphs
    assert GraphKind.IMPORT_GRAPH in plugin.metadata.requires_graphs


def test_function_ext_metrics_requires_call_graph() -> None:
    """Function ext metrics plugin requires call graph."""
    plugin = function_ext_metrics_plugin

    assert plugin.metadata.requires_graphs is not None
    assert GraphKind.CALL_GRAPH in plugin.metadata.requires_graphs


def test_module_ext_metrics_requires_import_graph() -> None:
    """Module ext metrics plugin requires import graph."""
    plugin = module_ext_metrics_plugin

    assert plugin.metadata.requires_graphs is not None
    assert GraphKind.IMPORT_GRAPH in plugin.metadata.requires_graphs
