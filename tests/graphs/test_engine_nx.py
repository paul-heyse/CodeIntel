"""NetworkX engine behavior mirrors direct nx_views loaders."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime

import networkx as nx
import pytest

from codeintel.graphs import nx_views
from codeintel.graphs.engine import NxGraphEngine
from tests._helpers.builders import (
    CallGraphEdgeRow,
    CallGraphNodeRow,
    ConfigValueRow,
    ImportGraphEdgeRow,
    ModuleRow,
    SymbolUseEdgeRow,
    TestCoverageEdgeRow,
    insert_call_graph_edges,
    insert_call_graph_nodes,
    insert_config_values,
    insert_import_graph_edges,
    insert_modules,
    insert_symbol_use_edges,
    insert_test_coverage_edges,
)
from tests._helpers.gateway import open_ingestion_gateway


def _node_payload(graph: nx.Graph) -> set[tuple[object, tuple[tuple[str, object], ...]]]:
    return {(node, tuple(sorted(data.items()))) for node, data in graph.nodes(data=True)}


def _edge_payload(graph: nx.Graph) -> set[tuple[object, object, object]]:
    return {(src, dst, data.get("weight", 1)) for src, dst, data in graph.edges(data=True)}


def _assert_graph_match(name: str, expected: nx.Graph, actual: nx.Graph) -> None:
    if _node_payload(expected) != _node_payload(actual):
        pytest.fail(f"{name} nodes differ between engine and nx_views")
    if _edge_payload(expected) != _edge_payload(actual):
        pytest.fail(f"{name} edges differ between engine and nx_views")


def test_engine_matches_nx_views_for_core_graphs() -> None:
    """NxGraphEngine should produce the same graphs as direct nx_views loaders."""
    gateway = open_ingestion_gateway(apply_schema=True, ensure_views=True, validate_schema=True)
    try:
        repo = "demo/repo"
        commit = "deadbeef"
        insert_call_graph_nodes(
            gateway,
            [
                CallGraphNodeRow(1, "python", "function", 0, is_public=True, rel_path="pkg/a.py"),
                CallGraphNodeRow(2, "python", "function", 0, is_public=True, rel_path="pkg/b.py"),
            ],
        )
        insert_call_graph_edges(
            gateway,
            [
                CallGraphEdgeRow(
                    repo=repo,
                    commit=commit,
                    caller_goid_h128=1,
                    callee_goid_h128=2,
                    callsite_path="pkg/a.py",
                    callsite_line=1,
                    callsite_col=0,
                    language="python",
                    kind="direct",
                    resolved_via="local_name",
                    confidence=1.0,
                )
            ],
        )
        insert_import_graph_edges(
            gateway,
            [
                ImportGraphEdgeRow(
                    repo=repo,
                    commit=commit,
                    src_module="pkg.a",
                    dst_module="pkg.b",
                    src_fan_out=1,
                    dst_fan_in=1,
                    cycle_group=0,
                )
            ],
        )
        insert_modules(
            gateway,
            [
                ModuleRow(module="pkg.a", path="pkg/a.py", repo=repo, commit=commit),
                ModuleRow(module="pkg.b", path="pkg/b.py", repo=repo, commit=commit),
            ],
        )
        insert_symbol_use_edges(
            gateway,
            [
                SymbolUseEdgeRow(
                    symbol="foo",
                    def_path="pkg/a.py",
                    use_path="pkg/b.py",
                    same_file=False,
                    same_module=False,
                    def_goid_h128=1,
                    use_goid_h128=2,
                )
            ],
        )
        insert_config_values(
            gateway,
            [
                ConfigValueRow(
                    repo=repo,
                    commit=commit,
                    config_path="config.yml",
                    format="yaml",
                    key="service.name",
                    reference_paths=["pkg/a.py"],
                    reference_modules=["pkg.a", "pkg.b"],
                    reference_count=1,
                )
            ],
        )
        insert_test_coverage_edges(
            gateway,
            [
                TestCoverageEdgeRow(
                    test_id="test_example",
                    function_goid_h128=1,
                    urn="urn:test:example",
                    repo=repo,
                    commit=commit,
                    rel_path="tests/test_example.py",
                    qualname="test_example",
                    covered_lines=1,
                    executable_lines=1,
                    coverage_ratio=1.0,
                    last_status="passed",
                    created_at=datetime.now(tz=UTC),
                )
            ],
        )
        engine = NxGraphEngine(gateway=gateway, repo=repo, commit=commit)
        comparisons: list[tuple[str, Callable[[], nx.Graph], Callable[[], nx.Graph]]] = [
            (
                "call_graph",
                lambda: nx_views.load_call_graph(gateway, repo, commit),
                engine.call_graph,
            ),
            (
                "import_graph",
                lambda: nx_views.load_import_graph(gateway, repo, commit),
                engine.import_graph,
            ),
            (
                "symbol_module_graph",
                lambda: nx_views.load_symbol_module_graph(gateway, repo, commit),
                engine.symbol_module_graph,
            ),
            (
                "symbol_function_graph",
                lambda: nx_views.load_symbol_function_graph(gateway, repo, commit),
                engine.symbol_function_graph,
            ),
            (
                "config_module_bipartite",
                lambda: nx_views.load_config_module_bipartite(gateway, repo, commit),
                engine.config_module_bipartite,
            ),
            (
                "test_function_bipartite",
                lambda: nx_views.load_test_function_bipartite(gateway, repo, commit),
                engine.test_function_bipartite,
            ),
        ]
        for name, direct_loader, engine_loader in comparisons:
            expected = direct_loader()
            actual = engine_loader()
            _assert_graph_match(name, expected, actual)
            if actual is not engine_loader():
                pytest.fail(f"{name} was not cached on subsequent engine calls")
    finally:
        gateway.close()
