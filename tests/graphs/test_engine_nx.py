"""NetworkX engine behavior mirrors direct nx_views loaders."""

from __future__ import annotations

import networkx as nx
import pytest

from codeintel.graphs import nx_views
from codeintel.graphs.engine import NxGraphEngine
from tests._helpers.builders import (
    CallGraphEdgeRow,
    CallGraphNodeRow,
    ImportGraphEdgeRow,
    insert_call_graph_edges,
    insert_call_graph_nodes,
    insert_import_graph_edges,
)
from tests._helpers.gateway import open_ingestion_gateway


def _node_payload(graph: nx.Graph) -> set[tuple[object, tuple[tuple[str, object], ...]]]:
    return {(node, tuple(sorted(data.items()))) for node, data in graph.nodes(data=True)}


def _edge_payload(graph: nx.Graph) -> set[tuple[object, object, object]]:
    return {(src, dst, data.get("weight", 1)) for src, dst, data in graph.edges(data=True)}


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
        engine = NxGraphEngine(gateway=gateway, repo=repo, commit=commit)

        direct_call = nx_views.load_call_graph(gateway, repo, commit)
        via_engine_call = engine.call_graph()
        if _node_payload(direct_call) != _node_payload(via_engine_call):
            pytest.fail("Call graph nodes differ between engine and nx_views")
        if _edge_payload(direct_call) != _edge_payload(via_engine_call):
            pytest.fail("Call graph edges differ between engine and nx_views")
        if via_engine_call is not engine.call_graph():
            pytest.fail("Call graph was not cached on subsequent engine calls")

        direct_import = nx_views.load_import_graph(gateway, repo, commit)
        via_engine_import = engine.import_graph()
        if _node_payload(direct_import) != _node_payload(via_engine_import):
            pytest.fail("Import graph nodes differ between engine and nx_views")
        if _edge_payload(direct_import) != _edge_payload(via_engine_import):
            pytest.fail("Import graph edges differ between engine and nx_views")
        if via_engine_import is not engine.import_graph():
            pytest.fail("Import graph was not cached on subsequent engine calls")
    finally:
        gateway.close()
