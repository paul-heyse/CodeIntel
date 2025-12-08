"""Tests for disk-backed graph cache and instrumentation."""

from __future__ import annotations

import json
from pathlib import Path

import networkx as nx
from networkx.readwrite import json_graph

from codeintel.analytics.runtime import GraphRuntime, GraphRuntimeOptions
from tests._helpers.assertions.expectation_assertions import expect_equal
from tests._helpers.factories import make_snapshot
from tests._helpers.gateway import GatewayFactory
from tests._helpers.graphs import GraphFixtures, GraphStubEngine


class _CountingGraphEngine(GraphStubEngine):
    """Graph stub that tracks load invocations."""

    def __init__(self, snapshot_root: Path) -> None:
        snapshot = make_snapshot(repo_root=snapshot_root)
        gateway = GatewayFactory().with_snapshot(snapshot.repo, snapshot.commit).open()
        fixtures = GraphFixtures(
            call_graph=nx.DiGraph([("a", "b")]),
            import_graph=nx.DiGraph(),
            config_graph=nx.Graph(),
            symbol_module_graph=nx.Graph(),
            symbol_function_graph=nx.Graph(),
        )
        super().__init__(
            gateway=gateway,
            snapshot=snapshot,
            call_graph_obj=fixtures.call_graph,
            import_graph_obj=fixtures.import_graph,
            symbol_module_graph_obj=fixtures.symbol_module_graph,
            symbol_function_graph_obj=fixtures.symbol_function_graph,
            config_bipartite_obj=fixtures.config_graph,
            copy_graphs=False,
        )
        self.calls = 0

    def load_call_graph(self) -> nx.DiGraph:
        self.calls += 1
        return super().load_call_graph()


def test_disk_cache_round_trip(tmp_path: Path) -> None:
    """Graphs should be read from disk cache when metadata matches."""
    engine = _CountingGraphEngine(tmp_path)
    try:
        opts = GraphRuntimeOptions(snapshot=engine.snapshot, graph_cache_dir=tmp_path)
        runtime = GraphRuntime(options=opts, engine=engine)

        graph1 = runtime.ensure_call_graph()
        expect_equal(engine.calls, 1)
        expected_nodes = 2
        expect_equal(graph1.number_of_nodes(), expected_nodes)

        engine2 = _CountingGraphEngine(tmp_path)
        try:
            runtime2 = GraphRuntime(options=opts, engine=engine2)
            graph2 = runtime2.ensure_call_graph()
            expect_equal(engine2.calls, 0)
            expect_equal(graph2.number_of_edges(), 1)
        finally:
            engine2.gateway.close()
    finally:
        engine.gateway.close()


def test_disk_cache_mismatch_falls_back_to_loader(tmp_path: Path) -> None:
    """Cache metadata mismatch should trigger loader path."""
    engine = _CountingGraphEngine(tmp_path)
    opts = GraphRuntimeOptions(snapshot=engine.snapshot, graph_cache_dir=tmp_path)

    base = f"other__c__auto__False__{('CALL_GRAPH').lower()}"
    graph_path = tmp_path / f"{base}.json"
    meta_path = tmp_path / f"{base}.meta"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with graph_path.open("w", encoding="utf-8") as fh:
        json.dump(json_graph.node_link_data(nx.DiGraph()), fh)
    meta_path.write_text("\n".join(["other", "c", "auto", "false"]), encoding="utf-8")

    runtime = GraphRuntime(options=opts, engine=engine)
    try:
        runtime.ensure_call_graph()
        expect_equal(engine.calls, 1)
    finally:
        engine.gateway.close()
