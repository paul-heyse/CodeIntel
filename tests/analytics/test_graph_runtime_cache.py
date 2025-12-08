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
from tests._helpers.graphs import (
    CountingGraphEngineAdapter,
    GraphFixtures,
    GraphStubEngine,
)


def _make_counting_engine(snapshot_root: Path) -> CountingGraphEngineAdapter:
    """Build a counting engine seeded with simple graphs.

    Returns
    -------
    CountingGraphEngineAdapter
        Adapter that records load invocations for assertions.
    """
    snapshot = make_snapshot(repo_root=snapshot_root)
    gateway = GatewayFactory().with_snapshot(snapshot.repo, snapshot.commit).open()
    fixtures = GraphFixtures(
        call_graph=nx.DiGraph([("a", "b")]),
        import_graph=nx.DiGraph(),
        config_graph=nx.Graph(),
        symbol_module_graph=nx.Graph(),
        symbol_function_graph=nx.Graph(),
    )
    runtime = GraphStubEngine(
        gateway=gateway,
        snapshot=snapshot,
        call_graph=fixtures.call_graph,
        import_graph=fixtures.import_graph,
        config_graph=fixtures.config_graph,
        symbol_module_graph=fixtures.symbol_module_graph,
        symbol_function_graph=fixtures.symbol_function_graph,
        copy_graphs=False,
    )
    return CountingGraphEngineAdapter(runtime, gateway=gateway, snapshot=snapshot)


def test_disk_cache_round_trip(tmp_path: Path) -> None:
    """Graphs should be read from disk cache when metadata matches."""
    engine = _make_counting_engine(tmp_path)
    try:
        opts = GraphRuntimeOptions(snapshot=engine.snapshot, graph_cache_dir=tmp_path)
        runtime = GraphRuntime(options=opts, engine=engine)

        graph1 = runtime.ensure_call_graph()
        expect_equal(engine.method_counts.get("load_call_graph", 0), 1)
        expected_nodes = 2
        expect_equal(graph1.number_of_nodes(), expected_nodes)

        engine2 = _make_counting_engine(tmp_path)
        try:
            runtime2 = GraphRuntime(options=opts, engine=engine2)
            graph2 = runtime2.ensure_call_graph()
            expect_equal(engine2.method_counts.get("load_call_graph", 0), 0)
            expect_equal(graph2.number_of_edges(), 1)
        finally:
            engine2.gateway.close()
    finally:
        engine.gateway.close()


def test_disk_cache_mismatch_falls_back_to_loader(tmp_path: Path) -> None:
    """Cache metadata mismatch should trigger loader path."""
    engine = _make_counting_engine(tmp_path)
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
        expect_equal(engine.method_counts.get("load_call_graph", 0), 1)
    finally:
        engine.gateway.close()
