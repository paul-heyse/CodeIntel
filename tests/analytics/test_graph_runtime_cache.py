"""Tests for disk-backed graph cache and instrumentation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import networkx as nx
from networkx.readwrite import json_graph

from codeintel.analytics.graph_runtime import GraphRuntime, GraphRuntimeOptions
from codeintel.config.primitives import SnapshotRef
from codeintel.graphs.engine import GraphEngine


def _expect(*, condition: bool, detail: str) -> None:
    if condition:
        return
    raise AssertionError(detail)


class _StubEngine:
    def __init__(self) -> None:
        self.calls = 0

    def load_call_graph(self) -> nx.DiGraph:
        self.calls += 1
        graph = nx.DiGraph()
        graph.add_edge("a", "b")
        return graph


def test_disk_cache_round_trip(tmp_path: Path) -> None:
    """Graphs should be read from disk cache when metadata matches."""
    snapshot = SnapshotRef(repo="r", commit="c", repo_root=tmp_path)
    opts = GraphRuntimeOptions(snapshot=snapshot, graph_cache_dir=tmp_path)
    engine = _StubEngine()
    runtime = GraphRuntime(options=opts, engine=cast("GraphEngine", engine))

    graph1 = runtime.ensure_call_graph()
    _expect(condition=engine.calls == 1, detail="loader should be invoked on first load")
    expected_nodes = 2
    _expect(
        condition=graph1.number_of_nodes() == expected_nodes,
        detail="graph should have expected node count",
    )

    engine2 = _StubEngine()
    runtime2 = GraphRuntime(options=opts, engine=cast("GraphEngine", engine2))
    graph2 = runtime2.ensure_call_graph()
    _expect(condition=engine2.calls == 0, detail="cache should be used on second load")
    _expect(condition=graph2.number_of_edges() == 1, detail="graph should retain edges from cache")


def test_disk_cache_mismatch_falls_back_to_loader(tmp_path: Path) -> None:
    """Cache metadata mismatch should trigger loader path."""
    snapshot = SnapshotRef(repo="r", commit="c", repo_root=tmp_path)
    opts = GraphRuntimeOptions(snapshot=snapshot, graph_cache_dir=tmp_path)

    base = f"other__c__auto__False__{('CALL_GRAPH').lower()}"
    graph_path = tmp_path / f"{base}.json"
    meta_path = tmp_path / f"{base}.meta"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with graph_path.open("w", encoding="utf-8") as fh:
        json.dump(json_graph.node_link_data(nx.DiGraph()), fh)
    meta_path.write_text("\n".join(["other", "c", "auto", "false"]), encoding="utf-8")

    engine = _StubEngine()
    runtime = GraphRuntime(options=opts, engine=cast("GraphEngine", engine))
    runtime.ensure_call_graph()
    _expect(condition=engine.calls == 1, detail="cache mismatch should trigger loader")
