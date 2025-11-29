"""Lightweight perf smoke test for runtime ensure paths."""

from __future__ import annotations

import time
from pathlib import Path
from typing import cast

import networkx as nx

from codeintel.analytics.graph_metrics.metrics import structural_metrics
from codeintel.analytics.graph_runtime import GraphRuntime, GraphRuntimeOptions
from codeintel.analytics.graphs.graph_metrics import GraphMetricFilters
from codeintel.config.primitives import SnapshotRef
from codeintel.graphs.engine import GraphEngine

FAST_BUDGET_SECONDS = 0.2
MEDIUM_BUDGET_SECONDS = 0.5
STRUCTURAL_BUDGET_SECONDS = 0.3
SUBGRAPH_BUDGET_SECONDS = 0.05
FILTER_BUDGET_SECONDS = 0.1
FILTER_ALLOWLIST_SIZE = 50


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
        for i in range(50):
            graph.add_edge(i, i + 1)
        return graph


def test_ensure_call_graph_is_fast(tmp_path: Path) -> None:
    """Ensure ensure_call_graph stays within a modest time budget."""
    snapshot = SnapshotRef(repo="demo/repo", commit="deadbeef", repo_root=tmp_path)
    runtime = GraphRuntime(
        options=GraphRuntimeOptions(snapshot=snapshot),
        engine=cast("GraphEngine", _StubEngine()),
    )
    start = time.perf_counter()
    for _ in range(3):
        runtime.ensure_call_graph()
    elapsed = time.perf_counter() - start
    _expect(
        condition=elapsed < FAST_BUDGET_SECONDS,
        detail="ensure_call_graph should stay within budget",
    )


def test_ensure_call_graph_medium_perf(tmp_path: Path) -> None:
    """Guard against regressions on medium graphs."""
    snapshot = SnapshotRef(repo="demo/repo", commit="deadbeef", repo_root=tmp_path)

    class _MediumEngine(_StubEngine):
        def load_call_graph(self) -> nx.DiGraph:
            graph = nx.DiGraph()
            size = 500
            for i in range(size):
                graph.add_edge(i, (i + 1) % size)
            self.calls += 1
            return graph

    runtime = GraphRuntime(
        options=GraphRuntimeOptions(snapshot=snapshot),
        engine=cast("GraphEngine", _MediumEngine()),
    )
    start = time.perf_counter()
    runtime.ensure_call_graph()
    runtime.ensure_call_graph()  # cached path
    elapsed = time.perf_counter() - start
    _expect(
        condition=elapsed < MEDIUM_BUDGET_SECONDS,
        detail="medium graph ensure should stay within budget",
    )


def test_structural_metrics_with_cap_is_fast() -> None:
    """Community cap should skip heavy detection on large graphs within a budget."""
    graph = nx.complete_graph(50)
    start = time.perf_counter()
    metrics = structural_metrics(graph, community_limit=10)
    elapsed = time.perf_counter() - start
    _expect(
        condition=elapsed < STRUCTURAL_BUDGET_SECONDS,
        detail="structural metrics should respect time budget",
    )
    _expect(
        condition=metrics.community_id == {}, detail="community ids should be empty when capped"
    )


def test_filter_subgraph_on_large_graph_is_fast() -> None:
    """Filtering a large graph with an allowlist should stay fast."""
    graph = nx.DiGraph()
    for i in range(2000):
        graph.add_edge(i, (i + 1) % 2000)
    allowlist = set(range(10))
    start = time.perf_counter()
    filtered = nx.subgraph(graph, allowlist).copy()
    elapsed = time.perf_counter() - start
    _expect(
        condition=elapsed < SUBGRAPH_BUDGET_SECONDS,
        detail="subgraph filter should be fast for allowlists",
    )
    _expect(
        condition=set(filtered.nodes) == allowlist, detail="filtered nodes should match allowlist"
    )


def test_graph_metric_filters_on_runtime_are_fast(tmp_path: Path) -> None:
    """GraphMetricFilters should prune large call graphs efficiently."""
    snapshot = SnapshotRef(repo="demo/repo", commit="deadbeef", repo_root=tmp_path)

    class _LargeEngine(_StubEngine):
        def load_call_graph(self) -> nx.DiGraph:
            graph = nx.DiGraph()
            for i in range(4000):
                graph.add_edge(i, (i + 1) % 4000)
            self.calls += 1
            return graph

    runtime = GraphRuntime(
        options=GraphRuntimeOptions(snapshot=snapshot),
        engine=cast("GraphEngine", _LargeEngine()),
    )
    filters = GraphMetricFilters(function_goids=set(range(FILTER_ALLOWLIST_SIZE)))
    start = time.perf_counter()
    filtered = filters.filter_call_graph(runtime.ensure_call_graph())
    elapsed = time.perf_counter() - start
    _expect(
        condition=elapsed < FILTER_BUDGET_SECONDS,
        detail="filter_call_graph should be fast on large graphs",
    )
    _expect(
        condition=len(filtered) == FILTER_ALLOWLIST_SIZE,
        detail="filtered graph should match allowlist size",
    )
