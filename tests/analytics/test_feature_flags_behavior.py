"""Behavioral tests for graph feature flags (eager hydration, community cap)."""

from __future__ import annotations

from pathlib import Path

import networkx as nx

from codeintel.analytics.graph_metrics.metrics import structural_metrics
from codeintel.analytics.graph_runtime import GraphRuntimeOptions, build_graph_runtime
from codeintel.config.primitives import GraphFeatureFlags, SnapshotRef
from codeintel.graphs.engine import GraphEngine, GraphKind
from codeintel.storage.gateway import StorageGateway


def _expect(*, condition: bool, detail: str) -> None:
    if condition:
        return
    raise AssertionError(detail)


class _StubEngine(GraphEngine):
    """Stub GraphEngine counting load calls."""

    def __init__(self, gateway: StorageGateway, snapshot: SnapshotRef) -> None:
        self.gateway: StorageGateway = gateway
        self._snapshot: SnapshotRef = snapshot
        self.call_loads = 0
        self.import_loads = 0
        self._empty_graph = nx.Graph()

    @property
    def use_gpu(self) -> bool:
        return False

    def load_call_graph(self) -> nx.DiGraph:
        self.call_loads += 1
        return nx.DiGraph([(1, 2)])

    def load_import_graph(self) -> nx.DiGraph:
        self.import_loads += 1
        return nx.DiGraph([("a", "b")])

    def call_graph(self) -> nx.DiGraph:  # pragma: no cover - unused
        return self.load_call_graph()

    def import_graph(self) -> nx.DiGraph:  # pragma: no cover - unused
        return self.load_import_graph()

    def load_symbol_module_graph(self) -> nx.Graph:  # pragma: no cover - unused
        return self._empty_graph

    def symbol_module_graph(self) -> nx.Graph:  # pragma: no cover - unused
        return self._empty_graph

    def load_symbol_function_graph(self) -> nx.Graph:  # pragma: no cover - unused
        return self._empty_graph

    def symbol_function_graph(self) -> nx.Graph:  # pragma: no cover - unused
        return self._empty_graph

    def load_config_module_bipartite(self) -> nx.Graph:  # pragma: no cover - unused
        return self._empty_graph

    def config_module_bipartite(self) -> nx.Graph:  # pragma: no cover - unused
        return self._empty_graph

    def load_test_function_bipartite(self) -> nx.Graph:  # pragma: no cover - unused
        return self._empty_graph

    def test_function_bipartite(self) -> nx.Graph:  # pragma: no cover - unused
        return self._empty_graph

    @property
    def snapshot(self) -> SnapshotRef:
        return self._snapshot


def test_eager_hydration_respects_feature_override(
    tmp_path: Path, fresh_gateway: StorageGateway
) -> None:
    """Eager hydration should preload graphs when the feature flag is enabled."""
    snapshot = SnapshotRef(repo="demo/repo", commit="deadbeef", repo_root=tmp_path)
    stub = _StubEngine(fresh_gateway, snapshot)

    opts = GraphRuntimeOptions(
        snapshot=snapshot,
        graphs=GraphKind.CALL_GRAPH | GraphKind.IMPORT_GRAPH,
        eager=False,
        engine=stub,
        features=GraphFeatureFlags(eager_hydration=True),
    )
    build_graph_runtime(fresh_gateway, opts)

    _expect(
        condition=stub.call_loads > 0 and stub.import_loads > 0,
        detail="Eager hydration should load call and import graphs when enabled",
    )


def test_eager_hydration_off_defers_graph_loads(
    tmp_path: Path, fresh_gateway: StorageGateway
) -> None:
    """Absent eager flag should defer graph loads until explicitly requested."""
    snapshot = SnapshotRef(repo="demo/repo", commit="deadbeef", repo_root=tmp_path)
    stub = _StubEngine(fresh_gateway, snapshot)

    opts = GraphRuntimeOptions(
        snapshot=snapshot,
        graphs=GraphKind.CALL_GRAPH | GraphKind.IMPORT_GRAPH,
        eager=False,
        engine=stub,
        features=GraphFeatureFlags(),
    )
    build_graph_runtime(fresh_gateway, opts)

    _expect(
        condition=stub.call_loads == 0 and stub.import_loads == 0,
        detail="Graphs should not be preloaded when eager hydration is disabled",
    )


def test_community_detection_cap_skips_when_exceeded() -> None:
    """Community detection should be skipped when graph exceeds the configured cap."""
    graph = nx.complete_graph(5)
    metrics = structural_metrics(graph, community_limit=3)
    _expect(
        condition=metrics.community_id == {},
        detail="Community ids should be empty when exceeding the cap",
    )

    small_graph = nx.path_graph(3)
    small_metrics = structural_metrics(small_graph, community_limit=10)
    _expect(
        condition=bool(small_metrics.community_id),
        detail="Community ids should be computed when under the cap",
    )
