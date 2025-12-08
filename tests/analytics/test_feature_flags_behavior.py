"""Behavioral tests for graph feature flags (eager hydration, community cap)."""

from __future__ import annotations

from pathlib import Path

import networkx as nx

from codeintel.analytics.compute.graphs.structural import structural_metrics
from codeintel.analytics.runtime import GraphRuntimeOptions, build_graph_runtime
from codeintel.config.primitives import GraphFeatureFlags, SnapshotRef
from codeintel.graphs.engine import GraphKind
from codeintel.storage.gateway import StorageGateway
from tests._helpers.assertions import expect_true
from tests._helpers.factories import make_snapshot
from tests._helpers.graphs import CountingGraphEngineAdapter, GraphStubEngine


def _make_counting_engine(
    gateway: StorageGateway, snapshot: SnapshotRef
) -> CountingGraphEngineAdapter:
    runtime = GraphStubEngine(
        gateway=gateway,
        snapshot=snapshot,
        call_graph=nx.DiGraph([(1, 2)]),
        import_graph=nx.DiGraph([("a", "b")]),
    )
    return CountingGraphEngineAdapter(runtime, gateway=gateway, snapshot=snapshot)


def test_eager_hydration_respects_feature_override(
    tmp_path: Path, fresh_gateway: StorageGateway
) -> None:
    """Eager hydration should preload graphs when the feature flag is enabled."""
    snapshot = make_snapshot(repo_root=tmp_path)
    stub = _make_counting_engine(fresh_gateway, snapshot)

    opts = GraphRuntimeOptions(
        snapshot=snapshot,
        graphs=GraphKind.CALL_GRAPH | GraphKind.IMPORT_GRAPH,
        eager=False,
        engine=stub,
        features=GraphFeatureFlags(eager_hydration=True),
    )
    build_graph_runtime(fresh_gateway, opts)

    expect_true(
        stub.method_counts.get("load_call_graph", 0) > 0
        and stub.method_counts.get("load_import_graph", 0) > 0,
        message="Eager hydration should load call and import graphs when enabled",
    )


def test_eager_hydration_off_defers_graph_loads(
    tmp_path: Path, fresh_gateway: StorageGateway
) -> None:
    """Absent eager flag should defer graph loads until explicitly requested."""
    snapshot = make_snapshot(repo_root=tmp_path)
    stub = _make_counting_engine(fresh_gateway, snapshot)

    opts = GraphRuntimeOptions(
        snapshot=snapshot,
        graphs=GraphKind.CALL_GRAPH | GraphKind.IMPORT_GRAPH,
        eager=False,
        engine=stub,
        features=GraphFeatureFlags(),
    )
    build_graph_runtime(fresh_gateway, opts)

    expect_true(
        stub.method_counts.get("load_call_graph", 0) == 0
        and stub.method_counts.get("load_import_graph", 0) == 0,
        message="Graphs should not be preloaded when eager hydration is disabled",
    )


def test_community_detection_cap_skips_when_exceeded() -> None:
    """Community detection should be skipped when graph exceeds the configured cap."""
    graph = nx.complete_graph(5)
    metrics = structural_metrics(graph, community_limit=3)
    expect_true(
        metrics.community_id == {},
        message="Community ids should be empty when exceeding the cap",
    )

    small_graph = nx.path_graph(3)
    small_metrics = structural_metrics(small_graph, community_limit=10)
    expect_true(
        bool(small_metrics.community_id),
        message="Community ids should be computed when under the cap",
    )
