"""Factory helpers for graph engines."""

from __future__ import annotations

import networkx as nx

from codeintel.config.models import GraphBackendConfig
from codeintel.graphs.engine.factory import EngineBuildOptions, build_graph_engine
from codeintel.storage.gateway import StorageGateway
from tests._helpers.assertions import expect_true


def test_build_graph_engine_uses_backend_flags(graph_gateway: StorageGateway) -> None:
    """Graph engine factory honors backend GPU preference."""
    env: dict[str, str] = {}
    engine = build_graph_engine(
        graph_gateway,
        ("demo/repo", "deadbeef"),
        EngineBuildOptions(
            graph_backend=GraphBackendConfig(use_gpu=True, backend="auto", strict=False),
            env=env,
        ),
    )
    expect_true(engine.use_gpu, message="Engine did not inherit GPU preference")
    graph: nx.DiGraph = engine.call_graph()
    expect_true(
        isinstance(graph, nx.DiGraph), message="Engine did not return a DiGraph for call_graph"
    )
    expect_true(
        env.get("NX_CUGRAPH_AUTOCONFIG") == "True",
        message="GPU backend env flag was not set by factory",
    )


def test_build_graph_engine_cpu_backend_leaves_env_clean(graph_gateway: StorageGateway) -> None:
    """CPU backend path should not set GPU env flags."""
    env: dict[str, str] = {}
    engine = build_graph_engine(
        graph_gateway,
        ("demo/repo", "deadbeef"),
        EngineBuildOptions(
            graph_backend=GraphBackendConfig(use_gpu=False, backend="cpu", strict=False),
            env=env,
        ),
    )
    expect_true(not engine.use_gpu, message="Engine should not request GPU when use_gpu is False")
    expect_true(
        "NX_CUGRAPH_AUTOCONFIG" not in env, message="CPU backend should not set GPU env flags"
    )
