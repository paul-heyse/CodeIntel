"""Factory helpers for graph engines."""

from __future__ import annotations

import networkx as nx
import pytest

from codeintel.config.models import GraphBackendConfig
from codeintel.graphs.engine.factory import EngineBuildOptions, build_graph_engine
from tests._helpers.gateway import GatewayFactory


def test_build_graph_engine_uses_backend_flags() -> None:
    """Graph engine factory honors backend GPU preference."""
    gateway = GatewayFactory().with_views().open()
    try:
        env: dict[str, str] = {}
        engine = build_graph_engine(
            gateway,
            ("demo/repo", "deadbeef"),
            EngineBuildOptions(
                graph_backend=GraphBackendConfig(use_gpu=True, backend="auto", strict=False),
                env=env,
            ),
        )
        if not engine.use_gpu:
            pytest.fail("Engine did not inherit GPU preference")
        graph: nx.DiGraph = engine.call_graph()
        if not isinstance(graph, nx.DiGraph):
            pytest.fail("Engine did not return a DiGraph for call_graph")
        if env.get("NX_CUGRAPH_AUTOCONFIG") != "True":
            pytest.fail("GPU backend env flag was not set by factory")
    finally:
        gateway.close()


def test_build_graph_engine_cpu_backend_leaves_env_clean() -> None:
    """CPU backend path should not set GPU env flags."""
    gateway = GatewayFactory().with_views().open()
    try:
        env: dict[str, str] = {}
        engine = build_graph_engine(
            gateway,
            ("demo/repo", "deadbeef"),
            EngineBuildOptions(
                graph_backend=GraphBackendConfig(use_gpu=False, backend="cpu", strict=False),
                env=env,
            ),
        )
        if engine.use_gpu:
            pytest.fail("Engine should not request GPU when use_gpu is False")
        if "NX_CUGRAPH_AUTOCONFIG" in env:
            pytest.fail("CPU backend should not set GPU env flags")
    finally:
        gateway.close()
