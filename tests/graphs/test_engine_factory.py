"""Factory helpers for graph engines."""

from __future__ import annotations

from typing import TYPE_CHECKING

import networkx as nx

from codeintel.build.graphs.engine.factory import EngineBuildOptions, build_graph_engine
from codeintel.config.models import GraphBackendConfig
from tests._helpers.assertions import expect_true

if TYPE_CHECKING:
    from codeintel.storage.gateway import StorageGateway


def test_build_graph_engine_uses_backend_flags(graph_gateway: StorageGateway) -> None:
    """Graph engine factory honors backend GPU preference."""
    env: dict[str, str] = {}
    engine = build_graph_engine(
        snapshot=("demo/repo", "deadbeef"),
        dataset_root_dir=graph_gateway.datasets.dataset_root_dir,
        options=EngineBuildOptions(
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
        snapshot=("demo/repo", "deadbeef"),
        dataset_root_dir=graph_gateway.datasets.dataset_root_dir,
        options=EngineBuildOptions(
            graph_backend=GraphBackendConfig(use_gpu=False, backend="cpu", strict=False),
            env=env,
        ),
    )
    expect_true(not engine.use_gpu, message="Engine should not request GPU when use_gpu is False")
    expect_true(
        "NX_CUGRAPH_AUTOCONFIG" not in env, message="CPU backend should not set GPU env flags"
    )


def test_build_graph_engine_rustworkx_disables_gpu(graph_gateway: StorageGateway) -> None:
    """Rustworkx selection should skip GPU enablement."""
    env: dict[str, str] = {}
    engine = build_graph_engine(
        snapshot=("demo/repo", "deadbeef"),
        dataset_root_dir=graph_gateway.datasets.dataset_root_dir,
        options=EngineBuildOptions(
            graph_backend=GraphBackendConfig(
                use_gpu=True,
                backend="nx-cugraph",
                strict=False,
                engine="rustworkx",
            ),
            env=env,
        ),
    )
    expect_true(not engine.use_gpu, message="Rustworkx engine should not request GPU")
    expect_true(
        "NX_CUGRAPH_AUTOCONFIG" not in env,
        message="Rustworkx engine should not set GPU env flags",
    )
    graph = engine.call_graph()
    expect_true(
        isinstance(graph, nx.DiGraph),
        message="Rustworkx compatibility shim should return a DiGraph",
    )
