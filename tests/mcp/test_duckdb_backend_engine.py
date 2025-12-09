"""Ensure DuckDBBackend reuses injected graph engines."""

from __future__ import annotations

import pytest

from codeintel.config.models import GraphBackendConfig
from codeintel.graphs.engine.factory import EngineBuildOptions, build_graph_engine
from tests._helpers.gateway import BackendOptions, GatewayFactory, build_duckdb_backend


def test_duckdb_backend_uses_injected_engine() -> None:
    """Backend should reuse a provided graph engine instead of rebuilding."""
    gateway = GatewayFactory().with_views().open()
    try:
        engine = build_graph_engine(
            gateway,
            ("demo/repo", "deadbeef"),
            EngineBuildOptions(
                graph_backend=GraphBackendConfig(use_gpu=False, backend="cpu", strict=False)
            ),
        )
        backend = build_duckdb_backend(
            gateway,
            repo="demo/repo",
            commit="deadbeef",
            options=BackendOptions(query_engine=engine),
        )
        if backend.query is None:
            pytest.fail("DuckDBBackend did not initialize query service")
        if backend.query.graph_engine is not engine:
            pytest.fail("DuckDBBackend did not reuse the provided engine")
    finally:
        gateway.close()
