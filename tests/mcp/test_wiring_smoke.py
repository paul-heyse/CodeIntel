"""MCP wiring smoke test using the shared backend resource helper."""

from __future__ import annotations

import pytest

from codeintel.config.serving_models import ServingConfig
from codeintel.serving.mcp.server import create_mcp_server
from codeintel.serving.services.factory import BackendResource, build_backend_resource
from codeintel.storage.gateway import StorageGateway
from tests._helpers.builders import RepoMapRow, insert_repo_map


def test_mcp_wiring_smoke(fresh_gateway: StorageGateway) -> None:
    """Server registers tools via shared helper and close hook is invoked."""
    called = False
    closed = False
    resource: BackendResource | None = None
    insert_repo_map(
        fresh_gateway,
        [
            RepoMapRow(
                repo=fresh_gateway.config.repo or "demo/repo",
                commit=fresh_gateway.config.commit or "deadbeef",
                modules={},
                overlays={},
            )
        ],
    )

    def _register_tools(_server: object, backend_arg: object) -> None:
        nonlocal called
        called = True
        if resource is not None and backend_arg is not resource.service:
            pytest.fail("Registry received unexpected backend service")

    cfg = ServingConfig(
        mode="local_db",
        repo=fresh_gateway.config.repo or "demo/repo",
        commit=fresh_gateway.config.commit or "deadbeef",
        db_path=fresh_gateway.config.db_path,
    )
    backend_factory = build_backend_resource
    resource = backend_factory(cfg, gateway=fresh_gateway)

    def _close_wrapper() -> None:
        nonlocal closed
        closed = True
        resource.close()

    def _backend_factory(_cfg: ServingConfig) -> BackendResource:
        return BackendResource(
            backend=resource.backend,
            service=resource.service,
            close=_close_wrapper,
        )

    server, close = create_mcp_server(
        cfg,
        backend_factory=_backend_factory,
        gateway=fresh_gateway,
        register_tools_fn=_register_tools,
    )

    if not called:
        pytest.fail("register_tools was not invoked")

    close()
    if not closed:
        pytest.fail("Close hook was not executed")
    if server is None:
        pytest.fail("MCP server was not created")
