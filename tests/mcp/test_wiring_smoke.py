"""MCP wiring smoke test using the shared backend resource helper."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import pytest

from codeintel.config.serving_models import ServingConfig
from codeintel.mcp.server import create_mcp_server
from codeintel.services.factory import BackendResource
from codeintel.services.query_service import QueryService

if TYPE_CHECKING:
    from codeintel.mcp.backend import QueryBackend


@dataclass
class _StubBackend:
    """Minimal backend placeholder exposing a service attribute."""

    service: object


def test_mcp_wiring_smoke() -> None:
    """Server registers tools via shared helper and close hook is invoked."""
    called = False
    closed = False
    service = cast("QueryService", object())
    backend = _StubBackend(service)

    def _fake_build_backend_resource(_cfg: ServingConfig) -> BackendResource:
        nonlocal closed

        def _close() -> None:
            nonlocal closed
            closed = True

        return BackendResource(
            backend=cast("QueryBackend", backend),
            service=service,
            close=_close,
        )

    def _fake_register_tools(_server: object, backend_arg: object) -> None:
        nonlocal called
        called = True
        if backend_arg is not service:
            pytest.fail("Registry received unexpected backend")

    cfg = ServingConfig(
        mode="remote_api",
        repo="r",
        commit="c",
        api_base_url="http://test",
    )
    server, close = create_mcp_server(
        cfg,
        backend_factory=_fake_build_backend_resource,
        register_tools_fn=_fake_register_tools,
    )

    if not called:
        pytest.fail("register_tools was not invoked")

    close()
    if not closed:
        pytest.fail("Close hook was not executed")
    if server is None:
        pytest.fail("MCP server was not created")
