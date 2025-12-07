"""Lightweight tests for MCP server creation."""

from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import pytest

from codeintel.config.serving_models import ServingConfig
from codeintel.serving.mcp import server

if TYPE_CHECKING:
    from codeintel.storage.gateway import StorageGateway


def test_create_mcp_server_requires_gateway_for_local_db() -> None:
    """Local DB mode should enforce gateway requirement."""
    cfg = ServingConfig(mode="local_db")
    with pytest.raises(ValueError, match="StorageGateway is required"):
        server.create_mcp_server(cfg=cfg, gateway=None)


def test_create_mcp_server_uses_register_tools_fn() -> None:
    """Server creation should delegate registration using service when present."""
    cfg = ServingConfig(mode="remote_api", api_base_url="https://example.invalid")
    close_called = False
    register_calls: list[tuple[Any, Any]] = []

    class _Backend:
        def __init__(self) -> None:
            self.service = SimpleNamespace(name="service")

    backend = _Backend()

    def _mark_close() -> None:
        nonlocal close_called
        close_called = True

    def _backend_factory(*args: tuple[object, ...], **kwargs: object) -> SimpleNamespace:
        _ = args
        _ = kwargs
        return SimpleNamespace(backend=backend, close=_mark_close)

    def _register_tools_fn(mcp: object, svc: object) -> None:
        register_calls.append((mcp, svc))

    srv, close = server.create_mcp_server(
        cfg=cfg,
        backend_factory=cast("server.BackendFactory", _backend_factory),
        gateway=cast("StorageGateway", SimpleNamespace()),
        register_tools_fn=_register_tools_fn,
    )

    assert register_calls
    assert register_calls[0][1] is backend.service
    close()
    assert close_called
    # FastMCP should be returned
    assert hasattr(srv, "run")


def test_main_uses_register_tools_and_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Main flow should wire backend->service and invoke run without real network."""
    run_called = False
    register_calls: list[tuple[object, ServingConfig | None]] = []

    class _FastMcpStub:
        def __init__(self, *_: object, **__: object) -> None:
            self.tools_registered: list[tuple[object, object]] = []

        def tool(
            self, *args: object, **kwargs: object
        ) -> Callable[[Callable[..., object]], Callable[..., object]]:
            _ = args
            _ = kwargs

            def _decorator(func: Callable[..., object]) -> Callable[..., object]:
                self.tools_registered.append((args, func))
                return func

            return _decorator

        def run(self) -> None:
            nonlocal run_called
            run_called = True

    def _backend_factory(_: ServingConfig, **__: object) -> object:
        return SimpleNamespace(
            backend=SimpleNamespace(service="svc", limits={"default_limit": 10}),
            close=lambda: None,
        )

    def _register_all_tools(
        mcp: object, backend: object, cfg: ServingConfig | None = None
    ) -> None:
        _ = mcp
        register_calls.append((backend, cfg))

    monkeypatch.setattr(server, "FastMCP", _FastMcpStub)
    monkeypatch.setattr(server, "register_tools", _register_all_tools)

    cfg = ServingConfig(mode="remote_api", api_base_url="https://api.invalid")
    srv, close = server.create_mcp_server(
        cfg=cfg,
        backend_factory=cast("server.BackendFactory", _backend_factory),
        gateway=cast("StorageGateway", SimpleNamespace()),
        register_tools_fn=None,
    )
    assert register_calls
    assert register_calls[0][0] == "svc"
    srv.run()
    assert run_called
    close()
