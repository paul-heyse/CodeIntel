"""Async wiring smoke test using the async registrar helper."""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast

from codeintel.config.serving_models import ServingConfig
from codeintel.serving.mcp import server
from tests._helpers.assertions import expect_true
from tests._helpers.mcp_async import AsyncRecordingMcp


def test_mcp_wiring_smoke_async_registrar() -> None:
    """Server should expose tools when registrar.list_tools is async."""
    called = False
    closed = False
    registered: list[str] = []

    def _backend_factory(_: ServingConfig, **__: object) -> SimpleNamespace:
        def _close() -> None:
            nonlocal closed
            closed = True

        return SimpleNamespace(
            backend=SimpleNamespace(service="svc"),
            close=_close,
        )

    def _register_tools_fn(registrar: object, svc: object, cfg: ServingConfig | None) -> None:
        _ = svc
        _ = cfg
        nonlocal called
        called = True
        registrar.tool("async_tool")(lambda: None)  # type: ignore[attr-defined]

    cfg = ServingConfig(mode="remote_api", api_base_url="https://api.invalid")
    mcp, close = server.create_mcp_server(
        cfg=cfg,
        backend_factory=cast("server.BackendFactory", _backend_factory),
        gateway=cast("server.StorageGateway", SimpleNamespace()),
        register_tools_fn=_register_tools_fn,
        mcp_factory=lambda _name: AsyncRecordingMcp(),
    )

    expect_true(called)
    tools = getattr(mcp, "tools", None)
    expect_true(isinstance(tools, list))
    registered.extend(tool["name"] for tool in tools if isinstance(tool, dict))
    expect_true("async_tool" in registered)
    close()
    expect_true(closed)
