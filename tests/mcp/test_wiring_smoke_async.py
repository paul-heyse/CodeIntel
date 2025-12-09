"""Async wiring smoke test using the async registrar helper."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from codeintel.config.serving_models import ServingConfig
from codeintel.serving.bootstrap import BackendResource, build_backend_resource
from codeintel.serving.mcp import server
from tests._helpers.assertions import expect_true
from tests._helpers.gateway import GatewayFactory, seed_repo_identity
from tests._helpers.mcp_registrar import (
    AsyncRecordingMcpRegistrar as AsyncRecordingMcp,
)
from tests._helpers.mcp_registrar import ToolRegistrar


def test_mcp_wiring_smoke_async_registrar() -> None:
    """Server should expose tools when registrar.list_tools is async."""
    called = False
    closed = False
    registered: list[str] = []

    @dataclass
    class FakeService:
        service: str

    cfg = ServingConfig(mode="local_db")
    gateway = GatewayFactory().with_macros().open()
    seed_repo_identity(gateway, repo=cfg.repo, commit=cfg.commit)
    backend_resource = build_backend_resource(cfg, gateway=gateway)

    def _backend_factory(_: ServingConfig, **__: object) -> BackendResource:
        original_close = backend_resource.close

        def _close() -> None:
            nonlocal closed
            closed = True
            original_close()

        backend_resource.close = _close  # type: ignore[assignment]
        return backend_resource

    def _register_tools_fn(registrar: object, svc: object, cfg: ServingConfig | None) -> None:
        _ = svc
        _ = cfg
        nonlocal called
        called = True
        cast("ToolRegistrar", registrar).tool("async_tool")(lambda: None)

    mcp, close = server.create_mcp_server(
        cfg=cfg,
        backend_factory=_backend_factory,
        gateway=gateway,
        register_tools_fn=_register_tools_fn,
        mcp_factory=lambda _name: AsyncRecordingMcp(),
    )

    expect_true(called)
    tools_obj = getattr(mcp, "tools", None)
    expect_true(isinstance(tools_obj, list))
    tools = cast("list[object]", tools_obj)
    registered.extend(tool["name"] for tool in tools if isinstance(tool, dict) and "name" in tool)
    registered.extend(tool.name for tool in tools if hasattr(tool, "name"))
    expect_true("async_tool" in registered)
    close()
    expect_true(closed)
