"""Lightweight tests for MCP server creation."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest

from codeintel.config.serving_models import ServingConfig
from codeintel.serving.bootstrap import BackendResource, build_backend_resource
from codeintel.serving.mcp import server
from tests._helpers.assertions import expect_equal, expect_true
from tests._helpers.gateway import GatewayFactory, seed_repo_identity
from tests._helpers.mcp_registrar import RecordingMcpRegistrar as RecordingMcp


def test_create_mcp_server_requires_gateway_for_local_db() -> None:
    """Local DB mode should enforce gateway requirement."""
    cfg = ServingConfig(mode="local_db")
    with pytest.raises(ValueError, match="StorageGateway is required"):
        server.create_mcp_server(cfg=cfg, gateway=None)


def test_create_mcp_server_uses_register_tools_fn() -> None:
    """Server creation should delegate registration using service when present."""
    cfg = ServingConfig(mode="local_db")
    close_called = False
    register_calls: list[tuple[Any, Any]] = []

    gateway = GatewayFactory().with_macros().open()
    seed_repo_identity(gateway, repo=cfg.repo, commit=cfg.commit)
    backend_resource = build_backend_resource(cfg, gateway=gateway)

    original_close = backend_resource.close
    wrapped_resource = cast(
        "BackendResource",
        SimpleNamespace(
            backend=backend_resource.backend,
            service=backend_resource.service,
            close=None,
        ),
    )

    def _mark_close() -> None:
        nonlocal close_called
        close_called = True
        original_close()

    wrapped_resource.close = _mark_close

    def _backend_factory(*args: tuple[object, ...], **kwargs: object) -> BackendResource:
        _ = (args, kwargs)
        return wrapped_resource

    def _register_tools_fn(mcp: object, svc: object, cfg: ServingConfig | None) -> None:
        _ = cfg
        register_calls.append((mcp, svc))

    srv, close = server.create_mcp_server(
        cfg=cfg,
        backend_factory=_backend_factory,
        gateway=gateway,
        register_tools_fn=_register_tools_fn,
    )

    expect_true(bool(register_calls))
    expect_true(register_calls[0][1] is backend_resource.service)
    close()
    expect_true(close_called)
    expect_true(hasattr(srv, "run"))


def test_main_uses_register_tools_and_runs() -> None:
    """Main flow should wire backend->service and invoke run without real network."""
    run_called = False
    register_calls: list[tuple[object, ServingConfig | None]] = []

    class _RecordingMcp(RecordingMcp):
        def run(self) -> None:
            _ = self
            nonlocal run_called
            run_called = True

    cfg = ServingConfig(mode="local_db")
    gateway = GatewayFactory().with_macros().open()
    seed_repo_identity(gateway, repo=cfg.repo, commit=cfg.commit)
    backend_resource = build_backend_resource(cfg, gateway=gateway)

    def _register_all_tools(mcp: object, backend: object, cfg: ServingConfig | None = None) -> None:
        register_calls.append((backend, cfg))
        _ = mcp

    def _backend_factory(_: ServingConfig, **__: object) -> BackendResource:
        return backend_resource

    srv, close = server.create_mcp_server(
        cfg=cfg,
        backend_factory=_backend_factory,
        gateway=gateway,
        register_tools_fn=_register_all_tools,
        mcp_factory=lambda _name: _RecordingMcp(),
    )
    expect_true(bool(register_calls))
    expect_equal(register_calls[0][0], backend_resource.service)
    cast("_RecordingMcp", srv).run()
    expect_true(run_called)
    close()
