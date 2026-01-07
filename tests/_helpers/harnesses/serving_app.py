"""Serving application harnesses for HTTP and MCP tests."""

from __future__ import annotations

from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, TypedDict

from fastapi.testclient import TestClient
from fastmcp.client import Client

from codeintel.serving.context import ServingContext
from codeintel.serving.http.app import create_serving_app
from codeintel.serving.mcp.app import build_mcp_app
from codeintel.serving.runtime import build_runtime
from codeintel.serving.settings import ServingSettings
from tests._helpers.serving_snapshot_factory import ServingSnapshot

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable, Iterator, Mapping
    from pathlib import Path

    from fastmcp import FastMCP
    from fastmcp.client import FastMCPTransport

    from codeintel.serving.mcp.protocols import SemanticKernelProtocol


class ServingSettingsOverrides(TypedDict, total=False):
    serve_dir: Path
    hot_swap: bool
    pool_size: int
    poll_interval_s: float
    mcp_transport: str
    host: str
    port: int
    auth_token: str | None
    schema_enforcement: str
    query_engine: str
    result_engine: str
    api_key: str | None
    cors_origins: tuple[str, ...]
    trusted_hosts: tuple[str, ...]
    gzip_minimum_size: int
    enable_gzip: bool
    export_max_rows: int
    export_batch_size: int
    enable_export_endpoints: bool
    mcp_enable_sampling: bool
    mcp_sample_threshold: int
    mcp_progress_reporting: bool
    mcp_mask_errors: bool
    mcp_max_concurrent_queries: int
    mcp_max_concurrent_exports: int
    mcp_enable_event_store: bool
    mcp_retry_interval_ms: int
    mcp_enable_structured_logging: bool
    mcp_rate_limit_rps: float
    mcp_rate_limit_burst: int
    mcp_cache_listings: bool
    mcp_cache_listings_ttl_seconds: int
    mcp_export_enable_tasks: bool
    mcp_export_ttl_seconds: int | None
    mcp_export_cleanup_interval_seconds: int
    mcp_export_max_full_read_bytes: int
    mcp_export_max_chunk_bytes: int
    mcp_export_max_chunk_lines: int
    uvicorn_workers: int
    uvicorn_loop: str
    uvicorn_http: str
    uvicorn_limit_concurrency: int | None
    uvicorn_limit_max_requests: int | None
    uvicorn_timeout_keep_alive: int
    uvicorn_backlog: int
    uvicorn_access_log: bool
    uvicorn_server_header: bool
    uvicorn_proxy_headers: bool
    uvicorn_forwarded_allow_ips: str
    auth_required_for_remote: bool
    metrics_auth_required: bool
    mcp_enable_search: bool
    mcp_enable_explain: bool
    mcp_enable_meta: bool
    mcp_enable_export: bool


@dataclass(frozen=True)
class ServingAppHarness:
    """Harness for serving HTTP and MCP applications."""

    snapshot: ServingSnapshot
    settings: ServingSettings

    @classmethod
    def from_snapshot(
        cls,
        snapshot: ServingSnapshot,
        *,
        settings_overrides: ServingSettingsOverrides | None = None,
    ) -> ServingAppHarness:
        """Create a harness from a serving snapshot.

        Parameters
        ----------
        snapshot
            Serving snapshot created by ServingSnapshotFactory.
        settings_overrides
            Optional setting overrides to apply.

        Returns
        -------
        ServingAppHarness
            Harness configured with resolved settings.
        """
        base = ServingSettings(
            serve_dir=snapshot.serve_dir,
            pool_size=1,
            poll_interval_s=0.01,
            schema_enforcement="strict",
        )
        resolved = _apply_overrides(base, settings_overrides)
        return cls(snapshot=snapshot, settings=resolved)

    @contextmanager
    def http_client(
        self,
        *,
        mount_mcp: bool = False,
        settings_overrides: ServingSettingsOverrides | None = None,
    ) -> Iterator[TestClient]:
        """Create a FastAPI TestClient for the serving app.

        Parameters
        ----------
        mount_mcp
            Whether to mount the MCP server under `/mcp`.
        settings_overrides
            Optional ServingSettings overrides.

        Yields
        ------
        TestClient
            Active TestClient bound to the serving app.
        """
        settings = _apply_overrides(self.settings, settings_overrides)
        app = create_serving_app(settings=settings, mount_mcp=mount_mcp)
        with TestClient(app) as client:
            yield client

    def build_runtime(
        self,
        *,
        settings_overrides: ServingSettingsOverrides | None = None,
    ) -> ServingContext:
        """Build a serving runtime for the configured snapshot.

        Parameters
        ----------
        settings_overrides
            Optional ServingSettings overrides.

        Returns
        -------
        ServingContext
            Context with a DB manager and semantic kernel.
        """
        settings = _apply_overrides(self.settings, settings_overrides)
        return build_runtime(settings)

    @asynccontextmanager
    async def mcp_client(
        self,
        *,
        settings_overrides: ServingSettingsOverrides | None = None,
        client_kwargs: Mapping[str, Any] | None = None,
        kernel_builder: Callable[[ServingContext], SemanticKernelProtocol] | None = None,
    ) -> AsyncIterator[Client[FastMCPTransport]]:
        """Create a FastMCP client bound to the serving MCP server.

        Parameters
        ----------
        settings_overrides
            Optional ServingSettings overrides.
        client_kwargs
            Optional keyword arguments passed to fastmcp.client.Client.
        kernel_builder
            Optional callback to build a kernel from the runtime.

        Yields
        ------
        Client
            Active FastMCP client.
        """
        settings = _apply_overrides(self.settings, settings_overrides)
        runtime = build_runtime(settings)
        kernel = kernel_builder(runtime) if kernel_builder is not None else runtime.kernel
        mcp_app = _build_mcp_app(runtime, settings, kernel)
        async with Client(mcp_app, **dict(client_kwargs or {})) as client:
            yield client


def _build_mcp_app(
    runtime: ServingContext,
    settings: ServingSettings,
    kernel: SemanticKernelProtocol,
) -> FastMCP:
    @asynccontextmanager
    async def lifespan(_server: FastMCP) -> AsyncIterator[object]:
        await runtime.db_manager.start()
        try:
            yield object()
        finally:
            await runtime.db_manager.stop()

    return build_mcp_app(kernel=kernel, settings=settings, lifespan=lifespan)


def _apply_overrides(
    settings: ServingSettings,
    overrides: ServingSettingsOverrides | None,
) -> ServingSettings:
    if not overrides:
        return settings
    unknown = [key for key in overrides if not hasattr(settings, key)]
    if unknown:
        message = f"Unknown ServingSettings overrides: {sorted(unknown)}"
        raise ValueError(message)
    return replace(settings, **overrides)


__all__ = ["ServingAppHarness"]
