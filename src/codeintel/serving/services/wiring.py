"""Shared backend wiring helpers for HTTP and MCP surfaces.

Preferred import path: ``from codeintel.serving.services.factory import build_backend_resource``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import anyio
import httpx

from codeintel.analytics.graph_runtime import build_graph_runtime
from codeintel.config.primitives import GraphBackendConfig, SnapshotRef
from codeintel.config.serving_models import ServingConfig, verify_db_identity
from codeintel.serving.http.datasets import build_registry_and_limits
from codeintel.serving.mcp.query_service import BackendLimits
from codeintel.serving.services.query_service import (
    LocalQueryService,
    QueryService,
    ServiceObservability,
)
from codeintel.storage.gateway import StorageGateway
from codeintel.storage.views import create_all_views

if TYPE_CHECKING:
    from codeintel.serving.mcp.backend import QueryBackend
    from codeintel.serving.services.factory import DatasetRegistryOptions

__all__ = ["BackendResource", "build_backend_resource"]


@dataclass
class BackendResource:
    """Bundle of backend, service, and cleanup hook."""

    backend: QueryBackend
    service: QueryService
    close: Callable[[], None]


def build_backend_resource(
    cfg: ServingConfig,
    *,
    gateway: StorageGateway | None = None,
    http_client: httpx.Client | httpx.AsyncClient | None = None,
    registry: DatasetRegistryOptions | None = None,
    observability: ServiceObservability | None = None,
) -> BackendResource:
    """
    Construct a backend and shared service with unified wiring.

    Requires a ``StorageGateway`` for local_db mode; legacy connection paths are removed.

    Parameters
    ----------
    cfg:
        Validated serving configuration.
    gateway:
        StorageGateway supplying connection and dataset registry for local_db mode.
    http_client:
        Optional pre-built HTTPX client for remote_api mode.
    registry:
        Optional dataset registry options.
    observability:
        Optional observability configuration.

    Returns
    -------
    BackendResource
        Backend, service, and close hook suitable for server/MCP startup.

    Raises
    ------
    ValueError
        When required inputs are missing for the configured mode or unsupported modes are requested.
    """
    from codeintel.serving.services.factory import (  # noqa: PLC0415
        DatasetRegistryOptions,
        get_observability_from_config,
    )

    resolved_observability = observability or get_observability_from_config(cfg)
    registry_opts = registry or DatasetRegistryOptions()
    _, limits = build_registry_and_limits(cfg)

    if cfg.mode == "local_db":
        return _build_local_resource(
            cfg,
            gateway=gateway,
            registry_opts=registry_opts,
            observability=resolved_observability,
            limits=limits,
        )

    if cfg.mode == "remote_api":
        return _build_remote_resource(
            cfg,
            http_client=http_client,
            observability=resolved_observability,
            limits=limits,
        )

    message = f"Unsupported serving mode: {cfg.mode}"
    raise ValueError(message)


def _build_local_resource(
    cfg: ServingConfig,
    *,
    gateway: StorageGateway | None,
    registry_opts: DatasetRegistryOptions,
    observability: ServiceObservability | None,
    limits: BackendLimits,
) -> BackendResource:
    """
    Construct a local DuckDB backend and service bundle.

    Returns
    -------
    BackendResource
        Backend, service, and close hook.

    Raises
    ------
    ValueError
        When the gateway is missing for local_db mode.
    """
    if gateway is None:
        message = "StorageGateway is required for local_db mode"
        raise ValueError(message)
    if cfg.db_path is None:
        message = "db_path is required for local_db mode"
        raise ValueError(message)
    connection = gateway.con
    effective_read_only = gateway.config.read_only

    verify_db_identity(gateway, cfg)
    if not effective_read_only:
        create_all_views(connection)

    from codeintel.serving.mcp.backend import DuckDBBackend  # noqa: PLC0415
    from codeintel.serving.services.factory import build_service_from_config  # noqa: PLC0415

    service = build_service_from_config(
        cfg,
        gateway=gateway,
        registry=registry_opts,
        observability=observability,
        graph_engine=runtime.engine,
    )
    snapshot = SnapshotRef(repo=cfg.repo, commit=cfg.commit, repo_root=cfg.repo_root)
    runtime = build_graph_runtime(
        gateway,
        snapshot,
        GraphBackendConfig(),
    )
    backend = DuckDBBackend(
        gateway=gateway,
        repo=cfg.repo,
        commit=cfg.commit,
        limits=limits,
        observability=observability,
        query_engine=runtime.engine,
        service_override=service if isinstance(service, LocalQueryService) else None,
    )

    def _close() -> None:
        if gateway is not None:
            gateway.close()

    return BackendResource(backend=backend, service=backend.service, close=_close)


def _build_remote_resource(
    cfg: ServingConfig,
    *,
    http_client: httpx.Client | httpx.AsyncClient | None,
    observability: ServiceObservability | None,
    limits: BackendLimits,
) -> BackendResource:
    """
    Construct a remote HTTP backend and service bundle.

    Returns
    -------
    BackendResource
        Backend, service, and close hook.

    Raises
    ------
    ValueError
        When api_base_url is missing for remote_api mode.
    """
    if not cfg.api_base_url:
        message = "api_base_url is required for remote_api mode"
        raise ValueError(message)

    owns_client = False
    client = http_client
    if client is None:
        client = httpx.Client(base_url=cfg.api_base_url, timeout=cfg.timeout_seconds)
        owns_client = True

    from codeintel.serving.mcp.backend import HttpBackend  # noqa: PLC0415

    backend = HttpBackend(
        base_url=cfg.api_base_url,
        repo=cfg.repo,
        commit=cfg.commit,
        timeout=cfg.timeout_seconds,
        limits=limits,
        client=client,
        observability=observability,
    )

    def _close_http() -> None:
        if not owns_client or client is None:
            return
        if isinstance(client, httpx.Client):
            client.close()
            return
        if isinstance(client, httpx.AsyncClient):

            async def _aclose_client(async_client: httpx.AsyncClient) -> None:
                await async_client.aclose()

            anyio.run(_aclose_client, client)

    return BackendResource(backend=backend, service=backend.service, close=_close_http)
