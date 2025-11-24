"""Shared backend wiring helpers for HTTP and MCP surfaces.

Preferred import path: ``from codeintel.services.factory import build_backend_resource``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import anyio
import duckdb
import httpx

from codeintel.config.serving_models import ServingConfig, verify_db_identity
from codeintel.mcp.query_service import BackendLimits
from codeintel.server.datasets import build_dataset_registry, build_registry_and_limits
from codeintel.services.query_service import (
    LocalQueryService,
    QueryService,
    ServiceObservability,
)
from codeintel.storage.gateway import StorageGateway
from codeintel.storage.views import create_all_views

if TYPE_CHECKING:
    from codeintel.mcp.backend import QueryBackend
    from codeintel.services.factory import DatasetRegistryOptions

__all__ = ["BackendResource", "build_backend_resource"]


@dataclass
class BackendResource:
    """Bundle of backend, service, and cleanup hook."""

    backend: QueryBackend
    service: QueryService
    close: Callable[[], None]


def build_backend_resource(  # noqa: PLR0913
    cfg: ServingConfig,
    *,
    gateway: StorageGateway | None = None,
    con: duckdb.DuckDBPyConnection | None = None,
    http_client: httpx.Client | httpx.AsyncClient | None = None,
    registry: DatasetRegistryOptions | None = None,
    observability: ServiceObservability | None = None,
    read_only: bool | None = None,
) -> BackendResource:
    """
    Construct a backend and shared service with unified wiring.

    Prefer supplying a ``StorageGateway`` for local_db mode; ``con`` and manual
    registry overrides remain for backward compatibility only.

    Parameters
    ----------
    cfg:
        Validated serving configuration.
    gateway:
        Optional StorageGateway supplying connection and dataset registry for local_db mode.
        Preferred over raw ``con``/``registry`` for consistency.
    con:
        Optional DuckDB connection for local_db mode.
    http_client:
        Optional pre-built HTTPX client for remote_api mode.
    registry:
        Optional dataset registry options.
    observability:
        Optional observability configuration.
    read_only:
        Optional override for DuckDB read-only flag; defaults to cfg.read_only.

    Returns
    -------
    BackendResource
        Backend, service, and close hook suitable for server/MCP startup.

    Raises
    ------
    ValueError
        When required inputs are missing for the configured mode or unsupported modes are requested.
    """
    from codeintel.services.factory import (  # noqa: PLC0415
        DatasetRegistryOptions,
        get_observability_from_config,
    )

    resolved_observability = observability or get_observability_from_config(cfg)
    registry_opts = registry or DatasetRegistryOptions()
    if registry_opts.tables is None and gateway is not None:
        registry_opts.tables = dict(gateway.datasets.mapping)
    resolved_read_only = cfg.read_only if read_only is None else read_only
    registry_default, limits = build_registry_and_limits(cfg)
    if registry_opts.tables is None:
        registry_opts.tables = registry_default

    if cfg.mode == "local_db":
        return _build_local_resource(
            cfg,
            gateway=gateway,
            con=con,
            registry_opts=registry_opts,
            observability=resolved_observability,
            read_only=resolved_read_only,
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


def _build_local_resource(  # noqa: PLR0913
    cfg: ServingConfig,
    *,
    gateway: StorageGateway | None,
    con: duckdb.DuckDBPyConnection | None,
    registry_opts: DatasetRegistryOptions,
    observability: ServiceObservability | None,
    read_only: bool,
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
        When the DuckDB path is missing for local_db mode.
    """
    owns_con = False
    connection = con or (gateway.con if gateway is not None else None)
    effective_read_only = gateway.config.read_only if gateway is not None else read_only
    if connection is None:
        if cfg.db_path is None:
            message = "db_path is required for local_db mode"
            raise ValueError(message)
        connection = duckdb.connect(str(cfg.db_path), read_only=effective_read_only)
        owns_con = True

    verify_db_identity(connection, cfg)
    if not effective_read_only:
        create_all_views(connection)

    tables = registry_opts.tables if registry_opts.tables is not None else build_dataset_registry()
    from codeintel.mcp.backend import DuckDBBackend  # noqa: PLC0415
    from codeintel.services.factory import build_service_from_config  # noqa: PLC0415

    service = build_service_from_config(
        cfg,
        con=connection,
        registry=registry_opts,
        observability=observability,
    )
    backend = DuckDBBackend(
        gateway=gateway,
        con=connection,
        repo=cfg.repo,
        commit=cfg.commit,
        limits=limits,
        dataset_tables=tables,
        observability=observability,
        service_override=service if isinstance(service, LocalQueryService) else None,
    )

    def _close() -> None:
        if owns_con and connection is not None:
            connection.close()

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

    from codeintel.mcp.backend import HttpBackend  # noqa: PLC0415

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
