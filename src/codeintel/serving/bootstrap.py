"""Unified service bootstrap module for serving layer.

This module provides a single entry point for constructing query services,
backends, and their dependencies.

Usage
-----
For most use cases, use the high-level builder:

    from codeintel.serving.bootstrap import build_service_stack

    stack = build_service_stack(config, gateway=gateway)
    # stack.backend, stack.service, stack.close()

For backend resource construction (includes both local and remote modes):

    from codeintel.serving.bootstrap import build_backend_resource

For more control, use the component builders:

    from codeintel.serving.bootstrap import (
        build_backend_context,
        build_repositories,
        build_query_service,
    )
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from importlib import import_module
from typing import TYPE_CHECKING

import anyio
import httpx

from codeintel.analytics.graph_runtime import (
    GraphRuntime,
    GraphRuntimeOptions,
    GraphRuntimePool,
    build_graph_runtime,
)
from codeintel.config.primitives import GraphBackendConfig, SnapshotRef
from codeintel.config.serving_models import ServingConfig, verify_db_identity
from codeintel.serving.backend import (
    BackendContext,
    BackendLimits,
    DuckDBQueryService,
    DuckDBRepositories,
    GraphEngineProvider,
)
from codeintel.serving.backend.datasets import (
    build_registry_and_limits,
    describe_dataset,
    validate_dataset_registry,
)
from codeintel.serving.services.observability import ServiceObservability
from codeintel.serving.services.query_service import (
    HttpQueryService,
    LocalQueryService,
    QueryService,
)
from codeintel.storage.gateway import StorageGateway
from codeintel.storage.views import create_all_views

if TYPE_CHECKING:
    from codeintel.graphs.engine import GraphEngine
    from codeintel.serving.mcp.backend import DuckDBBackend, HttpBackend, QueryBackend


# =============================================================================
# Configuration Types
# =============================================================================


@dataclass
class DatasetRegistryOptions:
    """
    Options controlling dataset registry composition and validation.

    Parameters
    ----------
    tables
        Optional mapping of table identifiers to descriptions.
    describe_fn
        Function to describe datasets (defaults to describe_dataset).
    validate
        Whether to validate the registry on construction.
    """

    tables: dict[str, str] | None = None
    describe_fn: Callable[[str, str], str] = describe_dataset
    validate: bool = True


@dataclass(frozen=True)
class ServiceBuildOptions:
    """
    Optional knobs for constructing query services.

    Parameters
    ----------
    registry
        Dataset registry options.
    observability
        Observability configuration.
    graph_runtime
        Pre-built graph runtime to reuse.
    graph_engine
        Pre-built graph engine to reuse.
    """

    registry: DatasetRegistryOptions | None = None
    observability: ServiceObservability | None = None
    graph_runtime: GraphRuntime | None = None
    graph_engine: GraphEngine | None = None


# =============================================================================
# Service Stack
# =============================================================================


@dataclass
class ServiceStack:
    """
    Bundle of service components with lifecycle management.

    This dataclass encapsulates all the components needed for serving,
    providing a clean interface for startup and shutdown.

    Parameters
    ----------
    service
        The query service for business logic operations.
    query
        The underlying DuckDB query service.
    context
        Backend context with gateway and configuration.
    repositories
        Repository collection for data access.
    runtime
        Graph runtime (if constructed).
    close_fn
        Cleanup function to release resources.
    """

    service: QueryService
    query: DuckDBQueryService
    context: BackendContext
    repositories: DuckDBRepositories
    runtime: GraphRuntime | None = None
    close_fn: Callable[[], None] = field(default=lambda: None)

    def close(self) -> None:
        """Release all resources associated with this service stack."""
        self.close_fn()


@dataclass
class BootstrapOptions:
    """
    Configuration options for service bootstrap.

    Parameters
    ----------
    create_views
        Whether to create docs views on startup.
    validate_registry
        Whether to validate the dataset registry.
    observability
        Optional observability configuration.
    graph_runtime
        Optional pre-built graph runtime to reuse.
    graph_engine
        Optional pre-built graph engine to reuse.
    """

    create_views: bool = True
    validate_registry: bool = True
    observability: ServiceObservability | None = None
    graph_runtime: GraphRuntime | None = None
    graph_engine: GraphEngine | None = None


def build_backend_context(
    gateway: StorageGateway,
    config: ServingConfig,
    *,
    limits: BackendLimits | None = None,
    graph_engine: GraphEngine | None = None,
) -> BackendContext:
    """
    Construct a BackendContext from a gateway and configuration.

    Parameters
    ----------
    gateway
        Storage gateway providing DuckDB connection.
    config
        Serving configuration with repo/commit.
    limits
        Backend limits for pagination; derived from config if not provided.
    graph_engine
        Optional pre-built graph engine.

    Returns
    -------
    BackendContext
        Context bundle for query services.
    """
    if limits is None:
        _, limits = build_registry_and_limits(config)
    return BackendContext(
        gateway=gateway,
        repo=config.repo,
        commit=config.commit,
        limits=limits,
        graph_engine=graph_engine,
    )


def build_repositories(
    gateway: StorageGateway,
    config: ServingConfig,
) -> DuckDBRepositories:
    """
    Construct the repository collection for a gateway.

    Parameters
    ----------
    gateway
        Storage gateway providing DuckDB connection.
    config
        Serving configuration with repo/commit.

    Returns
    -------
    DuckDBRepositories
        Lazily-constructed repository collection.
    """
    return DuckDBRepositories(
        gateway=gateway,
        repo=config.repo,
        commit=config.commit,
    )


def build_graph_runtime_for_config(
    gateway: StorageGateway,
    config: ServingConfig,
    *,
    backend_config: GraphBackendConfig | None = None,
) -> GraphRuntime:
    """
    Build a graph runtime from configuration.

    Parameters
    ----------
    gateway
        Storage gateway for graph data.
    config
        Serving configuration with repo/commit.
    backend_config
        Optional graph backend configuration.

    Returns
    -------
    GraphRuntime
        Configured graph runtime with engine.
    """
    snapshot = SnapshotRef(
        repo=config.repo,
        commit=config.commit,
        repo_root=config.repo_root,
    )
    options = GraphRuntimeOptions(
        snapshot=snapshot,
        backend=backend_config or GraphBackendConfig(),
        features=config.graph_features,
    )
    return build_graph_runtime(gateway, options)


def build_query_service(
    context: BackendContext,
    repositories: DuckDBRepositories,
    engine_provider: GraphEngineProvider,
) -> DuckDBQueryService:
    """
    Construct the DuckDB query service from components.

    Parameters
    ----------
    context
        Backend context with gateway and limits.
    repositories
        Repository collection for data access.
    engine_provider
        Provider for graph engine access.

    Returns
    -------
    DuckDBQueryService
        Configured query service.
    """
    return DuckDBQueryService(
        context=context,
        repositories=repositories,
        engine_provider=engine_provider,
    )


def build_service_stack(
    config: ServingConfig,
    *,
    gateway: StorageGateway,
    options: BootstrapOptions | None = None,
) -> ServiceStack:
    """
    Build a complete service stack from configuration.

    This is the primary entry point for constructing the serving layer.
    It handles all the wiring of components and provides a clean
    interface for lifecycle management.

    Parameters
    ----------
    config
        Serving configuration.
    gateway
        Storage gateway providing DuckDB connection.
    options
        Optional bootstrap configuration.

    Returns
    -------
    ServiceStack
        Complete service stack ready for use.

    Examples
    --------
    >>> stack = build_service_stack(config, gateway=gateway)
    >>> try:
    ...     result = stack.service.get_function_summary(goid_h128=123)
    ... finally:
    ...     stack.close()
    """
    opts = options or BootstrapOptions()

    # Verify gateway matches configuration
    verify_db_identity(gateway, config)

    # Create views if requested and not read-only
    if opts.create_views and not gateway.config.read_only:
        create_all_views(gateway.con)

    # Build limits
    _, limits = build_registry_and_limits(config)

    # Resolve graph runtime/engine
    runtime: GraphRuntime | None = None
    engine: GraphEngine | None = opts.graph_engine

    if opts.graph_runtime is not None:
        runtime = opts.graph_runtime
        engine = runtime.engine
    elif engine is None:
        runtime = build_graph_runtime_for_config(gateway, config)
        engine = runtime.engine

    # Build context and repositories
    context = build_backend_context(
        gateway,
        config,
        limits=limits,
        graph_engine=engine,
    )
    repositories = build_repositories(gateway, config)

    # Build engine provider and query service
    engine_provider = GraphEngineProvider(context=context, graph_engine=engine)
    query = build_query_service(context, repositories, engine_provider)

    # Build local query service wrapper
    service: LocalQueryService = LocalQueryService(
        query=query,
        observability=opts.observability,
    )

    def _close() -> None:
        gateway.close()

    return ServiceStack(
        service=service,
        query=query,
        context=context,
        repositories=repositories,
        runtime=runtime,
        close_fn=_close,
    )


def get_observability_from_config(cfg: ServingConfig) -> ServiceObservability | None:
    """
    Derive service observability settings from configuration flags.

    Parameters
    ----------
    cfg
        Serving configuration that may include observability toggles.

    Returns
    -------
    ServiceObservability | None
        Enabled observability config when toggled on; otherwise ``None``.
    """
    enabled = bool(
        getattr(cfg, "enable_observability", False) or getattr(cfg, "observability_enabled", False)
    )
    if not enabled:
        return None
    return ServiceObservability(enabled=True)


def build_local_query_service(
    gateway: StorageGateway,
    cfg: ServingConfig,
    *,
    query: DuckDBQueryService,
    registry: DatasetRegistryOptions | None = None,
    observability: ServiceObservability | None = None,
) -> LocalQueryService:
    """
    Construct a LocalQueryService with identity verification and dataset registry.

    Parameters
    ----------
    gateway
        StorageGateway providing the DuckDB connection and dataset registry.
    cfg
        Serving configuration describing repo/commit and limits.
    query
        Pre-constructed DuckDBQueryService to attach to the LocalQueryService.
    registry
        Dataset registry options including validation behavior.
    observability
        Optional observability configuration for structured logging.

    Returns
    -------
    LocalQueryService
        Service bound to the provided DuckDB connection.
    """
    verify_db_identity(gateway, cfg)
    opts = registry or DatasetRegistryOptions()
    if opts.validate:
        validate_dataset_registry(gateway)
    return LocalQueryService(
        query=query,
        describe_dataset_fn=opts.describe_fn,
        observability=observability,
    )


def build_http_query_service(
    request_json: Callable[[str, dict[str, object]], object],
    *,
    limits: BackendLimits,
    observability: ServiceObservability | None = None,
) -> HttpQueryService:
    """
    Construct an HttpQueryService for remote API delegation.

    Parameters
    ----------
    request_json
        Callable that performs HTTP GETs and returns decoded JSON.
    limits
        BackendLimits instance controlling clamping.
    observability
        Optional observability configuration for structured logging.

    Returns
    -------
    HttpQueryService
        Service wrapper for remote transport.
    """
    return HttpQueryService(
        request_json=request_json,
        limits=limits,
        observability=observability,
    )


def build_service_from_config(
    cfg: ServingConfig,
    *,
    gateway: StorageGateway | None = None,
    request_json: Callable[[str, dict[str, object]], object] | None = None,
    options: ServiceBuildOptions | None = None,
) -> LocalQueryService | HttpQueryService:
    """
    Construct a query service from ServingConfig using local or remote transport.

    Parameters
    ----------
    cfg
        Validated serving configuration.
    gateway
        StorageGateway for local_db mode.
    request_json
        HTTP JSON request callable for remote_api mode.
    options
        Optional bundle configuring registry, observability, and graph engine.

    Returns
    -------
    LocalQueryService | HttpQueryService
        Transport-agnostic service bound to DuckDB or HTTP.

    Raises
    ------
    ValueError
        When required inputs (gateway for local_db or request_json for remote_api)
        are missing or the serving mode is unsupported.
    """
    _, limits = build_registry_and_limits(cfg)
    resolved_options = options or ServiceBuildOptions()
    resolved_observability = resolved_options.observability or get_observability_from_config(cfg)
    runtime = resolved_options.graph_runtime

    if cfg.mode == "local_db":
        if gateway is None:
            message = "StorageGateway is required for local_db service construction"
            raise ValueError(message)
        registry_opts = resolved_options.registry or DatasetRegistryOptions()
        engine = resolved_options.graph_engine
        if runtime is not None:
            if runtime.options.snapshot is not None and (
                runtime.options.snapshot.repo != cfg.repo
                or runtime.options.snapshot.commit != cfg.commit
            ):
                message = "GraphRuntime snapshot mismatch for serving configuration"
                raise ValueError(message)
            engine = runtime.engine
        if engine is None:
            snapshot = SnapshotRef(repo=cfg.repo, commit=cfg.commit, repo_root=cfg.repo_root)
            runtime = build_graph_runtime(
                gateway,
                GraphRuntimeOptions(snapshot=snapshot, backend=GraphBackendConfig()),
            )
            engine = runtime.engine
        context = BackendContext(
            gateway=gateway,
            repo=cfg.repo,
            commit=cfg.commit,
            limits=limits,
            graph_engine=engine,
        )
        repositories = DuckDBRepositories(
            gateway=gateway,
            repo=cfg.repo,
            commit=cfg.commit,
        )
        provider = GraphEngineProvider(context=context, graph_engine=engine)
        query = DuckDBQueryService(
            context=context,
            repositories=repositories,
            engine_provider=provider,
        )
        return build_local_query_service(
            gateway,
            cfg,
            query=query,
            registry=registry_opts,
            observability=resolved_observability,
        )

    if cfg.mode == "remote_api":
        if request_json is None:
            message = "request_json callable is required for remote_api service construction"
            raise ValueError(message)
        return build_http_query_service(
            request_json=request_json,
            limits=limits,
            observability=resolved_observability,
        )

    message = f"Unsupported serving mode: {cfg.mode}"
    raise ValueError(message)


# =============================================================================
# Backend Resource
# =============================================================================

LOG = logging.getLogger(__name__)


def _load_mcp_backends() -> tuple[type[DuckDBBackend], type[HttpBackend]]:
    """
    Deferred import helper to avoid import cycles and heavy imports at module load.

    Returns
    -------
    tuple[type[DuckDBBackend], type[HttpBackend]]
        Backend classes for DuckDB and HTTP transports.
    """
    module = import_module("codeintel.serving.mcp.backend")
    return module.DuckDBBackend, module.HttpBackend


@dataclass
class BackendResource:
    """Bundle of backend, service, and cleanup hook."""

    backend: QueryBackend
    service: QueryService
    close: Callable[[], None]


@dataclass
class BackendResourceOptions:
    """Options controlling backend construction for serving."""

    registry: DatasetRegistryOptions | None = None
    observability: ServiceObservability | None = None
    graph_runtime: GraphRuntime | None = None
    runtime_pool: GraphRuntimePool | None = None


def build_backend_resource(
    cfg: ServingConfig,
    *,
    gateway: StorageGateway | None = None,
    http_client: httpx.Client | httpx.AsyncClient | None = None,
    options: BackendResourceOptions | None = None,
) -> BackendResource:
    """
    Construct a backend and shared service with unified wiring.

    Requires a ``StorageGateway`` for local_db mode; direct connection paths are removed.

    Parameters
    ----------
    cfg
        Validated serving configuration.
    gateway
        StorageGateway supplying connection and dataset registry for local_db mode.
    http_client
        Optional pre-built HTTPX client for remote_api mode.
    options
        Optional bundle controlling registry, observability, and runtime reuse.

    Returns
    -------
    BackendResource
        Backend, service, and close hook suitable for server/MCP startup.

    Raises
    ------
    ValueError
        When required inputs are missing for the configured mode or unsupported modes are requested.
    """
    resolved_options = options or BackendResourceOptions()
    resolved_observability = resolved_options.observability or get_observability_from_config(cfg)
    registry_opts = resolved_options.registry or DatasetRegistryOptions()
    _, limits = build_registry_and_limits(cfg)

    if cfg.mode == "local_db":
        return _build_local_resource(
            cfg,
            gateway=gateway,
            options=BackendResourceOptions(
                registry=registry_opts,
                observability=resolved_observability,
                graph_runtime=resolved_options.graph_runtime,
                runtime_pool=resolved_options.runtime_pool,
            ),
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
    limits: BackendLimits,
    options: BackendResourceOptions,
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

    duckdb_backend_cls, _ = _load_mcp_backends()

    snapshot = SnapshotRef(repo=cfg.repo, commit=cfg.commit, repo_root=cfg.repo_root)
    runtime_opts = GraphRuntimeOptions(
        snapshot=snapshot, backend=GraphBackendConfig(), features=cfg.graph_features
    )
    runtime_source = "new"
    if options.graph_runtime is not None:
        active_runtime = options.graph_runtime
        runtime_source = "provided"
    elif options.runtime_pool is not None:
        active_runtime = options.runtime_pool.get(gateway, runtime_opts)
        runtime_source = "pool"
    else:
        active_runtime = build_graph_runtime(
            gateway,
            runtime_opts,
        )
    service = build_service_from_config(
        cfg,
        gateway=gateway,
        options=ServiceBuildOptions(
            registry=options.registry,
            observability=options.observability,
            graph_runtime=active_runtime,
        ),
    )
    backend = duckdb_backend_cls(
        gateway=gateway,
        repo=cfg.repo,
        commit=cfg.commit,
        limits=limits,
        observability=options.observability,
        query_engine=active_runtime.engine,
        service_override=service if isinstance(service, LocalQueryService) else None,
    )
    LOG.info(
        "serving.backend wired repo=%s commit=%s runtime_source=%s backend=%s use_gpu=%s "
        "features=%s",
        cfg.repo,
        cfg.commit,
        runtime_source,
        active_runtime.backend.backend,
        active_runtime.backend.use_gpu,
        active_runtime.options.features,
    )
    if options.graph_runtime is not None:
        LOG.info(
            "serving.backend using provided runtime with engine=%s cache_key=%s",
            type(active_runtime.engine).__name__,
            active_runtime.options.cache_key,
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

    _, http_backend_cls = _load_mcp_backends()

    backend = http_backend_cls(
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


__all__ = [
    "BackendResource",
    "BackendResourceOptions",
    "BootstrapOptions",
    "DatasetRegistryOptions",
    "ServiceBuildOptions",
    "ServiceStack",
    "build_backend_context",
    "build_backend_resource",
    "build_graph_runtime_for_config",
    "build_http_query_service",
    "build_local_query_service",
    "build_query_service",
    "build_repositories",
    "build_service_from_config",
    "build_service_stack",
    "get_observability_from_config",
]
