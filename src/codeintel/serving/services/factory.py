"""Factories for building shared query services."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from codeintel.analytics.graph_runtime import GraphRuntimeOptions, build_graph_runtime
from codeintel.config.primitives import GraphBackendConfig, SnapshotRef
from codeintel.config.serving_models import ServingConfig, verify_db_identity
from codeintel.graphs.engine import GraphEngine
from codeintel.serving.http.datasets import (
    build_registry_and_limits,
    describe_dataset,
    validate_dataset_registry,
)
from codeintel.serving.mcp.query_service import BackendLimits, DuckDBQueryService
from codeintel.serving.services.query_service import (
    HttpQueryService,
    LocalQueryService,
    ServiceObservability,
)
from codeintel.serving.services.wiring import BackendResource, build_backend_resource
from codeintel.storage.gateway import StorageGateway


def _split_table_identifier(table: str) -> tuple[str, str]:
    """
    Split a schema-qualified table name into schema and table components.

    Parameters
    ----------
    table:
        Schema-qualified table name (e.g., "core.ast_nodes").

    Returns
    -------
    tuple[str, str]
        Schema name and table name.

    Raises
    ------
    ValueError
        When the identifier is not schema-qualified.
    """
    if "." not in table:
        message = f"Table identifier must be schema-qualified: {table}"
        raise ValueError(message)
    schema_name, table_name = table.split(".", maxsplit=1)
    return schema_name, table_name


@dataclass
class DatasetRegistryOptions:
    """Options controlling dataset registry composition and validation."""

    tables: dict[str, str] | None = None
    describe_fn: Callable[[str, str], str] = describe_dataset
    validate: bool = True


@dataclass(frozen=True)
class ServiceBuildOptions:
    """Optional knobs for constructing query services."""

    registry: DatasetRegistryOptions | None = None
    observability: ServiceObservability | None = None
    graph_engine: GraphEngine | None = None


def get_observability_from_config(cfg: ServingConfig) -> ServiceObservability | None:
    """
    Derive service observability settings from configuration flags.

    Parameters
    ----------
    cfg:
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
    gateway:
        StorageGateway providing the DuckDB connection and dataset registry.
    cfg:
        Serving configuration describing repo/commit and limits.
    query:
        Pre-constructed DuckDBQueryService to attach to the LocalQueryService.
    registry:
        Dataset registry options including validation behavior.
    observability:
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
    request_json:
        Callable that performs HTTP GETs and returns decoded JSON.
    limits:
        BackendLimits instance controlling clamping.
    observability:
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
    cfg:
        Validated serving configuration.
    gateway:
        StorageGateway for local_db mode.
    request_json:
        HTTP JSON request callable for remote_api mode.
    options:
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
    if cfg.mode == "local_db":
        if gateway is None:
            message = "StorageGateway is required for local_db service construction"
            raise ValueError(message)
        registry_opts = resolved_options.registry or DatasetRegistryOptions()
        engine = resolved_options.graph_engine
        if engine is None:
            snapshot = SnapshotRef(repo=cfg.repo, commit=cfg.commit, repo_root=cfg.repo_root)
            runtime = build_graph_runtime(
                gateway,
                GraphRuntimeOptions(snapshot=snapshot, backend=GraphBackendConfig()),
            )
            engine = runtime.engine
        query = DuckDBQueryService(
            gateway=gateway,
            repo=cfg.repo,
            commit=cfg.commit,
            limits=limits,
            graph_engine=engine,
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


__all__ = [
    "BackendResource",
    "DatasetRegistryOptions",
    "ServiceBuildOptions",
    "ServiceObservability",
    "build_backend_resource",
    "build_http_query_service",
    "build_local_query_service",
    "build_service_from_config",
    "get_observability_from_config",
]
