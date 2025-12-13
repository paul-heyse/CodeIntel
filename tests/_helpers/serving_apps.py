"""Shared builders for LocalQueryService-backed FastAPI apps in tests."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from fastapi.testclient import TestClient

from codeintel.config.serving_models import ServingConfig
from codeintel.serving.backend import BackendLimits
from codeintel.serving.bootstrap import BackendResource
from codeintel.serving.http.fastapi import create_app
from codeintel.serving.mcp.backend import DuckDBBackend
from codeintel.serving.services.query_service import LocalQueryService
from tests._helpers.gateway import build_duckdb_query_service
from tests._helpers.serving_harnesses import RecordingObservability

if TYPE_CHECKING:
    from collections.abc import Iterator

    from fastapi import FastAPI

    from codeintel.serving.services.observability import ServiceObservability
    from codeintel.storage.gateway import StorageGateway
    from tests.serving.mcp.conftest import McpBackendComponents

DEFAULT_LIMIT = 10
MAX_ROWS = 100


@dataclass(frozen=True)
class ServiceApp:
    """Aggregated service, backend, and app for delegate tests."""

    gateway: StorageGateway
    repo: str
    commit: str
    limits: BackendLimits
    service: LocalQueryService
    backend: DuckDBBackend
    app: FastAPI
    observability: ServiceObservability

    @contextmanager
    def client(self) -> Iterator[TestClient]:
        """Provide a TestClient bound to the configured app.

        Yields
        ------
        TestClient
            Configured client bound to the service app.
        """
        with TestClient(self.app) as test_client:
            yield test_client


def build_service_app(
    gateway: StorageGateway,
    *,
    snapshot: tuple[str, str],
    limits: BackendLimits | None = None,
    observability: ServiceObservability | None = None,
    components: McpBackendComponents | None = None,
) -> ServiceApp:
    """Construct a LocalQueryService, DuckDB backend, and FastAPI app for tests.

    Returns
    -------
    ServiceApp
        Aggregated service, backend, app, and observability harness.
    """
    repo, commit = snapshot
    effective_limits = limits or BackendLimits(
        default_limit=DEFAULT_LIMIT,
        max_rows_per_call=MAX_ROWS,
    )
    effective_observability = observability or RecordingObservability()

    if components is not None:
        repo = components.repo
        commit = components.commit
        gateway = components.gateway
        effective_limits = components.limits
        service = components.service
        if observability is not None:
            service.observability = observability
        effective_observability = getattr(service, "observability", None) or effective_observability
        backend = components.backend
    else:
        query = build_duckdb_query_service(
            gateway,
            repo=repo,
            commit=commit,
            limits=effective_limits,
        )
        service = LocalQueryService(
            query=query,
            dataset_tables=dict(gateway.datasets.mapping),
            observability=effective_observability,
        )
        backend = DuckDBBackend(
            gateway=gateway,
            repo=repo,
            commit=commit,
            limits=effective_limits,
            observability=effective_observability,
            service=service,
        )

    def load_config() -> ServingConfig:
        cfg_kwargs: dict[str, Any] = {
            "mode": "remote_api",
            "repo": repo,
            "commit": commit,
            "api_base_url": "http://test",
        }
        return ServingConfig(**cfg_kwargs)

    def backend_factory(cfg: ServingConfig, **_kwargs: object) -> BackendResource:
        _ = cfg
        return BackendResource(backend=backend, service=service, close=lambda: None)

    app = create_app(config_loader=load_config, backend_factory=backend_factory)

    return ServiceApp(
        gateway=gateway,
        repo=repo,
        commit=commit,
        limits=effective_limits,
        service=service,
        backend=backend,
        app=app,
        observability=effective_observability,
    )


__all__ = ["ServiceApp", "build_service_app"]


@dataclass(frozen=True)
class ServiceContext:
    """Bundled service/backend/app with convenience helpers for tests."""

    gateway: StorageGateway
    repo: str
    commit: str
    limits: BackendLimits
    service: LocalQueryService
    backend: DuckDBBackend
    app: FastAPI
    observability: ServiceObservability

    @contextmanager
    def client(self) -> Iterator[TestClient]:
        """Provide a TestClient bound to the configured app.

        Yields
        ------
        TestClient
            Client bound to the service application.
        """
        with TestClient(self.app) as test_client:
            yield test_client


def build_service_context_from_components(
    components: McpBackendComponents,
    *,
    observability: ServiceObservability | None = None,
) -> ServiceContext:
    """Construct a ServiceContext from prebuilt MCP backend components.

    Returns
    -------
    ServiceContext
        Aggregated context with service, backend, and FastAPI app.
    """
    service_app = build_service_app(
        components.gateway,
        snapshot=(components.repo, components.commit),
        limits=components.limits,
        observability=observability or components.service.observability,
        components=components,
    )
    return ServiceContext(
        gateway=service_app.gateway,
        repo=service_app.repo,
        commit=service_app.commit,
        limits=service_app.limits,
        service=service_app.service,
        backend=service_app.backend,
        app=service_app.app,
        observability=service_app.observability,
    )


__all__ += ["ServiceContext", "build_service_context_from_components"]
