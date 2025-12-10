"""Shared fixtures for HTTP route tests."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING, Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from codeintel.config.serving_models import ServingConfig
from codeintel.serving.backend import BackendLimits
from codeintel.serving.http.fastapi import BackendResource, create_app
from codeintel.serving.http.routes.architecture import build_architecture_router
from codeintel.serving.http.routes.datasets import build_datasets_router
from codeintel.serving.http.routes.functions import build_functions_router
from codeintel.serving.http.routes.health import build_health_router
from codeintel.serving.http.routes.meta import build_meta_router
from codeintel.serving.http.routes.subsystems import build_subsystem_router
from codeintel.storage.gateway import StorageGateway
from tests._helpers.analytics_samples import AnalyticsSamples, load_analytics_samples
from tests._helpers.serving_routes import RouteApp, service_app_factory_with_routes
from tests.serving.http.client_harness import adapt_route
from tests.serving.mcp.conftest import (
    McpBackendComponents,
)
from tests.serving.mcp.conftest import (
    mcp_backend_factory as _mcp_backend_factory,
)

mcp_backend_factory = _mcp_backend_factory

if TYPE_CHECKING:
    from tests._helpers import ProvisionedGateway


@pytest.fixture
def make_http_app(
    mcp_backend_factory: Callable[..., McpBackendComponents],
) -> Callable[..., FastAPI]:
    """Build a FastAPI app factory wired to a DuckDB backend for HTTP route tests.

    Returns
    -------
    Callable[..., FastAPI]
        Factory that constructs configured FastAPI applications given gateway, snapshot,
        optional limits, config overrides, and auto-pipeline settings.
    """

    def _build(
        *,
        gateway: StorageGateway,
        snapshot: tuple[str, str],
        limits: BackendLimits | None = None,
        config_overrides: dict[str, Any] | None = None,
        auto_pipeline: bool = False,
    ) -> FastAPI:
        repo, commit = snapshot
        effective_limits = limits or BackendLimits(default_limit=10, max_rows_per_call=100)
        components = mcp_backend_factory(
            gateway=gateway,
            repo=repo,
            commit=commit,
            limits=effective_limits,
        )
        backend = components.backend
        service = components.service

        def load_config() -> ServingConfig:
            cfg_kwargs: dict[str, Any] = {
                "mode": "remote_api",
                "repo": repo,
                "commit": commit,
                "api_base_url": "http://test",
            }
            if config_overrides:
                cfg_kwargs.update(config_overrides)
            return ServingConfig(**cfg_kwargs)

        def backend_factory(_cfg: ServingConfig, **_kwargs: object) -> BackendResource:
            return BackendResource(backend=backend, service=service, close=lambda: None)

        return create_app(
            config_loader=load_config,
            backend_factory=backend_factory,
            auto_pipeline=auto_pipeline,
        )

    return _build


@pytest.fixture
def architecture_samples(
    architecture_gateway: StorageGateway,
) -> AnalyticsSamples:
    """Analytics identifiers for architecture gateway snapshots.

    Returns
    -------
    AnalyticsSamples
        Sample identifiers loaded from the architecture gateway fixture.
    """
    return load_analytics_samples(architecture_gateway)


__all__ = ["architecture_samples", "make_http_app"]


@pytest.fixture
def architecture_http_app(
    architecture_gateway: StorageGateway,
    make_http_app: Callable[..., FastAPI],
) -> FastAPI:
    """FastAPI app bound to the seeded architecture gateway.

    Returns
    -------
    FastAPI
        Application wired to the architecture gateway snapshot.
    """
    return make_http_app(
        gateway=architecture_gateway,
        snapshot=("demo/repo", "deadbeef"),
    )


@pytest.fixture
def architecture_http_client(architecture_http_app: FastAPI) -> Iterator[TestClient]:
    """Test client bound to the architecture FastAPI app.

    Yields
    ------
    TestClient
        Client bound to the architecture HTTP application.
    """
    with TestClient(architecture_http_app) as client:
        yield client


@pytest.fixture
def provisioned_http_app(
    provisioned_repo: ProvisionedGateway,
    make_http_app: Callable[..., FastAPI],
) -> FastAPI:
    """FastAPI app bound to the provisioned gateway snapshot.

    Returns
    -------
    FastAPI
        Application configured against the provisioned gateway snapshot.
    """
    return make_http_app(
        gateway=provisioned_repo.gateway,
        snapshot=(provisioned_repo.repo, provisioned_repo.commit),
        limits=BackendLimits(default_limit=10, max_rows_per_call=100),
    )


@pytest.fixture
def provisioned_http_client(provisioned_http_app: FastAPI) -> Iterator[TestClient]:
    """Test client bound to the provisioned FastAPI app.

    Yields
    ------
    TestClient
        Client bound to the provisioned FastAPI application.
    """
    with TestClient(provisioned_http_app) as client:
        yield client


__all__ += [
    "architecture_http_app",
    "architecture_http_client",
    "provisioned_http_app",
    "provisioned_http_client",
]


def _provisioned_backend_source(
    provisioned_repo: ProvisionedGateway,
) -> tuple[StorageGateway, tuple[str, str]]:
    return provisioned_repo.gateway, (provisioned_repo.repo, provisioned_repo.commit)


@pytest.fixture
def datasets_route_app(provisioned_repo: ProvisionedGateway) -> RouteApp:
    """Route-scoped app for dataset endpoints.

    Returns
    -------
    RouteApp
        Application wrapper exposing dataset routes and client factory.
    """
    return service_app_factory_with_routes(
        route_builders=[adapt_route(build_datasets_router)],
        backend_source=_provisioned_backend_source(provisioned_repo),
    )


@pytest.fixture
def datasets_http_client(datasets_route_app: RouteApp) -> Iterator[TestClient]:
    """Client bound to dataset routes.

    Yields
    ------
    TestClient
        Client configured for dataset routes.
    """
    with datasets_route_app.client() as client:
        yield client


@pytest.fixture
def functions_route_app(provisioned_repo: ProvisionedGateway) -> RouteApp:
    """Route-scoped app for function endpoints.

    Returns
    -------
    RouteApp
        Application wrapper exposing function routes and client factory.
    """
    return service_app_factory_with_routes(
        route_builders=[adapt_route(build_functions_router)],
        backend_source=_provisioned_backend_source(provisioned_repo),
    )


@pytest.fixture
def functions_http_client(functions_route_app: RouteApp) -> Iterator[TestClient]:
    """Client bound to function routes.

    Yields
    ------
    TestClient
        Client configured for function routes.
    """
    with functions_route_app.client() as client:
        yield client


@pytest.fixture
def meta_route_app(provisioned_repo: ProvisionedGateway) -> RouteApp:
    """Route-scoped app for meta endpoints.

    Returns
    -------
    RouteApp
        Application wrapper exposing meta routes and client factory.
    """
    return service_app_factory_with_routes(
        route_builders=[adapt_route(build_meta_router)],
        backend_source=_provisioned_backend_source(provisioned_repo),
    )


@pytest.fixture
def meta_http_client(meta_route_app: RouteApp) -> Iterator[TestClient]:
    """Client bound to meta routes.

    Yields
    ------
    TestClient
        Client configured for meta routes.
    """
    with meta_route_app.client() as client:
        yield client


@pytest.fixture
def health_route_app(provisioned_repo: ProvisionedGateway) -> RouteApp:
    """Route-scoped app for health endpoints.

    Returns
    -------
    RouteApp
        Application wrapper exposing health routes and client factory.
    """
    return service_app_factory_with_routes(
        route_builders=[adapt_route(build_health_router)],
        backend_source=_provisioned_backend_source(provisioned_repo),
    )


@pytest.fixture
def health_http_client(health_route_app: RouteApp) -> Iterator[TestClient]:
    """Client bound to health routes.

    Yields
    ------
    TestClient
        Client configured for health routes.
    """
    with health_route_app.client() as client:
        yield client


@pytest.fixture
def architecture_route_app(architecture_gateway: StorageGateway) -> RouteApp:
    """Route-scoped app for architecture/subsystem endpoints.

    Returns
    -------
    RouteApp
        Application wrapper exposing architecture and subsystem routes.
    """
    backend_source = (architecture_gateway, ("demo/repo", "deadbeef"))
    return service_app_factory_with_routes(
        route_builders=[
            adapt_route(build_architecture_router),
            adapt_route(build_subsystem_router),
        ],
        backend_source=backend_source,
    )


@pytest.fixture
def architecture_route_client(architecture_route_app: RouteApp) -> Iterator[TestClient]:
    """Client bound to architecture and subsystem routes.

    Yields
    ------
    TestClient
        Client configured for architecture and subsystem routes.
    """
    with architecture_route_app.client() as client:
        yield client


__all__ += [
    "architecture_route_app",
    "architecture_route_client",
    "datasets_http_client",
    "datasets_route_app",
    "functions_http_client",
    "functions_route_app",
    "health_http_client",
    "health_route_app",
    "meta_http_client",
    "meta_route_app",
]
