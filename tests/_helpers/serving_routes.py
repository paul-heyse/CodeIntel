"""Helpers for building route-scoped FastAPI apps in tests."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, FastAPI
from fastapi.testclient import TestClient

from codeintel.config.serving_models import ServingConfig
from codeintel.serving.backend import BackendLimits
from codeintel.serving.http.fastapi import BackendResource, create_app
from codeintel.serving.http.routes.functions import RouterOptions
from tests._helpers.serving_apps import ServiceApp, build_service_app

if TYPE_CHECKING:
    from codeintel.serving.services.observability import ServiceObservability
    from codeintel.storage.gateway import StorageGateway
    from tests.serving.mcp.conftest import McpBackendComponents


@dataclass(frozen=True)
class RouteApp:
    """Bundled FastAPI app and TestClient factory for route-focused tests."""

    app: FastAPI
    client_factory: Callable[[], Iterator[TestClient]]

    @contextmanager
    def client(self) -> Iterator[TestClient]:
        """Provide a TestClient bound to the configured app.

        Yields
        ------
        Iterator[TestClient]
            Client bound to the configured application.
        """
        with self.client_factory() as client:
            yield client


@dataclass(frozen=True)
class RouteAppOptions:
    """Configuration for building route-scoped apps."""

    limits: BackendLimits | None = None
    observability: ServiceObservability | None = None
    auto_pipeline: bool = False
    config_overrides: dict[str, Any] | None = None
    router_options: RouterOptions | None = None


def _config_loader(repo: str, commit: str) -> Callable[[], ServingConfig]:
    return _config_loader_with_overrides(repo=repo, commit=commit, config_overrides=None)


def _config_loader_with_overrides(
    *,
    repo: str,
    commit: str,
    config_overrides: dict[str, Any] | None,
) -> Callable[[], ServingConfig]:
    def _loader() -> ServingConfig:
        cfg_kwargs: dict[str, Any] = {
            "mode": "remote_api",
            "repo": repo,
            "commit": commit,
            "api_base_url": "http://test",
        }
        if config_overrides:
            cfg_kwargs.update(config_overrides)
        return ServingConfig(**cfg_kwargs)

    return _loader


def _backend_resource_from_service_app(service_app: ServiceApp) -> BackendResource:
    return BackendResource(
        backend=service_app.backend,
        service=service_app.service,
        close=lambda: None,
    )


def service_app_factory_with_routes(
    *,
    route_builders: Iterable[Callable[[RouterOptions | None], APIRouter]],
    backend_source: McpBackendComponents | tuple[StorageGateway, tuple[str, str]],
    options: RouteAppOptions | None = None,
) -> RouteApp:
    """Build a FastAPI app registering only the provided route builders.

    Parameters
    ----------
    route_builders
        Iterable of router factory callables (e.g., build_functions_router).
    backend_source
        Either prebuilt MCP components or a ``(gateway, snapshot)`` tuple.
    options
        Optional configuration for limits, observability, auto-pipeline, router options,
        and ServingConfig overrides.

    Returns
    -------
    RouteApp
        Configured app with a client context manager.
    """
    opts = options or RouteAppOptions()
    repo: str
    commit: str
    backend_resource: BackendResource
    try:
        gateway, snapshot = backend_source  # type: ignore[misc]
        repo, commit = snapshot  # type: ignore[misc]
        service_app = build_service_app(
            gateway,
            snapshot=snapshot,  # type: ignore[arg-type]
            limits=opts.limits,
            observability=opts.observability,
        )
        backend_resource = _backend_resource_from_service_app(service_app)
        repo, commit = service_app.repo, service_app.commit
    except (TypeError, ValueError):
        components = backend_source  # type: ignore[assignment]
        service = components.service
        if opts.observability is not None:
            service.observability = opts.observability
        backend_resource = BackendResource(
            backend=components.backend,
            service=service,
            close=lambda: None,
        )
        repo, commit = components.repo, components.commit

    router_options = opts.router_options or (
        RouterOptions(auto_pipeline=opts.auto_pipeline) if opts.auto_pipeline else None
    )

    def backend_factory(_cfg: ServingConfig, **_kwargs: object) -> BackendResource:
        return backend_resource

    app = create_app(
        config_loader=_config_loader_with_overrides(
            repo=repo,
            commit=commit,
            config_overrides=opts.config_overrides,
        ),
        backend_factory=backend_factory,
        auto_pipeline=opts.auto_pipeline,
    )

    for builder in route_builders:
        try:
            router = builder(router_options)
        except TypeError:
            router = builder()
        app.include_router(router)

    @contextmanager
    def _client() -> Iterator[TestClient]:
        with TestClient(app) as client:
            yield client

    return RouteApp(app=app, client_factory=_client)


__all__ = ["RouteApp", "RouteAppOptions", "service_app_factory_with_routes"]
