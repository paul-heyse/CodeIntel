"""Shared HTTP client harness for route tests."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import contextmanager
from typing import TYPE_CHECKING

from fastapi import APIRouter

from codeintel.serving.http.routes.functions import RouterOptions
from tests._helpers.serving_routes import (
    service_app_factory_with_routes,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from fastapi.testclient import TestClient

    from codeintel.storage.gateway import StorageGateway
    from tests._helpers.serving_routes import (
        RouteAppOptions,
    )
    from tests.serving.mcp.conftest import McpBackendComponents


RouteBuilder = Callable[[RouterOptions | None], APIRouter]


def adapt_route(builder: Callable[[], APIRouter]) -> RouteBuilder:
    """Wrap a zero-arg router builder to accept optional RouterOptions.

    Parameters
    ----------
    builder
        Router factory that does not consume RouterOptions.

    Returns
    -------
    RouteBuilder
        Adapter that forwards to the zero-arg builder.
    """

    def _wrapped(_options: RouterOptions | None) -> APIRouter:
        return builder()

    return _wrapped


@contextmanager
def build_client(
    *,
    route_builders: Iterable[RouteBuilder],
    backend_source: McpBackendComponents | tuple[StorageGateway, tuple[str, str]],
    options: RouteAppOptions | None = None,
) -> Iterator[TestClient]:
    """Construct a RouteApp and yield a bound TestClient with consistent cleanup.

    Yields
    ------
    Iterator[TestClient]
        Client bound to the configured FastAPI app.
    """
    route_app = service_app_factory_with_routes(
        route_builders=route_builders,
        backend_source=backend_source,
        options=options,
    )
    with route_app.client() as client:
        yield client


__all__ = ["RouteBuilder", "adapt_route", "build_client"]
