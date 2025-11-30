"""FastAPI dependency injection helpers for CodeIntel serving."""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request

from codeintel.config.serving_models import ServingConfig
from codeintel.serving.mcp import errors
from codeintel.serving.mcp.backend import QueryBackend
from codeintel.serving.services.query_service import QueryService


def get_app_config(request: Request) -> ServingConfig:
    """
    Retrieve the validated application configuration from state.

    Raises
    ------
    errors.backend_failure
        If the configuration is missing.

    Returns
    -------
    ServingConfig
        Loaded server configuration.
    """
    config: ServingConfig | None = getattr(request.app.state, "config", None)
    if config is not None:
        return config
    message = "Server configuration is not initialized"
    raise errors.backend_failure(message)


def get_backend(request: Request) -> QueryBackend:
    """
    Retrieve the shared backend from state.

    Raises
    ------
    errors.backend_failure
        If the backend is missing.

    Returns
    -------
    QueryBackend
        Backend connected to the configured data source.
    """
    backend: QueryBackend | None = getattr(request.app.state, "backend", None)
    if backend is None:
        message = "Backend is not initialized"
        raise errors.backend_failure(message)
    return backend


def get_service(request: Request) -> QueryService:
    """
    Retrieve the shared query service from state.

    Raises
    ------
    errors.backend_failure
        If the service is missing.

    Returns
    -------
    QueryService
        Shared query service instance.
    """
    service: QueryService | None = getattr(request.app.state, "service", None)
    if service is None:
        backend: QueryBackend | None = getattr(request.app.state, "backend", None)
        service = getattr(backend, "service", None) if backend is not None else None
    if service is None:
        message = "Query service is not initialized"
        raise errors.backend_failure(message)
    return service


ConfigDep = Annotated[ServingConfig, Depends(get_app_config)]
BackendDep = Annotated[QueryBackend, Depends(get_backend)]
ServiceDep = Annotated[QueryService, Depends(get_service)]

__all__ = ["BackendDep", "ConfigDep", "ServiceDep", "get_app_config", "get_backend", "get_service"]
