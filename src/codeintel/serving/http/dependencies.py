"""FastAPI dependency injection helpers for CodeIntel serving."""

from __future__ import annotations

from collections.abc import Callable
from typing import Annotated

from fastapi import Depends, Request

from codeintel.config.serving_models import ServingConfig
from codeintel.serving.auto_pipeline import ensure_prereqs_for_http
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


def make_op_prereq_dependency(op_id: str) -> Callable[[ServingConfig, QueryBackend], None]:
    """Create a FastAPI dependency that runs operation prerequisites.

    This factory creates a dependency that can be attached to route handlers
    to automatically run pipeline prerequisites before the operation executes.

    Parameters
    ----------
    op_id
        Operation identifier to run prerequisites for.

    Returns
    -------
    Callable[[ServingConfig, QueryBackend], None]
        Dependency function suitable for use with FastAPI's Depends.

    Examples
    --------
    >>> @router.get("/function/summary", dependencies=[Depends(make_op_prereq_dependency("function.summary"))])
    ... def get_function_summary(...):
    ...     ...
    """

    def _run_prereqs(config: ConfigDep, backend: BackendDep) -> None:
        """Run operation prerequisites if auto-pipeline is enabled."""
        ensure_prereqs_for_http(op_id=op_id, config=config, backend=backend)

    return _run_prereqs


__all__ = [
    "BackendDep",
    "ConfigDep",
    "ServiceDep",
    "get_app_config",
    "get_backend",
    "get_service",
    "make_op_prereq_dependency",
]
