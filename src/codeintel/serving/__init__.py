"""Serving surfaces exposing CodeIntel data via HTTP (FastAPI) and MCP protocol."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

__all__ = [
    "BackendResource",
    "DuckDBBackend",
    "HasModelDump",
    "HttpBackend",
    "HttpQueryService",
    "LocalQueryService",
    "QueryBackend",
    "QueryService",
    "build_backend_resource",
    "build_service_from_config",
]

if TYPE_CHECKING:
    BackendResource: object
    DuckDBBackend: object
    HasModelDump: object
    HttpBackend: object
    HttpQueryService: object
    LocalQueryService: object
    QueryBackend: object
    QueryService: object
    build_backend_resource: object
    build_service_from_config: object

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "DuckDBBackend": ("codeintel.serving.mcp.backend", "DuckDBBackend"),
    "HttpBackend": ("codeintel.serving.mcp.backend", "HttpBackend"),
    "QueryBackend": ("codeintel.serving.mcp.backend", "QueryBackend"),
    "HasModelDump": ("codeintel.serving.protocols", "HasModelDump"),
    "BackendResource": ("codeintel.serving.services.factory", "BackendResource"),
    "build_backend_resource": ("codeintel.serving.services.factory", "build_backend_resource"),
    "build_service_from_config": (
        "codeintel.serving.services.factory",
        "build_service_from_config",
    ),
    "HttpQueryService": ("codeintel.serving.services.query_service", "HttpQueryService"),
    "LocalQueryService": ("codeintel.serving.services.query_service", "LocalQueryService"),
    "QueryService": ("codeintel.serving.services.query_service", "QueryService"),
}


def __getattr__(name: str) -> object:
    """
    Lazily import serving attributes to avoid import-time circular dependencies.

    Returns
    -------
    object
        Requested attribute loaded from its defining module.

    Raises
    ------
    AttributeError
        If the requested attribute is not registered for lazy loading.
    """
    if name in _LAZY_IMPORTS:
        module_name, attr_name = _LAZY_IMPORTS[name]
        module = import_module(module_name)
        value = getattr(module, attr_name)
        globals()[name] = value
        return value
    message = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(message)
