"""Shared application services for CodeIntel surfaces."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

__all__ = ["BackendResource", "HttpQueryService", "LocalQueryService", "build_backend_resource"]

if TYPE_CHECKING:
    BackendResource: object
    HttpQueryService: object
    LocalQueryService: object
    build_backend_resource: object

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "HttpQueryService": ("codeintel.serving.services.query_service", "HttpQueryService"),
    "LocalQueryService": ("codeintel.serving.services.query_service", "LocalQueryService"),
    "BackendResource": ("codeintel.serving.services.wiring", "BackendResource"),
    "build_backend_resource": ("codeintel.serving.services.wiring", "build_backend_resource"),
}


def __getattr__(name: str) -> object:
    """
    Lazily import service attributes to avoid circular imports during initialization.

    Returns
    -------
    object
        Requested attribute resolved from its defining module.

    Raises
    ------
    AttributeError
        When the attribute is not registered for lazy loading.
    """
    if name in _LAZY_IMPORTS:
        module_name, attr_name = _LAZY_IMPORTS[name]
        module = import_module(module_name)
        value = getattr(module, attr_name)
        globals()[name] = value
        return value
    message = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(message)
