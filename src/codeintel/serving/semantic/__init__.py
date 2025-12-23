"""Semantic serving primitives (registry, inventory, query kernel)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from codeintel.core.imports.lazy import lazy_import

__all__ = [
    "FilterSpec",
    "RegistryService",
    "SemanticQueryKernel",
    "SemanticQueryRequest",
    "SemanticQueryResponse",
    "SemanticViewSpec",
]

if TYPE_CHECKING:
    from codeintel.core.registry import RegistryService
    from codeintel.serving.semantic.kernel import SemanticQueryKernel
    from codeintel.serving.semantic.models import (
        FilterSpec,
        SemanticQueryRequest,
        SemanticQueryResponse,
        SemanticViewSpec,
    )

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "FilterSpec": ("codeintel.serving.semantic.models", "FilterSpec"),
    "RegistryService": ("codeintel.core.registry", "RegistryService"),
    "SemanticQueryKernel": ("codeintel.serving.semantic.kernel", "SemanticQueryKernel"),
    "SemanticQueryRequest": ("codeintel.serving.semantic.models", "SemanticQueryRequest"),
    "SemanticQueryResponse": ("codeintel.serving.semantic.models", "SemanticQueryResponse"),
    "SemanticViewSpec": ("codeintel.serving.semantic.models", "SemanticViewSpec"),
}


def __getattr__(name: str) -> object:
    """Lazily import semantic symbols to avoid import-time cycles.

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
        module = lazy_import(module_name)
        value = getattr(module, attr_name)
        globals()[name] = value
        return value
    message = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(message)
