"""Lazy import utilities with caching."""

from __future__ import annotations

import importlib
from collections.abc import Callable
from functools import lru_cache
from types import ModuleType
from typing import cast

LazyAttrMap = dict[str, tuple[str, str]]

__all__ = ["LazyAttrMap", "lazy_callable", "lazy_getattr", "lazy_import", "make_lazy_getattr"]


@lru_cache(maxsize=32)
def lazy_import(module_name: str) -> ModuleType:
    """Import a module lazily with LRU caching.

    Parameters
    ----------
    module_name
        Fully qualified module name to import.

    Returns
    -------
    ModuleType
        Imported module instance.
    """
    return importlib.import_module(module_name)


def lazy_getattr(module_name: str, attr_name: str) -> object:
    """Return an attribute from a lazily imported module.

    Parameters
    ----------
    module_name
        Fully qualified module name to import.
    attr_name
        Attribute name to resolve on the imported module.

    Returns
    -------
    object
        Attribute value from the module.
    """
    module = lazy_import(module_name)
    return getattr(module, attr_name)


def make_lazy_getattr(
    lazy_attrs: LazyAttrMap,
    module_name: str,
    *,
    cache_in_globals: dict[str, object] | None = None,
) -> Callable[[str], object]:
    """Create a __getattr__ function for lazy module attribute loading.

    Returns
    -------
    Callable[[str], object]
        __getattr__ implementation that loads attributes on demand.
    """

    def _getattr_impl(name: str) -> object:
        if name not in lazy_attrs:
            message = f"module {module_name!r} has no attribute {name!r}"
            raise AttributeError(message)
        module_path, attr_name = lazy_attrs[name]
        attr = lazy_getattr(module_path, attr_name)
        if cache_in_globals is not None:
            cache_in_globals[name] = attr
        return attr

    return _getattr_impl


def lazy_callable(
    lazy_attrs: LazyAttrMap,
    name: str,
) -> Callable[..., object]:
    """Create a lazy-loading callable wrapper for a module attribute.

    Returns
    -------
    Callable[..., object]
        Callable that loads the target attribute on first invocation.
    """

    def wrapper(*args: object, **kwargs: object) -> object:
        module_path, attr_name = lazy_attrs[name]
        func = cast("Callable[..., object]", lazy_getattr(module_path, attr_name))
        return func(*args, **kwargs)

    wrapper.__name__ = name
    wrapper.__qualname__ = name
    return wrapper
