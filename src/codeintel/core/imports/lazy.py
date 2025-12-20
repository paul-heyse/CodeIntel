"""Lazy import utilities with caching."""

from __future__ import annotations

import importlib
from functools import lru_cache
from types import ModuleType

__all__ = ["lazy_getattr", "lazy_import"]


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
