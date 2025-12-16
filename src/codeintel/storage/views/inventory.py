"""Deterministic view inventory helpers.

This module replaces legacy constant lists of docs views with Hamilton tag discovery over
canonical view-builder modules.
"""

from __future__ import annotations

import importlib
from functools import lru_cache
from typing import TYPE_CHECKING

from codeintel.storage.views.discovery import discover_view_builders

if TYPE_CHECKING:
    from types import ModuleType


@lru_cache(maxsize=1)
def _view_modules() -> tuple[ModuleType, ...]:
    """Return the canonical module set to scan for view builders.

    Returns
    -------
    tuple[ModuleType, ...]
        Modules containing view-builder node functions.
    """
    return (importlib.import_module("codeintel.storage.views.ibis_views"),)


@lru_cache(maxsize=1)
def discover_view_table_keys() -> tuple[str, ...]:
    """Discover all view table keys exposed by view-builder modules.

    Returns
    -------
    tuple[str, ...]
        Discovered view table keys, sorted deterministically.
    """
    discovered = discover_view_builders(modules=_view_modules(), config=None)
    keys = {d.table_key for d in discovered}
    return tuple(sorted(keys))


@lru_cache(maxsize=1)
def discover_derived_docs_views() -> tuple[str, ...]:
    """Discover docs.* views defined by view builders.

    Returns
    -------
    tuple[str, ...]
        Discovered docs view keys, sorted deterministically.
    """
    keys = discover_view_table_keys()
    return tuple(key for key in keys if key.startswith("docs.v_"))


def clear_view_inventory_cache() -> None:
    """Clear cached view inventory (for testing)."""
    discover_derived_docs_views.cache_clear()
    discover_view_table_keys.cache_clear()
    _view_modules.cache_clear()


__all__ = [
    "clear_view_inventory_cache",
    "discover_derived_docs_views",
    "discover_view_table_keys",
]
