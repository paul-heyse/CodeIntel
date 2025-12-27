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

    from hamilton.driver import Driver

    from codeintel.core.hamilton.tag_query import TagQuery


@lru_cache(maxsize=1)
def _view_modules() -> tuple[ModuleType, ...]:
    """Return the canonical module set to scan for view builders.

    Returns
    -------
    tuple[ModuleType, ...]
        Modules containing view-builder node functions.
    """
    return (importlib.import_module("codeintel.storage.views.ibis_views"),)


def view_builder_modules() -> tuple[ModuleType, ...]:
    """Return the canonical module set to scan for view builders.

    Returns
    -------
    tuple[ModuleType, ...]
        Modules containing view-builder node functions.
    """
    return _view_modules()


def discover_view_table_keys(
    *,
    dr: Driver | None = None,
    tag_query: TagQuery | None = None,
) -> tuple[str, ...]:
    """Discover all view table keys exposed by view-builder modules.

    Returns
    -------
    tuple[str, ...]
        Discovered view table keys, sorted deterministically.
    """
    discovered = discover_view_builders(
        dr=dr,
        tag_query=tag_query,
        modules=view_builder_modules(),
    )
    keys = {d.table_key for d in discovered}
    return tuple(sorted(keys))


def discover_derived_docs_views(
    *,
    dr: Driver | None = None,
    tag_query: TagQuery | None = None,
) -> tuple[str, ...]:
    """Discover docs.* views defined by view builders.

    Returns
    -------
    tuple[str, ...]
        Discovered docs view keys, sorted deterministically.
    """
    keys = discover_view_table_keys(dr=dr, tag_query=tag_query)
    return tuple(key for key in keys if key.startswith("docs.v_"))


def clear_view_inventory_cache() -> None:
    """Clear cached view inventory (for testing)."""
    _view_modules.cache_clear()


__all__ = [
    "clear_view_inventory_cache",
    "discover_derived_docs_views",
    "discover_view_table_keys",
    "view_builder_modules",
]
