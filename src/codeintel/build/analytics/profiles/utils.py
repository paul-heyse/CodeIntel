"""Shared helpers for analytics profile recipes.

Note
----
For type coercion helpers (``int_or_default``, ``optional_int``, etc.),
import directly from ``codeintel.build.analytics.utilities.type_coercion``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from collections.abc import Mapping

    from codeintel.core.catalog import FunctionCatalogProvider

CATALOG_MODULE_TABLE = "temp.catalog_modules"
DEFAULT_MODULE_TABLE = "core.modules"
log = logging.getLogger(__name__)

__all__ = [
    "CATALOG_MODULE_TABLE",
    "DEFAULT_MODULE_TABLE",
    "seed_catalog_modules",
]


def seed_catalog_modules(
    catalog_provider: FunctionCatalogProvider | None,
    repo: str,
    commit: str,
    *,
    module_map_override: Mapping[str, str] | None = None,
) -> pl.DataFrame | None:
    """Build an in-memory module mapping frame from a catalog provider.

    Returns a DataFrame for module lookups. When neither a provider nor override
    data are available, this returns None to signal a fallback to core.modules.

    Returns
    -------
    pl.DataFrame | None
        Module mapping frame or None when unavailable.
    """
    if catalog_provider is None and module_map_override is None:
        return None

    module_by_path = (
        module_map_override
        if module_map_override is not None
        else catalog_provider.catalog().module_by_path
        if catalog_provider is not None
        else {}
    )
    if not module_by_path:
        return None

    rows = [
        {
            "path": path,
            "module": module,
            "repo": repo,
            "commit": commit,
            "language": "python",
            "tags": [],
            "owners": [],
        }
        for path, module in module_by_path.items()
    ]

    return pl.from_dicts(rows)
