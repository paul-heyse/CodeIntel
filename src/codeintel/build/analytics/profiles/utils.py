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
    from codeintel.storage.gateway import StorageGateway

CATALOG_MODULE_TABLE = "temp.catalog_modules"
DEFAULT_MODULE_TABLE = "core.modules"
log = logging.getLogger(__name__)

__all__ = [
    "CATALOG_MODULE_TABLE",
    "DEFAULT_MODULE_TABLE",
    "seed_catalog_modules",
]


def seed_catalog_modules(
    gateway: StorageGateway,
    catalog_provider: FunctionCatalogProvider | None,
    repo: str,
    commit: str,
    *,
    module_map_override: Mapping[str, str] | None = None,
) -> str:
    """Create or refresh a temp module mapping table from a catalog provider.

    Returns the table name that should be used for module lookups. When neither
    a provider nor override data are available, this falls back to the default
    ``core.modules`` table.

    Parameters
    ----------
    gateway
        Storage gateway for database access.
    catalog_provider
        Optional catalog provider for module mappings.
    repo
        Repository identifier.
    commit
        Commit identifier.
    module_map_override
        Optional explicit module mappings to use instead of catalog.

    Returns
    -------
    str
        Table name to use for module lookups.
    """
    if catalog_provider is None and module_map_override is None:
        return DEFAULT_MODULE_TABLE

    module_by_path = (
        module_map_override
        if module_map_override is not None
        else catalog_provider.catalog().module_by_path
        if catalog_provider is not None
        else {}
    )
    if not module_by_path:
        return DEFAULT_MODULE_TABLE

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

    frame = pl.from_dicts(rows)

    gateway.execute(f"DROP TABLE IF EXISTS {CATALOG_MODULE_TABLE}")
    temp_name = "catalog_modules_seed"
    gateway.register(temp_name, frame)
    try:
        gateway.execute(
            f"CREATE OR REPLACE TEMP TABLE catalog_modules AS SELECT * FROM {temp_name}"
        )
    finally:
        gateway.unregister(temp_name)
    return CATALOG_MODULE_TABLE
