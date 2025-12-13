"""Shared helpers for analytics profile recipes."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Mapping

    from codeintel.graphs.catalog import FunctionCatalogProvider
    from codeintel.storage.gateway import StorageGateway

CATALOG_MODULE_TABLE = "temp.catalog_modules"
DEFAULT_MODULE_TABLE = "core.modules"
log = logging.getLogger(__name__)


def seed_catalog_modules(
    gateway: StorageGateway,
    catalog_provider: FunctionCatalogProvider | None,
    repo: str,
    commit: str,
    *,
    module_map_override: Mapping[str, str] | None = None,
) -> str:
    """
    Create or refresh a temp module mapping table from a catalog provider.

    Returns the table name that should be used for module lookups. When neither
    a provider nor override data are available, this falls back to the default
    ``core.modules`` table.

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
            "tags": "[]",
            "owners": "[]",
        }
        for path, module in module_by_path.items()
    ]

    df = pd.DataFrame(rows)

    gateway.execute(f"DROP TABLE IF EXISTS {CATALOG_MODULE_TABLE}")

    gateway.ibis.con.create_table(
        CATALOG_MODULE_TABLE.rsplit(".", maxsplit=1)[-1], df, temp=True, overwrite=True
    )
    return CATALOG_MODULE_TABLE


def optional_str(value: object | None) -> str | None:
    """
    Return a string representation or ``None``.

    Returns
    -------
    str | None
        Converted string or ``None`` when input is missing.
    """
    return str(value) if value is not None else None


def optional_int(value: object | None) -> int | None:
    """
    Return an integer or ``None`` when value is not provided.

    Returns
    -------
    int | None
        Converted integer or ``None`` when input is missing.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str) and value.strip():
        try:
            return int(value)
        except ValueError:
            return None
    return None


def int_or_default(value: object | None, default: int = 0) -> int:
    """
    Return an integer, falling back to ``default`` when value is falsy.

    Returns
    -------
    int
        Integer value or ``default`` when empty.
    """
    converted = optional_int(value)
    return converted if converted is not None else default


def optional_float(value: object | None) -> float | None:
    """
    Return a float or ``None`` when value is not provided.

    Returns
    -------
    float | None
        Converted float or ``None`` when input is missing.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return float(int(value))
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str) and value.strip():
        try:
            return float(value)
        except ValueError:
            return None
    return None


def optional_bool(value: object | None) -> bool | None:
    """
    Return a boolean or ``None`` when value is not provided.

    Returns
    -------
    bool | None
        Converted boolean or ``None`` when input is missing.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes"}:
            return True
        if lowered in {"false", "0", "no"}:
            return False
    return None
