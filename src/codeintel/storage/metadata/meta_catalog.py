"""Helpers for attaching and referencing the meta catalog."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from codeintel.storage.constants import META_CATALOG_NAME, META_DB_FILENAME
from codeintel.storage.helpers.table_key import fully_qualified_table_ref

if TYPE_CHECKING:
    from duckdb import DuckDBPyConnection

    from codeintel.storage.gateway.config import StorageConfig

__all__ = [
    "attach_meta_database",
    "default_meta_db_path",
    "meta_table_ref",
    "resolve_meta_db_path",
]

log = logging.getLogger(__name__)


def default_meta_db_path(db_path: Path) -> Path:
    """Return the default meta database path for a primary database.

    Returns
    -------
    Path
        Resolved meta database path for the primary database.
    """
    if str(db_path) == ":memory:":
        return Path(":memory:")
    return db_path.with_name(META_DB_FILENAME)


def resolve_meta_db_path(config: StorageConfig) -> Path:
    """Resolve the meta database path from config.

    Returns
    -------
    Path
        Meta database path derived from configuration.
    """
    return config.meta_db_path or default_meta_db_path(config.db_path)


def meta_table_ref(table_key: str, *, catalog: str = META_CATALOG_NAME) -> str:
    """Return a catalog-qualified table reference for meta tables.

    Returns
    -------
    str
        Fully qualified table reference for the meta catalog.
    """
    return fully_qualified_table_ref(table_key, catalog=catalog)


def attach_meta_database(con: DuckDBPyConnection, *, config: StorageConfig) -> None:
    """Attach the meta database to a connection if configured.

    Parameters
    ----------
    con
        DuckDB connection to attach the meta catalog.
    config
        Storage configuration containing meta catalog settings.

    """
    if not config.attach_meta:
        return

    catalog = META_CATALOG_NAME
    if _catalog_attached(con, catalog):
        return

    meta_path = resolve_meta_db_path(config)
    escaped_alias = catalog.replace('"', '""')
    if str(meta_path) == ":memory:":
        con.execute(f"ATTACH DATABASE ':memory:' AS \"{escaped_alias}\"")
        return

    if config.read_only:
        if not meta_path.exists():
            log.warning("Meta database not found for read-only attach: %s", meta_path)
            return
        escaped_path = str(meta_path).replace("'", "''")
        con.execute(f"ATTACH DATABASE '{escaped_path}' AS \"{escaped_alias}\" (READ_ONLY)")
    else:
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        escaped_path = str(meta_path).replace("'", "''")
        con.execute(f"ATTACH DATABASE '{escaped_path}' AS \"{escaped_alias}\"")


def _catalog_attached(con: DuckDBPyConnection, catalog: str) -> bool:
    rows = con.execute("PRAGMA database_list").fetchall()
    return any(row[1] == catalog for row in rows)
