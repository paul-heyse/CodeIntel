"""Serving snapshot preparation helpers for build workflows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from codeintel.storage.backend import DuckDBSession
from codeintel.storage.constants import META_CATALOG_NAME
from codeintel.storage.duckdb_policy_backend import duckdb_default_catalog
from codeintel.storage.gateway.config import StorageConfig
from codeintel.storage.gateway.protocol import DuckDBError
from codeintel.storage.helpers.table_key import fully_qualified_table_ref
from codeintel.storage.serving.search_index import build_search_documents_table, ensure_fts_index

if TYPE_CHECKING:
    from codeintel.storage.gateway.protocol import DuckDBConnection


class ServingSnapshotError(RuntimeError):
    """Base error for serving snapshot preparation failures."""


class SearchIndexBuildError(ServingSnapshotError):
    """Raised when search index preparation fails."""


class LineageMetadataError(ServingSnapshotError):
    """Raised when lineage metadata is missing from the snapshot."""


def _table_exists(
    con: DuckDBConnection,
    *,
    schema: str,
    table: str,
    catalog: str | None = None,
) -> bool:
    if catalog is None:
        result = con.execute(
            """
            SELECT 1
            FROM information_schema.tables
            WHERE table_schema = ? AND table_name = ?
            LIMIT 1
            """,
            [schema, table],
        ).fetchone()
    else:
        result = con.execute(
            """
            SELECT 1
            FROM information_schema.tables
            WHERE table_catalog = ? AND table_schema = ? AND table_name = ?
            LIMIT 1
            """,
            [catalog, schema, table],
        ).fetchone()
    return result is not None


def _require_table(
    con: DuckDBConnection,
    *,
    schema: str,
    table: str,
    catalog: str | None = None,
) -> None:
    if _table_exists(con, schema=schema, table=table, catalog=catalog):
        return
    msg = f"Required table missing: {schema}.{table}"
    raise RuntimeError(msg)


def _require_search_documents(con: DuckDBConnection) -> None:
    _require_table(con, schema="docs", table="search_documents")
    search_documents_ref = fully_qualified_table_ref(
        "docs.search_documents",
        catalog=duckdb_default_catalog(con),
    )
    row = con.execute(f"SELECT COUNT(*) FROM {search_documents_ref}").fetchone()
    count = int(row[0]) if row is not None and row[0] is not None else 0
    if count <= 0:
        msg = "Search documents table is empty: docs.search_documents"
        raise RuntimeError(msg)


def _require_lineage_tables(con: DuckDBConnection) -> None:
    _require_table(
        con,
        schema="metadata",
        table="derived_lineage_edges",
        catalog=META_CATALOG_NAME,
    )
    _require_table(
        con,
        schema="metadata",
        table="derived_lineage_columns",
        catalog=META_CATALOG_NAME,
    )


@dataclass(frozen=True, slots=True)
class ServingSnapshotService:
    """Prepare serving snapshot databases for publish workflows."""

    def prepare_snapshot(self, *, db_path: Path) -> None:
        """Build serving snapshot tables and validate prerequisites."""
        config = StorageConfig(
            db_path=db_path,
            read_only=False,
            apply_schema=False,
            ensure_views=False,
            validate_schema=False,
        )
        session = DuckDBSession(config)
        con = session.open()
        try:
            self._build_search_index(con)
            self._validate_lineage(con)
        finally:
            con.commit()
            con.close()

    @staticmethod
    def _build_search_index(con: DuckDBConnection) -> None:
        try:
            build_search_documents_table(con)
            ensure_fts_index(con)
            _require_search_documents(con)
        except (OSError, ValueError, DuckDBError, RuntimeError) as exc:
            msg = "Search index build failed"
            raise SearchIndexBuildError(msg) from exc

    @staticmethod
    def _validate_lineage(con: DuckDBConnection) -> None:
        try:
            _require_lineage_tables(con)
        except RuntimeError as exc:
            msg = "Lineage metadata missing"
            raise LineageMetadataError(msg) from exc


__all__ = [
    "LineageMetadataError",
    "SearchIndexBuildError",
    "ServingSnapshotError",
    "ServingSnapshotService",
]
