"""Storage service for serving snapshot preparation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from codeintel.core.manifests import ArrowDatasetManifest, ServingSnapshotManifest
from codeintel.storage.backend import DuckDBSession
from codeintel.storage.constants import META_CATALOG_NAME
from codeintel.storage.datasets.manifests import read_dataset_manifest
from codeintel.storage.duckdb_policy_backend import duckdb_default_catalog
from codeintel.storage.gateway.config import StorageConfig
from codeintel.storage.gateway.minimal import MinimalStorageGateway
from codeintel.storage.gateway.protocol import DuckDBError
from codeintel.storage.helpers.table_key import fully_qualified_table_ref, split_table_key
from codeintel.storage.serving.search_index import build_search_documents_table, ensure_fts_index

if TYPE_CHECKING:
    from codeintel.storage.gateway.protocol import DuckDBConnection, DuckDBRelation


class ServingSnapshotError(RuntimeError):
    """Base error for serving snapshot preparation failures."""


class SearchIndexBuildError(ServingSnapshotError):
    """Raised when search index preparation fails."""


class LineageMetadataError(ServingSnapshotError):
    """Raised when lineage metadata is missing from the snapshot."""


class DatasetManifestError(ServingSnapshotError):
    """Raised when dataset manifests are missing or inconsistent."""


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

    def prepare_snapshot(
        self,
        *,
        db_path: Path,
        snapshot_manifest: ServingSnapshotManifest,
    ) -> None:
        """Build serving snapshot tables and validate prerequisites.

        Parameters
        ----------
        db_path
            Path to the snapshot DuckDB database.
        snapshot_manifest
            Snapshot manifest describing dataset and artifact metadata.
        """
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
            self._validate_datasets(snapshot_manifest)
            self._register_dataset_views(con, snapshot_manifest=snapshot_manifest)
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

    @staticmethod
    def _validate_datasets(snapshot_manifest: ServingSnapshotManifest) -> None:
        try:
            schema_hashes = _load_schema_hashes(Path(snapshot_manifest.schema_manifest_path))
            for table_key, entry in snapshot_manifest.datasets.items():
                manifest_path = Path(entry.manifest_path)
                manifest = read_dataset_manifest(manifest_path)
                if manifest.table_key != table_key:
                    msg = (
                        f"Dataset manifest table_key mismatch: {table_key} != {manifest.table_key}"
                    )
                    raise ValueError(msg)
                if entry.schema_hash is None:
                    msg = f"Dataset schema hash missing for {table_key}"
                    raise ValueError(msg)
                expected = schema_hashes.get(table_key)
                if expected is None:
                    msg = f"Schema manifest missing hash for {table_key}"
                    raise KeyError(msg)
                if entry.schema_hash != expected:
                    msg = (
                        "Dataset schema hash mismatch for "
                        f"{table_key}: {entry.schema_hash} != {expected}"
                    )
                    raise ValueError(msg)
                if manifest.schema_hash is not None and manifest.schema_hash != entry.schema_hash:
                    msg = (
                        "Dataset manifest schema hash mismatch for "
                        f"{table_key}: {manifest.schema_hash} != {entry.schema_hash}"
                    )
                    raise ValueError(msg)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            msg = "Dataset manifest validation failed"
            raise DatasetManifestError(msg) from exc

    @staticmethod
    def _register_dataset_views(
        con: DuckDBConnection,
        *,
        snapshot_manifest: ServingSnapshotManifest,
    ) -> None:
        if not snapshot_manifest.datasets:
            return
        backend = MinimalStorageGateway(con).policy
        for table_key, entry in snapshot_manifest.datasets.items():
            manifest_path = Path(entry.manifest_path)
            manifest = read_dataset_manifest(manifest_path)
            schema, table = split_table_key(table_key)
            backend.create_schema_if_not_exists(schema)
            _create_dataset_view(
                con=con,
                schema=schema,
                table=table,
                manifest=manifest,
                manifest_path=manifest_path,
            )


def _load_schema_hashes(schema_manifest_path: Path) -> dict[str, str]:
    payload = json.loads(schema_manifest_path.read_text(encoding="utf-8"))
    obj = _expect_dict(payload, ctx="schema_manifest")
    version = str(obj.get("version", "")).strip()
    if version != "v2":
        msg = f"Unsupported schema manifest version: {version or 'unknown'}"
        raise ValueError(msg)
    table_hashes = _extract_schema_hashes(obj.get("tables", []), ctx="tables")
    view_hashes = _extract_schema_hashes(obj.get("views", []), ctx="views")
    combined = dict(table_hashes)
    for table_key, schema_hash in view_hashes.items():
        if table_key in combined:
            msg = f"Duplicate schema hash entry for {table_key}"
            raise ValueError(msg)
        combined[table_key] = schema_hash
    return combined


def _extract_schema_hashes(items: object, *, ctx: str) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for idx, raw in enumerate(_expect_list(items, ctx=ctx)):
        entry = _expect_dict(raw, ctx=f"{ctx}[{idx}]")
        table_key = _table_key_from_manifest(entry, ctx=f"{ctx}[{idx}]")
        raw_hash = entry.get("schema_hash")
        if not isinstance(raw_hash, str) or not raw_hash:
            msg = f"schema_hash is required for {ctx}[{idx}]"
            raise TypeError(msg)
        if table_key in hashes:
            msg = f"Duplicate schema hash entry for {table_key}"
            raise ValueError(msg)
        hashes[table_key] = raw_hash
    return hashes


def _table_key_from_manifest(entry: dict[str, object], *, ctx: str) -> str:
    table_key = entry.get("table_key")
    if isinstance(table_key, str) and table_key:
        return table_key
    schema = entry.get("schema")
    name = entry.get("name")
    if not isinstance(schema, str) or not schema.strip():
        msg = f"schema is required for {ctx}"
        raise TypeError(msg)
    if not isinstance(name, str) or not name.strip():
        msg = f"name is required for {ctx}"
        raise TypeError(msg)
    return f"{schema}.{name}"


def _expect_dict(value: object, *, ctx: str) -> dict[str, object]:
    if not isinstance(value, dict):
        msg = f"Expected object for {ctx}"
        raise TypeError(msg)
    return value


def _expect_list(value: object, *, ctx: str) -> list[object]:
    if not isinstance(value, list):
        msg = f"Expected array for {ctx}"
        raise TypeError(msg)
    return value


def _create_dataset_view(
    *,
    con: DuckDBConnection,
    schema: str,
    table: str,
    manifest: ArrowDatasetManifest,
    manifest_path: Path,
) -> None:
    current_schema = _current_schema(con)
    if current_schema != schema:
        _set_schema(con, schema)
    try:
        relation = _dataset_read_parquet_relation(
            con=con,
            manifest=manifest,
            manifest_path=manifest_path,
        )
        relation.create_view(table)
    finally:
        if current_schema != schema:
            _set_schema(con, current_schema)


def _dataset_read_parquet_relation(
    *,
    con: DuckDBConnection,
    manifest: ArrowDatasetManifest,
    manifest_path: Path,
) -> DuckDBRelation:
    dataset_dir = manifest_path.parent.resolve()
    hive_partitioning = bool(manifest.partition_columns)
    if manifest.files:
        paths = [str(dataset_dir / path) for path in manifest.files]
        return con.read_parquet(paths, hive_partitioning=hive_partitioning, union_by_name=True)
    glob_path = str(dataset_dir / "**" / "*.parquet")
    return con.read_parquet(glob_path, hive_partitioning=hive_partitioning, union_by_name=True)


def _current_schema(con: DuckDBConnection) -> str:
    row = con.execute("SELECT current_schema()").fetchone()
    if row is None or row[0] is None:
        msg = "DuckDB returned empty current_schema()"
        raise RuntimeError(msg)
    return str(row[0])


def _set_schema(con: DuckDBConnection, schema: str) -> None:
    escaped = schema.replace("'", "''")
    con.execute(f"SET schema='{escaped}'")


__all__ = [
    "DatasetManifestError",
    "LineageMetadataError",
    "SearchIndexBuildError",
    "ServingSnapshotError",
    "ServingSnapshotService",
]
