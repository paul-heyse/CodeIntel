"""Storage service for serving snapshot preparation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import pyarrow as pa
from sqlglot import exp

from codeintel.core.columnar.schema_alignment import (
    align_reader_to_contract,
    extras_policy_from_schema,
)
from codeintel.core.manifests import ArrowDatasetManifest, ServingSnapshotManifest
from codeintel.storage.backend import DuckDBSession
from codeintel.storage.constants import DEFAULT_ARROW_BATCH_SIZE, META_CATALOG_NAME
from codeintel.storage.datasets.manifest_index import dataset_for_manifest
from codeintel.storage.datasets.manifests import read_dataset_manifest
from codeintel.storage.datasets.scanning import DatasetScanOptions, build_scanner
from codeintel.storage.gateway.config import StorageConfig
from codeintel.storage.gateway.minimal import MinimalStorageGateway
from codeintel.storage.gateway.protocol import DuckDBError
from codeintel.storage.helpers.table_key import split_table_key
from codeintel.storage.schema.duckdb_contracts import contract_schema_for_table_key
from codeintel.storage.serving.search_index import build_search_documents_table, ensure_fts_index
from codeintel.storage.sqlglot_tools import render_sql_duckdb, table_expr_from_ref

if TYPE_CHECKING:
    from codeintel.storage.gateway.protocol import DuckDBConnection, DuckDBRelation


DEFAULT_FRAGMENT_READAHEAD = 2


@dataclass(frozen=True, slots=True)
class DatasetViewRequest:
    """Dataset manifest context for view registration."""

    table_key: str
    schema: str
    table: str
    manifest: ArrowDatasetManifest
    manifest_path: Path


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
    info_table = table_expr_from_ref("information_schema.tables")
    conditions: list[exp.Expression] = []
    params: list[object] = []
    if catalog is not None:
        conditions.append(exp.EQ(this=exp.column("table_catalog"), expression=exp.Placeholder()))
        params.append(catalog)
    conditions.extend(
        [
            exp.EQ(this=exp.column("table_schema"), expression=exp.Placeholder()),
            exp.EQ(this=exp.column("table_name"), expression=exp.Placeholder()),
        ]
    )
    params.extend([schema, table])
    where_expr = conditions[0]
    for condition in conditions[1:]:
        where_expr = exp.and_(where_expr, condition)
    query = exp.select(exp.Literal.number(1)).from_(info_table).where(where_expr).limit(1)
    result = con.execute(render_sql_duckdb(query), params).fetchone()
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
    relation = MinimalStorageGateway(con).relation_from_table_key("docs.search_documents")
    row = relation.count("*").fetchone()
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
            _validate_dataset_entries(snapshot_manifest)
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
            request = DatasetViewRequest(
                table_key=table_key,
                schema=schema,
                table=table,
                manifest=manifest,
                manifest_path=manifest_path,
            )
            _create_dataset_view(
                con=con,
                request=request,
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


def _validate_dataset_entries(snapshot_manifest: ServingSnapshotManifest) -> None:
    schema_hashes = _load_schema_hashes(Path(snapshot_manifest.schema_manifest_path))
    for table_key, entry in snapshot_manifest.datasets.items():
        manifest_path = Path(entry.manifest_path)
        manifest = read_dataset_manifest(manifest_path)
        if manifest.table_key != table_key:
            msg = f"Dataset manifest table_key mismatch: {table_key} != {manifest.table_key}"
            raise ValueError(msg)
        if entry.schema_hash is None:
            msg = f"Dataset schema hash missing for {table_key}"
            raise ValueError(msg)
        expected = schema_hashes.get(table_key)
        if expected is None:
            msg = f"Schema manifest missing hash for {table_key}"
            raise KeyError(msg)
        if entry.schema_hash != expected:
            msg = f"Dataset schema hash mismatch for {table_key}: {entry.schema_hash} != {expected}"
            raise ValueError(msg)
        if manifest.schema_hash is not None and manifest.schema_hash != entry.schema_hash:
            msg = (
                "Dataset manifest schema hash mismatch for "
                f"{table_key}: {manifest.schema_hash} != {entry.schema_hash}"
            )
            raise ValueError(msg)


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
    request: DatasetViewRequest,
) -> None:
    current_schema = _current_schema(con)
    if current_schema != request.schema:
        _set_schema(con, request.schema)
    try:
        contract_schema = contract_schema_for_table_key(
            con=con,
            table_key=request.table_key,
        )
        relation = _dataset_read_parquet_relation(
            con=con,
            manifest=request.manifest,
            manifest_path=request.manifest_path,
            contract_schema=contract_schema,
        )
        relation.create_view(request.table)
    finally:
        if current_schema != request.schema:
            _set_schema(con, current_schema)


def _dataset_read_parquet_relation(
    *,
    con: DuckDBConnection,
    manifest: ArrowDatasetManifest,
    manifest_path: Path,
    contract_schema: pa.Schema | None,
) -> DuckDBRelation:
    dataset = dataset_for_manifest(manifest=manifest, manifest_path=manifest_path)
    scan_options = DatasetScanOptions(
        batch_size=DEFAULT_ARROW_BATCH_SIZE,
        fragment_readahead=DEFAULT_FRAGMENT_READAHEAD,
        schema=contract_schema,
    )
    scanner = build_scanner(dataset, options=scan_options)
    if contract_schema is not None:
        reader = scanner.to_reader()
        aligned = align_reader_to_contract(
            reader,
            contract_schema,
            extras_policy=extras_policy_from_schema(contract_schema),
        )
        try:
            return con.from_arrow(aligned)
        except (TypeError, ValueError):
            return con.from_arrow(scanner)
    try:
        return con.from_arrow(scanner)
    except (TypeError, ValueError):
        reader = scanner.to_reader()
        try:
            return con.from_arrow(reader)
        except (TypeError, ValueError):
            return con.from_arrow(dataset)


def _current_schema(con: DuckDBConnection) -> str:
    query = exp.select(exp.Anonymous(this="current_schema"))
    row = con.execute(render_sql_duckdb(query)).fetchone()
    if row is None or row[0] is None:
        msg = "DuckDB returned empty current_schema()"
        raise RuntimeError(msg)
    return str(row[0])


def _set_schema(con: DuckDBConnection, schema: str) -> None:
    statement = exp.Set(
        expressions=[
            exp.EQ(
                this=exp.Var(this="schema"),
                expression=exp.Literal.string(schema),
            )
        ]
    )
    con.execute(render_sql_duckdb(statement))


__all__ = [
    "DatasetManifestError",
    "LineageMetadataError",
    "SearchIndexBuildError",
    "ServingSnapshotError",
    "ServingSnapshotService",
]
