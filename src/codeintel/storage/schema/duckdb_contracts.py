"""DuckDB-backed contract resolution helpers for storage and serving."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pyarrow as pa

from codeintel.core.columnar.schema_metadata import (
    decode_metadata,
    merge_field_metadata,
    merge_metadata,
)
from codeintel.core.schemas.arrow_gen import ArrowSchemaMetadata, arrow_schema_from_table_schema
from codeintel.core.schemas.type_mappings import normalize_table_schema_types
from codeintel.storage.duckdb_types import (
    DuckDBCatalogException,
    DuckDBConnection,
    DuckDBRelation,
)
from codeintel.storage.metadata import load_derived_lineage_columns
from codeintel.storage.tracking.schema_catalog import load_table_schema_from_connection

if TYPE_CHECKING:
    from collections.abc import Mapping

    from codeintel.core.schemas.primitives import TableSchema


def table_schema_for_table_key(
    *,
    con: DuckDBConnection | None,
    table_key: str,
) -> TableSchema | None:
    """Return the TableSchema for a table key using DuckDB metadata.

    Parameters
    ----------
    con
        DuckDB connection for metadata lookups.
    table_key
        Fully qualified table key.

    Returns
    -------
    TableSchema | None
        Resolved TableSchema or None when unavailable.
    """
    if con is None:
        return None
    schema = load_table_schema_from_connection(con, table_key=table_key)
    if schema is None:
        return None
    return normalize_table_schema_types(schema)


def contract_schema_for_table_key(
    *,
    con: DuckDBConnection | None,
    table_key: str,
    repo: str | None = None,
    commit: str | None = None,
    pii_by_column: Mapping[str, str] | None = None,
) -> pa.Schema | None:
    """Return a contract schema for a table key from DuckDB.

    Returns
    -------
    pa.Schema | None
        The resolved contract schema, or None when unavailable.
    """
    if con is None:
        return None
    metadata_schema = _metadata_schema_for_table(
        con,
        table_key=table_key,
        repo=repo,
        commit=commit,
        pii_by_column=pii_by_column,
    )
    relation_schema = _relation_schema_for_table(con, table_key=table_key)
    if relation_schema is None:
        return metadata_schema
    if metadata_schema is None:
        return relation_schema
    return _merge_schema_metadata(relation_schema, metadata_schema)


def _metadata_schema_for_table(
    con: DuckDBConnection,
    *,
    table_key: str,
    repo: str | None,
    commit: str | None,
    pii_by_column: Mapping[str, str] | None,
) -> pa.Schema | None:
    table_schema = load_table_schema_from_connection(con, table_key=table_key)
    if table_schema is None:
        return None
    table_schema = normalize_table_schema_types(table_schema)
    column_lineage = None
    if repo and commit:
        column_lineage = load_derived_lineage_columns(
            con,
            repo=repo,
            commit=commit,
            downstream_table=table_key,
        )
    metadata = ArrowSchemaMetadata(
        column_lineage=column_lineage,
        pii_by_column=pii_by_column,
    )
    return arrow_schema_from_table_schema(table_schema=table_schema, metadata=metadata)


def _relation_schema_for_table(
    con: DuckDBConnection,
    *,
    table_key: str,
) -> pa.Schema | None:
    try:
        relation = con.table(table_key)
    except DuckDBCatalogException:
        return None
    limited = _limit_relation(relation)
    reader = _fetch_arrow_reader(limited, batch_size=1)
    return reader.schema


def _merge_schema_metadata(base: pa.Schema, metadata_schema: pa.Schema) -> pa.Schema:
    schema_updates = decode_metadata(metadata_schema.metadata)
    merged_metadata = merge_metadata(base.metadata, schema_updates)
    metadata_fields = {field.name: field for field in metadata_schema}
    merged_fields: list[pa.Field] = []
    for field in base:
        metadata_field = metadata_fields.get(field.name)
        if metadata_field is None:
            merged_fields.append(field)
            continue
        updates = decode_metadata(metadata_field.metadata)
        merged_fields.append(merge_field_metadata(field, updates))
    return pa.schema(merged_fields, metadata=merged_metadata)


def _limit_relation(relation: DuckDBRelation) -> DuckDBRelation:
    limiter = getattr(relation, "limit", None)
    if not callable(limiter):
        return relation
    try:
        limited = limiter(0)
    except TypeError:
        return relation
    return limited if isinstance(limited, DuckDBRelation) else relation


def _fetch_arrow_reader(
    relation: DuckDBRelation,
    *,
    batch_size: int,
) -> pa.RecordBatchReader:
    fetcher = getattr(relation, "fetch_arrow_reader", None)
    if callable(fetcher):
        try:
            return fetcher(batch_size)
        except TypeError:
            return fetcher()
    return relation.fetch_record_batch(batch_size)


__all__ = ["contract_schema_for_table_key", "table_schema_for_table_key"]
