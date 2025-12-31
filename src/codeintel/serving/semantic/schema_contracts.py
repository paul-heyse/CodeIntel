"""DuckDB-backed schema resolution for serving engines."""

from __future__ import annotations

import pyarrow as pa

from codeintel.core.columnar.schema_metadata import (
    decode_metadata,
    merge_field_metadata,
    merge_metadata,
)
from codeintel.storage.duckdb_types import (
    DuckDBCatalogException,
    DuckDBConnection,
    DuckDBRelation,
)
from codeintel.storage.schema.arrow_schema import arrow_schema_for_table_key


def contract_schema_for_table_key(
    *,
    con: DuckDBConnection | None,
    table_key: str,
    repo: str | None = None,
    commit: str | None = None,
) -> pa.Schema | None:
    """Return a contract schema for a table key from DuckDB.

    Returns
    -------
    pa.Schema | None
        The resolved contract schema, or None when unavailable.
    """
    if con is None:
        return None
    relation_schema = _relation_schema_for_table(con, table_key=table_key)
    if relation_schema is None:
        return None
    metadata_schema = arrow_schema_for_table_key(
        con,
        table_key=table_key,
        repo=repo,
        commit=commit,
    )
    if metadata_schema is None:
        return relation_schema
    return _merge_schema_metadata(relation_schema, metadata_schema)


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


__all__ = ["contract_schema_for_table_key"]
