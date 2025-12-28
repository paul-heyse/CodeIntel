"""Arrow schema rendering helpers for storage boundaries."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

import pyarrow as pa

from codeintel.core.schemas.arrow_gen import (
    ArrowSchemaMetadata,
    ArrowSchemaProvenance,
    arrow_schema_from_table_schema,
)
from codeintel.storage.contracts.schema_provider import get_schema_provider
from codeintel.storage.metadata import load_derived_lineage_columns
from codeintel.storage.metadata.meta_catalog import meta_table_ref

if TYPE_CHECKING:
    from duckdb import DuckDBPyConnection


def _load_registry_metadata(con: DuckDBPyConnection, table_key: str) -> dict[str, object]:
    registry_ref = meta_table_ref("metadata.table_schema_registry")
    row = con.execute(
        f"""
        SELECT schema_digest,
               schema_hash,
               derivation_kind,
               derivation_source,
               inference_status,
               inference_error
        FROM {registry_ref}
        WHERE table_key = ?
        """,
        [table_key],
    ).fetchone()
    if row is None:
        return {}
    return {
        "schema_digest": row[0],
        "schema_hash": row[1],
        "derivation_kind": row[2],
        "derivation_source": row[3],
        "inference_status": row[4],
        "inference_error": row[5],
    }


def arrow_schema_for_table_key(
    con: DuckDBPyConnection,
    *,
    table_key: str,
    repo: str | None = None,
    commit: str | None = None,
    pii_by_column: Mapping[str, str] | None = None,
) -> pa.Schema | None:
    """Render a PyArrow schema enriched with metadata for a table key.

    Parameters
    ----------
    con
        DuckDB connection used to load registry metadata and lineage.
    table_key
        Fully qualified table key (schema.table).
    repo
        Optional repository identifier for lineage lookups.
    commit
        Optional commit hash for lineage lookups.
    pii_by_column
        Optional mapping of column name to PII classification labels.

    Returns
    -------
    pa.Schema | None
        Rendered PyArrow schema with metadata, or None if the table is unknown.
    """
    table_schema = get_schema_provider().get_table_schema(table_key)
    if table_schema is None:
        return None

    registry_metadata = _load_registry_metadata(con, table_key)
    schema_digest = _normalize_str(registry_metadata.get("schema_digest"))
    schema_hash_value = _normalize_str(registry_metadata.get("schema_hash"))
    derivation_kind = _normalize_str(registry_metadata.get("derivation_kind"))
    derivation_source = _normalize_str(registry_metadata.get("derivation_source"))
    inference_status = _normalize_str(registry_metadata.get("inference_status"))
    inference_error = _normalize_str(registry_metadata.get("inference_error"))

    column_lineage = None
    if repo and commit:
        column_lineage = load_derived_lineage_columns(
            con,
            repo=repo,
            commit=commit,
            downstream_table=table_key,
        )

    provenance = ArrowSchemaProvenance(
        derivation_kind=derivation_kind,
        derivation_source=derivation_source,
        inference_status=inference_status,
        inference_error=inference_error,
    )
    metadata = ArrowSchemaMetadata(
        schema_hash=schema_hash_value,
        schema_digest=schema_digest,
        provenance=provenance,
        column_lineage=column_lineage,
        pii_by_column=pii_by_column,
    )

    return arrow_schema_from_table_schema(table_schema=table_schema, metadata=metadata)


def _normalize_str(value: object | None) -> str | None:
    if value is None:
        return None
    return str(value)


def arrow_schema_hash(schema: pa.Schema) -> str | None:
    """Return the CodeIntel schema hash embedded in Arrow metadata.

    Parameters
    ----------
    schema
        PyArrow schema to inspect.

    Returns
    -------
    str | None
        Schema hash when present, otherwise None.
    """
    return _schema_metadata_value(schema, "codeintel.schema_hash")


def arrow_schema_digest(schema: pa.Schema) -> str | None:
    """Return the schema digest embedded in Arrow metadata.

    Parameters
    ----------
    schema
        PyArrow schema to inspect.

    Returns
    -------
    str | None
        Schema digest when present, otherwise None.
    """
    return _schema_metadata_value(schema, "codeintel.schema_digest")


def _schema_metadata_value(schema: pa.Schema, key: str) -> str | None:
    metadata = schema.metadata
    if not metadata:
        return None
    raw = metadata.get(key.encode("utf-8"))
    if raw is None:
        return None
    return raw.decode("utf-8")


__all__ = [
    "arrow_schema_digest",
    "arrow_schema_for_table_key",
    "arrow_schema_hash",
]
