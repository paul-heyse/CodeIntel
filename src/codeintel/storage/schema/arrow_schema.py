"""Arrow schema rendering helpers for storage boundaries."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pyarrow as pa

from codeintel.core.columnar.ipc import schema_from_ipc_payload
from codeintel.core.columnar.schema_metadata import merge_field_metadata
from codeintel.storage.helpers.json import decode_json_dict
from codeintel.storage.metadata import load_derived_lineage_columns
from codeintel.storage.metadata.meta_catalog import meta_table_ref

if TYPE_CHECKING:
    from duckdb import DuckDBPyConnection


def _load_contract_schema(
    con: DuckDBPyConnection,
    *,
    table_key: str,
    require_inferred: bool,
) -> pa.Schema | None:
    registry_ref = meta_table_ref("metadata.table_schema_registry")
    versions_ref = meta_table_ref("metadata.schema_versions")
    filter_clause = ""
    if require_inferred:
        filter_clause = (
            "AND ("
            "registry.inference_status IN ('inferred', 'override') "
            "OR registry.inference_status IS NULL "
            "OR registry.derivation_kind IN ('inferred_relation', 'view_inferred')"
            ")"
        )
    row = con.execute(
        f"""
        SELECT versions.renderer_cache
        FROM {registry_ref} AS registry
        JOIN {versions_ref} AS versions
          ON registry.schema_digest = versions.schema_digest
        WHERE registry.table_key = ?
        {filter_clause}
        """,
        [table_key],
    ).fetchone()
    if row is None or row[0] is None:
        return None
    renderer_cache = decode_json_dict(row[0])
    ipc_payload = renderer_cache.get("arrow_schema_ipc_b64")
    if not isinstance(ipc_payload, str):
        return None
    return schema_from_ipc_payload(ipc_payload)


def _load_observed_schema(
    con: DuckDBPyConnection,
    *,
    table_key: str,
) -> pa.Schema | None:
    observations_ref = meta_table_ref("metadata.schema_observations")
    registry_ref = meta_table_ref("metadata.table_schema_registry")
    row = con.execute(
        f"""
        SELECT arrow_schema_ipc_b64
        FROM {observations_ref} AS o
        JOIN {registry_ref} AS r
          ON r.table_key = o.table_key
        WHERE o.table_key = ?
          AND r.derivation_kind IN ('inferred_relation', 'view_inferred')
        ORDER BY o.observed_at DESC
        LIMIT 1
        """,
        [table_key],
    ).fetchone()
    if row is None or row[0] is None:
        return None
    if not isinstance(row[0], str):
        return None
    return schema_from_ipc_payload(row[0])


def _apply_runtime_metadata(
    schema: pa.Schema,
    *,
    column_lineage: Mapping[str, list[tuple[str, str]]] | None,
    pii_by_column: Mapping[str, str] | None,
) -> pa.Schema:
    if not column_lineage and not pii_by_column:
        return schema
    fields: list[pa.Field] = []
    for field in schema:
        updates: dict[str, object] = {}
        updated_field = field
        if pii_by_column is not None:
            pii_class = pii_by_column.get(field.name)
            if pii_class is not None:
                updates["codeintel.pii_class"] = pii_class
        if column_lineage is not None:
            lineage = column_lineage.get(field.name)
            if lineage:
                updates["codeintel.lineage_edges"] = _lineage_payload(lineage)
        if updates:
            updated_field = merge_field_metadata(field, updates)
        fields.append(updated_field)
    return pa.schema(fields, metadata=schema.metadata)


def _lineage_payload(lineage: list[tuple[str, str]]) -> list[dict[str, str]]:
    entries = sorted(lineage, key=lambda item: (item[0], item[1]))
    return [{"table_key": table_key, "column": column} for table_key, column in entries]


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
    column_lineage = None
    if repo and commit:
        column_lineage = load_derived_lineage_columns(
            con,
            repo=repo,
            commit=commit,
            downstream_table=table_key,
        )
    observed_schema = _load_observed_schema(
        con,
        table_key=table_key,
    )
    contract_schema = _load_contract_schema(
        con,
        table_key=table_key,
        require_inferred=True,
    )
    resolved_schema = observed_schema or contract_schema
    if resolved_schema is None:
        return None
    return _apply_runtime_metadata(
        resolved_schema,
        column_lineage=column_lineage,
        pii_by_column=pii_by_column,
    )


@dataclass(frozen=True, slots=True)
class RegistryArrowSchemaProvider:
    """Arrow schema provider backed by registry metadata."""

    con: DuckDBPyConnection
    repo: str | None = None
    commit: str | None = None

    def get_arrow_schema(self, table_key: str) -> pa.Schema | None:
        """Return Arrow schema for the table key.

        Returns
        -------
        pa.Schema | None
            Arrow schema for the table when available.
        """
        return arrow_schema_for_table_key(
            self.con,
            table_key=table_key,
            repo=self.repo,
            commit=self.commit,
        )


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
    "RegistryArrowSchemaProvider",
    "arrow_schema_digest",
    "arrow_schema_for_table_key",
    "arrow_schema_hash",
]
