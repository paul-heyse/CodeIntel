"""Arrow schema rendering helpers for storage boundaries."""

from __future__ import annotations

import json
from collections.abc import Mapping
from functools import lru_cache
from typing import TYPE_CHECKING

import pyarrow as pa
from sqlglot import exp

from codeintel.core.schemas.contracts import (
    arrow_schema_digest,
    arrow_schema_hash,
    arrow_schema_from_fields,
    try_decode_schema_ipc_b64,
)
from codeintel.storage.helpers.json import decode_json_dict
from codeintel.storage.metadata import load_derived_lineage_columns
from codeintel.storage.metadata.meta_catalog import meta_table_ref
from codeintel.storage.sqlglot_tools import render_sql_duckdb

if TYPE_CHECKING:
    from duckdb import DuckDBPyConnection

_TABLE_REF_WITH_CATALOG_PARTS = 3
_TABLE_REF_WITH_SCHEMA_PARTS = 2


def _load_contract_schema(
    con: DuckDBPyConnection,
    *,
    table_key: str,
    require_inferred: bool,
) -> pa.Schema | None:
    registry_ref = meta_table_ref("metadata.table_schema_registry")
    versions_ref = meta_table_ref("metadata.schema_versions")
    row = con.execute(
        _contract_schema_sql(
            registry_ref=registry_ref,
            versions_ref=versions_ref,
            require_inferred=require_inferred,
        ),
        [table_key],
    ).fetchone()
    if row is None or row[0] is None:
        return None
    renderer_cache = decode_json_dict(row[0])
    ipc_payload = renderer_cache.get("arrow_schema_ipc_b64")
    if not isinstance(ipc_payload, str):
        return None
    return try_decode_schema_ipc_b64(ipc_payload)


def _load_observed_schema(
    con: DuckDBPyConnection,
    *,
    table_key: str,
) -> pa.Schema | None:
    observations_ref = meta_table_ref("metadata.schema_observations")
    row = con.execute(
        _observed_schema_sql(observations_ref=observations_ref),
        [table_key],
    ).fetchone()
    if row is None or row[0] is None:
        return None
    if not isinstance(row[0], str):
        return None
    return try_decode_schema_ipc_b64(row[0])


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
            updated_field = _merge_field_metadata(field, updates)
        fields.append(updated_field)
    return arrow_schema_from_fields(fields=fields, metadata=schema.metadata)


def _merge_field_metadata(field: pa.Field, updates: Mapping[str, object]) -> pa.Field:
    existing = _decode_metadata(field.metadata)
    merged = dict(existing)
    for key, value in updates.items():
        if value is None or key in merged:
            continue
        merged[key] = value
    return field.with_metadata(_encode_metadata(merged))


def _decode_metadata(metadata: Mapping[bytes, bytes] | None) -> dict[str, object]:
    if not metadata:
        return {}
    decoded: dict[str, object] = {}
    for key, raw in metadata.items():
        key_str = key.decode("utf-8")
        raw_str = raw.decode("utf-8")
        decoded[key_str] = _decode_metadata_value(raw_str)
    return decoded


def _decode_metadata_value(raw: str) -> object:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def _encode_metadata(metadata: Mapping[str, object]) -> dict[bytes, bytes] | None:
    if not metadata:
        return None
    encoded: dict[bytes, bytes] = {}
    for key, value in metadata.items():
        if value is None:
            continue
        if isinstance(value, str):
            raw = value
        else:
            raw = json.dumps(value, sort_keys=True, separators=(",", ":"))
        encoded[key.encode("utf-8")] = raw.encode("utf-8")
    return encoded or None


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
    contract_schema = _load_contract_schema(
        con,
        table_key=table_key,
        require_inferred=True,
    )
    observed_schema = None
    if contract_schema is None:
        observed_schema = _load_observed_schema(
            con,
            table_key=table_key,
        )
    resolved_schema = contract_schema or observed_schema
    if resolved_schema is None:
        return None
    return _apply_runtime_metadata(
        resolved_schema,
        column_lineage=column_lineage,
        pii_by_column=pii_by_column,
    )


@lru_cache(maxsize=8)
def _contract_schema_sql(
    *,
    registry_ref: str,
    versions_ref: str,
    require_inferred: bool,
) -> str:
    registry = _table_expr(registry_ref, alias="registry")
    versions = _table_expr(versions_ref, alias="versions")
    join_on = exp.EQ(
        this=exp.column("schema_digest", table="registry"),
        expression=exp.column("schema_digest", table="versions"),
    )
    predicate = exp.EQ(
        this=exp.column("table_key", table="registry"),
        expression=exp.Parameter(),
    )
    if require_inferred:
        predicate = exp.and_(predicate, _inferred_registry_predicate())
    query = exp.select(exp.column("renderer_cache", table="versions")).from_(registry)
    query = query.join(versions, on=join_on)
    query = query.where(predicate)
    return render_sql_duckdb(query)


@lru_cache(maxsize=4)
def _observed_schema_sql(*, observations_ref: str) -> str:
    observations = _table_expr(observations_ref, alias="observations")
    query = exp.select(exp.column("arrow_schema_ipc_b64", table="observations")).from_(observations)
    query = query.where(
        exp.EQ(
            this=exp.column("table_key", table="observations"),
            expression=exp.Parameter(),
        )
    )
    query = query.order_by(
        exp.Ordered(
            this=exp.column("observed_at", table="observations"),
            desc=True,
        )
    )
    query = query.limit(1)
    return render_sql_duckdb(query)


def _inferred_registry_predicate() -> exp.Expression:
    status_col = exp.column("inference_status", table="registry")
    return exp.or_(
        exp.In(
            this=status_col,
            expressions=[exp.Literal.string("inferred"), exp.Literal.string("override")],
        ),
        exp.Is(this=status_col, expression=exp.Null()),
        exp.In(
            this=exp.column("derivation_kind", table="registry"),
            expressions=[
                exp.Literal.string("inferred_relation"),
                exp.Literal.string("view_inferred"),
            ],
        ),
    )


def _table_expr(table_ref: str, *, alias: str | None = None) -> exp.Table:
    parts = table_ref.split(".")
    catalog: str | None = None
    schema: str | None = None
    table: str
    if len(parts) == _TABLE_REF_WITH_CATALOG_PARTS:
        catalog, schema, table = parts
    elif len(parts) == _TABLE_REF_WITH_SCHEMA_PARTS:
        schema, table = parts
    else:
        table = parts[0]
    return exp.Table(
        this=exp.to_identifier(table),
        db=exp.to_identifier(schema) if schema else None,
        catalog=exp.to_identifier(catalog) if catalog else None,
        alias=exp.TableAlias(this=exp.to_identifier(alias)) if alias else None,
    )


__all__ = [
    "arrow_schema_digest",
    "arrow_schema_for_table_key",
    "arrow_schema_hash",
]
