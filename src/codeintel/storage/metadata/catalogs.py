"""Canonical catalog persistence for contract and target registries."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Protocol

from sqlglot import exp

from codeintel.core.serialization.payload import decode_payload, encode_payload
from codeintel.core.sqlglot_tools import render_sql_duckdb, table_expr_from_ref
from codeintel.storage.metadata.meta_catalog import meta_table_ref

if TYPE_CHECKING:
    from duckdb import DuckDBPyConnection

_JSON_PAYLOAD_TYPES = (str, bytes, bytearray, memoryview)


class CatalogGateway(Protocol):
    """Protocol for gateway-like objects with a DuckDB connection."""

    @property
    def con(self) -> DuckDBPyConnection:
        """Return an open DuckDB connection."""
        ...


@dataclass(frozen=True, slots=True)
class CanonicalCatalogEntry:
    """Stored canonical catalog payload."""

    catalog_kind: str
    catalog_hash: str
    payload: dict[str, Any]
    created_at: datetime
    inputs: dict[str, Any] | None = None


def _coerce_json_payload(value: object) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, _JSON_PAYLOAD_TYPES):
        decoded = decode_payload(value)
        if isinstance(decoded, Mapping):
            return dict(decoded)
        if isinstance(value, (bytes, bytearray, memoryview)):
            raw = bytes(value).decode("utf-8")
        else:
            raw = value
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return dict(parsed)
        msg = "Catalog payload must be a JSON object"
        raise TypeError(msg)
    msg = "Catalog payload must be a mapping or JSON string"
    raise TypeError(msg)


def load_canonical_catalog(
    gateway: CatalogGateway,
    *,
    catalog_kind: str,
    catalog_hash: str,
) -> CanonicalCatalogEntry | None:
    """Load a canonical catalog entry by kind and hash.

    Returns
    -------
    CanonicalCatalogEntry | None
        Catalog entry when present; otherwise None.
    """
    if not catalog_kind or not catalog_hash:
        return None
    table_ref = meta_table_ref("metadata.canonical_catalogs")
    table_expr = table_expr_from_ref(table_ref)
    predicate = exp.and_(
        exp.EQ(this=exp.column("catalog_kind"), expression=exp.Placeholder()),
        exp.EQ(this=exp.column("catalog_hash"), expression=exp.Placeholder()),
    )
    query = (
        exp.select(
            exp.column("payload"),
            exp.column("inputs"),
            exp.column("created_at"),
        )
        .from_(table_expr)
        .where(predicate)
    )
    row = gateway.con.execute(
        render_sql_duckdb(query),
        [catalog_kind, catalog_hash],
    ).fetchone()
    if row is None:
        return None
    payload_raw, inputs_raw, created_at = row
    payload = _coerce_json_payload(payload_raw)
    inputs = _coerce_json_payload(inputs_raw) if inputs_raw is not None else None
    return CanonicalCatalogEntry(
        catalog_kind=catalog_kind,
        catalog_hash=catalog_hash,
        payload=payload,
        inputs=inputs,
        created_at=created_at,
    )


def load_latest_canonical_catalog(
    gateway: CatalogGateway,
    *,
    catalog_kind: str,
) -> CanonicalCatalogEntry | None:
    """Load the latest canonical catalog entry for a kind.

    Returns
    -------
    CanonicalCatalogEntry | None
        Latest catalog entry for the kind, or None when unavailable.
    """
    if not catalog_kind:
        return None
    table_ref = meta_table_ref("metadata.canonical_catalogs")
    table_expr = table_expr_from_ref(table_ref)
    query = (
        exp.select(
            exp.column("catalog_hash"),
            exp.column("payload"),
            exp.column("inputs"),
            exp.column("created_at"),
        )
        .from_(table_expr)
        .where(exp.EQ(this=exp.column("catalog_kind"), expression=exp.Placeholder()))
        .order_by(exp.Ordered(this=exp.column("created_at"), desc=True))
        .limit(1)
    )
    row = gateway.con.execute(render_sql_duckdb(query), [catalog_kind]).fetchone()
    if row is None:
        return None
    catalog_hash, payload_raw, inputs_raw, created_at = row
    payload = _coerce_json_payload(payload_raw)
    inputs = _coerce_json_payload(inputs_raw) if inputs_raw is not None else None
    return CanonicalCatalogEntry(
        catalog_kind=catalog_kind,
        catalog_hash=catalog_hash,
        payload=payload,
        inputs=inputs,
        created_at=created_at,
    )


def load_latest_canonical_catalog_from_connection(
    con: DuckDBPyConnection,
    *,
    catalog_kind: str,
) -> CanonicalCatalogEntry | None:
    """Load the latest canonical catalog entry for a kind using a DuckDB connection.

    Returns
    -------
    CanonicalCatalogEntry | None
        Latest catalog entry for the kind, or None when unavailable.
    """
    if not catalog_kind:
        return None
    table_ref = meta_table_ref("metadata.canonical_catalogs")
    table_expr = table_expr_from_ref(table_ref)
    query = (
        exp.select(
            exp.column("catalog_hash"),
            exp.column("payload"),
            exp.column("inputs"),
            exp.column("created_at"),
        )
        .from_(table_expr)
        .where(exp.EQ(this=exp.column("catalog_kind"), expression=exp.Placeholder()))
        .order_by(exp.Ordered(this=exp.column("created_at"), desc=True))
        .limit(1)
    )
    row = con.execute(render_sql_duckdb(query), [catalog_kind]).fetchone()
    if row is None:
        return None
    catalog_hash, payload_raw, inputs_raw, created_at = row
    payload = _coerce_json_payload(payload_raw)
    inputs = _coerce_json_payload(inputs_raw) if inputs_raw is not None else None
    return CanonicalCatalogEntry(
        catalog_kind=catalog_kind,
        catalog_hash=catalog_hash,
        payload=payload,
        inputs=inputs,
        created_at=created_at,
    )


def upsert_canonical_catalog(
    gateway: CatalogGateway,
    entry: CanonicalCatalogEntry,
) -> None:
    """Insert or update a canonical catalog entry."""
    created_at = entry.created_at.astimezone(UTC)
    payload_json = encode_payload(entry.payload)
    inputs_json = encode_payload(entry.inputs) if entry.inputs is not None else None
    table_ref = meta_table_ref("metadata.canonical_catalogs")
    table_expr = table_expr_from_ref(table_ref)
    columns = [
        "catalog_kind",
        "catalog_hash",
        "payload",
        "inputs",
        "created_at",
    ]
    placeholders = [exp.Placeholder() for _ in columns]
    insert = exp.Insert(
        this=exp.Schema(
            this=table_expr,
            expressions=[exp.to_identifier(column) for column in columns],
        ),
        expression=exp.Values(expressions=[exp.Tuple(expressions=placeholders)]),
        conflict=exp.OnConflict(
            conflict_keys=[exp.to_identifier("catalog_kind"), exp.to_identifier("catalog_hash")],
            action=exp.Var(this="DO UPDATE"),
            expressions=[
                exp.EQ(
                    this=exp.column("payload"),
                    expression=exp.column("payload", table="excluded"),
                ),
                exp.EQ(
                    this=exp.column("inputs"),
                    expression=exp.column("inputs", table="excluded"),
                ),
                exp.EQ(
                    this=exp.column("created_at"),
                    expression=exp.column("created_at", table="excluded"),
                ),
            ],
        ),
    )
    gateway.con.execute(
        render_sql_duckdb(insert),
        [entry.catalog_kind, entry.catalog_hash, payload_json, inputs_json, created_at],
    )


def build_catalog_entry(
    *,
    catalog_kind: str,
    catalog_hash: str,
    payload: dict[str, Any],
    inputs: dict[str, Any] | None = None,
) -> CanonicalCatalogEntry:
    """Construct a canonical catalog entry with a current timestamp.

    Returns
    -------
    CanonicalCatalogEntry
        New catalog entry with current timestamp.
    """
    return CanonicalCatalogEntry(
        catalog_kind=catalog_kind,
        catalog_hash=catalog_hash,
        payload=payload,
        inputs=inputs,
        created_at=datetime.now(tz=UTC),
    )


__all__ = [
    "CanonicalCatalogEntry",
    "build_catalog_entry",
    "load_canonical_catalog",
    "load_latest_canonical_catalog",
    "load_latest_canonical_catalog_from_connection",
    "upsert_canonical_catalog",
]
