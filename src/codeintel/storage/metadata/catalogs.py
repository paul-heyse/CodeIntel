"""Canonical catalog persistence for contract and target registries."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from duckdb import DuckDBPyConnection

_JSON_PAYLOAD_TYPES = (str, bytes, bytearray)


class CatalogGateway(Protocol):
    """Protocol for gateway-like objects with a DuckDB connection."""

    @property
    def con(self) -> DuckDBPyConnection:
        """Return an open DuckDB connection."""
        ...


@dataclass(frozen=True)
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
        raw = value.decode("utf-8") if isinstance(value, (bytes, bytearray)) else value
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
    row = gateway.con.execute(
        """
        SELECT payload, inputs, created_at
        FROM metadata.canonical_catalogs
        WHERE catalog_kind = ? AND catalog_hash = ?
        """,
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
    row = gateway.con.execute(
        """
        SELECT catalog_hash, payload, inputs, created_at
        FROM metadata.canonical_catalogs
        WHERE catalog_kind = ?
        ORDER BY created_at DESC
        LIMIT 1
        """,
        [catalog_kind],
    ).fetchone()
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
    row = con.execute(
        """
        SELECT catalog_hash, payload, inputs, created_at
        FROM metadata.canonical_catalogs
        WHERE catalog_kind = ?
        ORDER BY created_at DESC
        LIMIT 1
        """,
        [catalog_kind],
    ).fetchone()
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
    payload_json = json.dumps(entry.payload, sort_keys=True)
    inputs_json = json.dumps(entry.inputs, sort_keys=True) if entry.inputs is not None else None
    gateway.con.execute(
        """
        INSERT INTO metadata.canonical_catalogs (
            catalog_kind,
            catalog_hash,
            payload,
            inputs,
            created_at
        )
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT (catalog_kind, catalog_hash) DO UPDATE SET
            payload = excluded.payload,
            inputs = excluded.inputs,
            created_at = excluded.created_at
        """,
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
