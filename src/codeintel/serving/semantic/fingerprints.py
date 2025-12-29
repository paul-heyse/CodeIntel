"""Stable query fingerprint utilities for serving.

These helpers compute deterministic fingerprints for serving-layer queries.
Fingerprints are intended for:

- Observability (attach to logs/metrics)
- Future caching keys
- Debuggability (link a response back to its normalized inputs)

Fingerprints are computed over *validated, canonicalized* inputs and must be:

- Stable across runs for identical inputs
- Sensitive to relevant changes (filters/select/order/limit/offset, snapshot, schema)
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

from codeintel.storage.sqlglot_tools import fingerprint_sql_duckdb_safe

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence


def stable_json_dumps(payload: object) -> str:
    """Serialize payload to a stable JSON string.

    Parameters
    ----------
    payload
        JSON-serializable payload.

    Returns
    -------
    str
        Stable JSON string (sorted keys, compact separators).
    """
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    )


def short_sha256_hex(text: str, *, length: int = 16) -> str:
    """Return a short SHA-256 hex digest for a string.

    Parameters
    ----------
    text
        Input text to hash.
    length
        Number of hex characters to return.

    Returns
    -------
    str
        Lowercase hex digest prefix.
    """
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return digest[:length]


def sqlglot_canonical_sha256(sql: str) -> str:
    """Canonicalize SQL via SQLGlot and return sha256 hex digest.

    Parameters
    ----------
    sql
        SQL string to canonicalize and hash.

    Returns
    -------
    str
        SHA256 hex digest of the canonical SQL form.
    """
    return fingerprint_sql_duckdb_safe(sql)


def fingerprint(payload: Mapping[str, object], *, prefix: str = "q_") -> str:
    """Compute a stable fingerprint for a JSON payload.

    Parameters
    ----------
    payload
        Mapping of canonicalized inputs.
    prefix
        Prefix for the returned fingerprint string.

    Returns
    -------
    str
        Fingerprint string.
    """
    canon = stable_json_dumps(payload)
    return f"{prefix}{short_sha256_hex(canon)}"


def canonicalize_list(items: Sequence[object]) -> list[object]:
    """Return a canonical list representation for hashing.

    Parameters
    ----------
    items
        Sequence of JSON-serializable items.

    Returns
    -------
    list[object]
        List copy.
    """
    return list(items)


def canonicalize_order_insensitive_dicts(
    items: Iterable[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Canonicalize a set-like collection of dicts for hashing.

    This is intended for inputs whose order is semantically irrelevant, e.g.
    filter conjunctions.

    Parameters
    ----------
    items
        Iterable of JSON-serializable mapping objects.

    Returns
    -------
    list[dict[str, object]]
        List of dicts sorted by stable JSON string.
    """
    normalized: list[dict[str, object]] = [dict(item) for item in items]
    normalized.sort(key=stable_json_dumps)
    return normalized


def canonicalize_strings_unordered(values: Sequence[str] | None) -> list[str] | None:
    """Canonicalize an optional string sequence where order is not meaningful.

    Parameters
    ----------
    values
        Optional sequence of strings.

    Returns
    -------
    list[str] | None
        Sorted unique list or None when no values are provided.
    """
    if values is None:
        return None
    items = [v for v in values if v]
    if not items:
        return None
    return sorted(set(items))


def coerce_jsonable(value: object) -> object:
    """Coerce a value into a JSON-serializable form for hashing.

    Parameters
    ----------
    value
        Arbitrary value.

    Returns
    -------
    object
        JSON-serializable value.
    """
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, list):
        return [coerce_jsonable(item) for item in value]
    if isinstance(value, dict):
        out: dict[str, object] = {}
        for k, v in value.items():
            out[str(k)] = coerce_jsonable(v)
        return out
    return str(value)


def canonicalize_filter_dicts(filters: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    """Canonicalize semantic filter specs for hashing.

    Parameters
    ----------
    filters
        Sequence of mapping objects with keys: column, op, value.

    Returns
    -------
    list[dict[str, object]]
        Canonical filter dicts sorted deterministically.
    """
    normalized: list[dict[str, object]] = []
    for raw in filters:
        column = raw.get("column")
        op = raw.get("op")
        value = raw.get("value")
        normalized.append(
            {
                "column": str(column) if column is not None else "",
                "op": str(op) if op is not None else "",
                "value": coerce_jsonable(value),
            }
        )
    return canonicalize_order_insensitive_dicts(normalized)


@dataclass(frozen=True, slots=True)
class SemanticQueryFingerprintInput:
    """Inputs for fingerprinting a semantic query/export request.

    Parameters
    ----------
    snapshot
        Snapshot identity (repo/commit/run_id).
    view_id
        Semantic view identifier.
    table_key
        Underlying table/view key.
    select
        Selected columns in order.
    order_by
        Ordering spec in order.
    filters
        Filter specs (mapping objects) in any order.
    limit
        Effective limit (user-visible).
    offset
        Effective offset.
    schema_hash
        Optional schema fingerprint for stability across schema changes.
    ast_hash
        Optional canonical SQLGlot AST fingerprint for semantic stability.
    """

    snapshot: Mapping[str, str]
    view_id: str
    table_key: str
    select: Sequence[str]
    order_by: Sequence[str]
    filters: Sequence[Mapping[str, object]]
    limit: int
    offset: int
    schema_hash: str | None = None
    ast_hash: str | None = None

    def canonical_payload(self) -> dict[str, object]:
        """Return a canonical JSON payload for hashing.

        Returns
        -------
        dict[str, object]
            Canonical payload with stable ordering.
        """
        return {
            "kind": "semantic",
            "snapshot": dict(self.snapshot),
            "view_id": self.view_id,
            "table_key": self.table_key,
            "select": canonicalize_list(self.select),
            "order_by": canonicalize_list(self.order_by),
            "filters": canonicalize_filter_dicts(self.filters),
            "limit": int(self.limit),
            "offset": int(self.offset),
            "schema_hash": self.schema_hash,
            "ast_hash": self.ast_hash,
        }


def fingerprint_semantic_query(inputs: SemanticQueryFingerprintInput) -> str:
    """Fingerprint a semantic query/export request.

    Parameters
    ----------
    inputs
        Normalized inputs for fingerprinting.

    Returns
    -------
    str
        Query fingerprint string.
    """
    return fingerprint(inputs.canonical_payload())


def fingerprint_search(
    *,
    snapshot: Mapping[str, str],
    query: str,
    kinds: Sequence[str] | None,
    limit: int,
    offset: int,
) -> str:
    """Fingerprint a search request.

    Parameters
    ----------
    snapshot
        Snapshot identity (repo/commit/run_id).
    query
        Search query text.
    kinds
        Optional list of kinds (order-insensitive).
    limit
        Effective limit.
    offset
        Effective offset.

    Returns
    -------
    str
        Query fingerprint string.
    """
    payload: dict[str, object] = {
        "kind": "search",
        "snapshot": dict(snapshot),
        "query": query,
        "kinds": canonicalize_strings_unordered(kinds),
        "limit": int(limit),
        "offset": int(offset),
    }
    return fingerprint(payload)


__all__ = [
    "SemanticQueryFingerprintInput",
    "canonicalize_filter_dicts",
    "canonicalize_list",
    "canonicalize_order_insensitive_dicts",
    "canonicalize_strings_unordered",
    "coerce_jsonable",
    "fingerprint",
    "fingerprint_search",
    "fingerprint_semantic_query",
    "short_sha256_hex",
    "sqlglot_canonical_sha256",
    "stable_json_dumps",
]
