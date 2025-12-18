"""Semantic SQL diffing utilities for view evolution.

These helpers produce stable, JSON-serializable summaries of changes between
two compiled SQL statements or view SQL maps. They are designed for build-time
artifacts and cached diff records.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import TYPE_CHECKING

from sqlglot import exp, parse_one
from sqlglot.errors import ParseError

from codeintel.storage.constants import DUCKDB_DIALECT

if TYPE_CHECKING:
    from collections.abc import Mapping


def _sha256_text(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _canonical_sql(sql: str) -> str:
    root = parse_one(sql, dialect=DUCKDB_DIALECT)
    return root.sql(dialect=DUCKDB_DIALECT)


def _extract_tables(sql: str) -> frozenset[str]:
    root = parse_one(sql, dialect=DUCKDB_DIALECT)
    tables: set[str] = set()
    for table in root.find_all(exp.Table):
        name = table.name
        schema = table.db
        if schema:
            tables.add(f"{schema}.{name}".lower())
        else:
            tables.add(name.lower())
    return frozenset(tables)


@dataclass(frozen=True, slots=True)
class SqlDiffSummary:
    """Summary of changes between two SQL strings."""

    changed: bool
    from_hash: str
    to_hash: str
    tables_added: tuple[str, ...] = ()
    tables_removed: tuple[str, ...] = ()
    parse_error: str | None = None

    def to_json_obj(self) -> dict[str, object]:
        """Return a JSON-serializable representation.

        Returns
        -------
        dict[str, object]
            JSON-ready payload with stable keys.
        """
        return {
            "changed": self.changed,
            "from_hash": self.from_hash,
            "to_hash": self.to_hash,
            "tables_added": list(self.tables_added),
            "tables_removed": list(self.tables_removed),
            "parse_error": self.parse_error,
        }


def diff_sql(*, before: str, after: str) -> SqlDiffSummary:
    """Compute a semantic-ish diff summary for two SQL strings.

    Returns
    -------
    SqlDiffSummary
        Parsed diff summary (hashes + referenced table deltas).
    """
    try:
        before_canon = _canonical_sql(before)
        after_canon = _canonical_sql(after)
        before_hash = _sha256_text(before_canon)
        after_hash = _sha256_text(after_canon)
        before_tables = _extract_tables(before)
        after_tables = _extract_tables(after)
        tables_added = tuple(sorted(after_tables - before_tables))
        tables_removed = tuple(sorted(before_tables - after_tables))
        return SqlDiffSummary(
            changed=before_hash != after_hash,
            from_hash=before_hash,
            to_hash=after_hash,
            tables_added=tables_added,
            tables_removed=tables_removed,
            parse_error=None,
        )
    except (ParseError, ValueError, TypeError) as exc:
        before_hash = _sha256_text(before)
        after_hash = _sha256_text(after)
        return SqlDiffSummary(
            changed=before_hash != after_hash,
            from_hash=before_hash,
            to_hash=after_hash,
            tables_added=(),
            tables_removed=(),
            parse_error=str(exc),
        )


def diff_view_sql_maps(
    *,
    before: Mapping[str, str],
    after: Mapping[str, str],
) -> dict[str, dict[str, object]]:
    """Diff two view SQL maps keyed by table_key.

    Returns
    -------
    dict[str, dict[str, object]]
        Mapping of lowercased view_key -> status/diff payload.
    """
    before_by_lower = {k.lower(): v for k, v in before.items()}
    after_by_lower = {k.lower(): v for k, v in after.items()}
    before_keys = set(before_by_lower)
    after_keys = set(after_by_lower)

    out: dict[str, dict[str, object]] = {}

    for key in sorted(before_keys | after_keys):
        before_sql = before_by_lower.get(key, "")
        after_sql = after_by_lower.get(key, "")
        if key not in before_keys:
            out[key] = {
                "status": "added",
                "diff": diff_sql(before="", after=after_sql).to_json_obj(),
            }
            continue
        if key not in after_keys:
            out[key] = {
                "status": "removed",
                "diff": diff_sql(before=before_sql, after="").to_json_obj(),
            }
            continue
        summary = diff_sql(before=before_sql, after=after_sql)
        out[key] = {
            "status": "changed" if summary.changed else "unchanged",
            "diff": summary.to_json_obj(),
        }

    return out


__all__ = [
    "SqlDiffSummary",
    "diff_sql",
    "diff_view_sql_maps",
]
