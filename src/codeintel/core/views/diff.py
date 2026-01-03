"""Semantic SQL diffing utilities for view evolution.

These helpers produce stable, JSON-serializable summaries of changes between
two compiled SQL statements or view SQL maps. They are designed for build-time
artifacts and cached diff records.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from codeintel.core.queries.ast import coerce_ast, diff_ast
from codeintel.core.sqlglot_tools import (
    ParseError,
    extract_table_keys_duckdb,
    fingerprint_sql_duckdb,
)

if TYPE_CHECKING:
    from collections.abc import Mapping


def _sha256_text(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


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


@dataclass(frozen=True, slots=True)
class SqlStructuralDiffSummary:
    """AST-level diff summary between two SQL strings."""

    changed: bool
    actions: dict[str, int]
    parse_error: str | None = None
    before_ast: list[dict[str, object]] | None = None
    after_ast: list[dict[str, object]] | None = None

    def to_json_obj(self) -> dict[str, object]:
        """Return a JSON-serializable representation.

        Returns
        -------
        dict[str, object]
            JSON-ready payload with stable keys.
        """
        payload: dict[str, object] = {
            "changed": self.changed,
            "actions": dict(self.actions),
            "parse_error": self.parse_error,
        }
        if self.before_ast is not None:
            payload["before_ast"] = self.before_ast
        if self.after_ast is not None:
            payload["after_ast"] = self.after_ast
        return payload


def diff_sql_structural(
    *,
    before: str,
    after: str,
    include_ast: bool = False,
) -> SqlStructuralDiffSummary:
    """Compute a SQLGlot structural diff summary for two SQL strings.

    Notes
    -----
    This is intentionally additive to the existing `diff_sql` function; callers
    can opt into structural diffs without changing artifact formats.

    Returns
    -------
    SqlStructuralDiffSummary
        Structural diff summary.
    """
    try:
        before_ast = coerce_ast(before)
        after_ast = coerce_ast(after)
        actions = diff_ast(before_ast, after_ast)
        counts: dict[str, int] = {}
        for action in actions:
            name = type(action).__name__
            counts[name] = counts.get(name, 0) + 1
        before_payload = _dump_ast(before_ast) if include_ast else None
        after_payload = _dump_ast(after_ast) if include_ast else None
        return SqlStructuralDiffSummary(
            changed=fingerprint_sql_duckdb(before) != fingerprint_sql_duckdb(after),
            actions=counts,
            before_ast=before_payload,
            after_ast=after_payload,
        )
    except (ParseError, ValueError, TypeError) as exc:
        return SqlStructuralDiffSummary(
            changed=before != after,
            actions={},
            parse_error=str(exc),
        )


def diff_sql(*, before: str, after: str) -> SqlDiffSummary:
    """Compute a semantic-ish diff summary for two SQL strings.

    Returns
    -------
    SqlDiffSummary
        Parsed diff summary (hashes + referenced table deltas).
    """
    try:
        before_hash = fingerprint_sql_duckdb(before)
        after_hash = fingerprint_sql_duckdb(after)
        before_tables = extract_table_keys_duckdb(before)
        after_tables = extract_table_keys_duckdb(after)
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
    include_structural: bool = False,
    include_ast: bool = False,
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
            payload: dict[str, object] = {
                "status": "added",
                "diff": diff_sql(before="", after=after_sql).to_json_obj(),
            }
            if include_structural:
                payload["structural_diff"] = diff_sql_structural(
                    before="",
                    after=after_sql,
                    include_ast=include_ast,
                ).to_json_obj()
            out[key] = payload
            continue
        if key not in after_keys:
            payload = {
                "status": "removed",
                "diff": diff_sql(before=before_sql, after="").to_json_obj(),
            }
            if include_structural:
                payload["structural_diff"] = diff_sql_structural(
                    before=before_sql,
                    after="",
                    include_ast=include_ast,
                ).to_json_obj()
            out[key] = payload
            continue
        summary = diff_sql(before=before_sql, after=after_sql)
        payload = {
            "status": "changed" if summary.changed else "unchanged",
            "diff": summary.to_json_obj(),
        }
        if include_structural:
            payload["structural_diff"] = diff_sql_structural(
                before=before_sql,
                after=after_sql,
                include_ast=include_ast,
            ).to_json_obj()
        out[key] = payload

    return out


def _dump_ast(root: object) -> list[dict[str, object]]:
    payload = getattr(root, "dump", None)
    if not callable(payload):
        return []
    return cast("list[dict[str, object]]", payload())


__all__ = [
    "SqlDiffSummary",
    "diff_sql",
    "diff_view_sql_maps",
]
