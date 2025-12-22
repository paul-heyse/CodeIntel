"""Utilities for opt-in db.query.text emission."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from typing import cast

from sqlglot import exp, parse_one
from sqlglot.errors import ParseError, SqlglotError

_DUCKDB_PLACEHOLDER_RE = re.compile(r"(\?)|(\$[0-9]+)|(\$[A-Za-z_][A-Za-z0-9_]*)")
_GENERIC_PLACEHOLDER_RE = re.compile(r"(\?)|(\$[0-9]+)|(:[A-Za-z_][A-Za-z0-9_]*)")
_BLOCK_COMMENT_RE = re.compile(r"/\*.*?\*/", re.DOTALL)
_LINE_COMMENT_RE = re.compile(r"--[^\n]*")
_IN_LIST_RE = re.compile(
    r"\bIN\s*\(\s*(\?\s*,\s*){2,}\?\s*\)",
    flags=re.IGNORECASE,
)

_BOOLEAN_EXPR = cast("type[exp.Expression] | None", getattr(exp, "Boolean", None))
_INTERVAL_EXPR = cast("type[exp.Expression] | None", getattr(exp, "Interval", None))
_HEX_STRING_EXPR = cast("type[exp.Expression] | None", getattr(exp, "HexString", None))
_PLACEHOLDER_EXPR = cast("type[exp.Expression] | None", getattr(exp, "Placeholder", None))


class DbQueryTextPolicy(StrEnum):
    """Policies for db.query.text emission."""

    NEVER = "never"
    PARAMETERIZED = "parameterized"
    REDACTED = "redacted"
    PARAMETERIZED_OR_REDACTED = "parameterized_or_redacted"
    FULL = "full"


@dataclass(frozen=True, slots=True)
class DbQueryTextConfig:
    """Configuration for db.query.text emission."""

    policy: DbQueryTextPolicy = DbQueryTextPolicy.NEVER
    max_len: int = 4096
    strip_comments: bool = True
    collapse_in_lists: bool = True


def looks_parameterized(sql: str, *, db_system_name: str) -> bool:
    """Return whether SQL text appears to use placeholders.

    Parameters
    ----------
    sql
        SQL statement text.
    db_system_name
        Database system identifier (for example, ``duckdb``).

    Returns
    -------
    bool
        True when placeholder tokens are detected.

    Examples
    --------
    >>> looks_parameterized("SELECT * FROM t WHERE id = $id", db_system_name="duckdb")
    True

    Notes
    -----
    The detection is heuristic and tuned for common placeholders.
    """
    system = (db_system_name or "").lower()
    if system == "duckdb":
        return bool(_DUCKDB_PLACEHOLDER_RE.search(sql))
    return bool(_GENERIC_PLACEHOLDER_RE.search(sql))


def redact_sql_literals_with_sqlglot(
    sql: str,
    *,
    dialect: str | None,
    config: DbQueryTextConfig,
) -> str | None:
    """Return SQL text with literals replaced by placeholders, or None on failure.

    Parameters
    ----------
    sql
        SQL statement text.
    dialect
        Optional SQL dialect name for sqlglot parsing.
    config
        Redaction configuration settings.

    Returns
    -------
    str | None
        Redacted SQL text, or None when parsing fails.

    Examples
    --------
    >>> config = DbQueryTextConfig()
    >>> redact_sql_literals_with_sqlglot("SELECT 1", dialect="duckdb", config=config) is not None
    True

    Notes
    -----
    The returned text is truncated to the configured maximum length and does not raise.
    """
    text = _strip_comments(sql) if config.strip_comments else sql
    try:
        root = parse_one(text, dialect=dialect) if dialect else parse_one(text)
    except (ParseError, SqlglotError, ValueError):
        return None

    sanitized = root.transform(_redact_literals)
    rendered = sanitized.sql(dialect=dialect) if dialect else sanitized.sql()
    if config.collapse_in_lists:
        rendered = _collapse_in_lists(rendered)
    return _truncate(rendered, config.max_len)


def _strip_comments(sql: str) -> str:
    return _LINE_COMMENT_RE.sub(" ", _BLOCK_COMMENT_RE.sub(" ", sql))


def _collapse_in_lists(sql: str) -> str:
    return _IN_LIST_RE.sub("IN (?)", sql)


def _truncate(text: str, max_len: int) -> str:
    if max_len <= 0:
        return ""
    if len(text) <= max_len:
        return text
    if max_len == 1:
        return "."
    return f"{text[: max_len - 3]}..."


def _redact_literals(node: exp.Expression) -> exp.Expression:
    if isinstance(node, exp.Literal):
        return _new_placeholder()
    if _BOOLEAN_EXPR is not None and isinstance(node, _BOOLEAN_EXPR):
        return _new_placeholder()
    if _INTERVAL_EXPR is not None and isinstance(node, _INTERVAL_EXPR):
        return _new_placeholder()
    if _HEX_STRING_EXPR is not None and isinstance(node, _HEX_STRING_EXPR):
        return _new_placeholder()
    return node


def _new_placeholder() -> exp.Expression:
    if _PLACEHOLDER_EXPR is not None:
        return _PLACEHOLDER_EXPR()
    return exp.Literal.string("?")


__all__ = [
    "DbQueryTextConfig",
    "DbQueryTextPolicy",
    "looks_parameterized",
    "redact_sql_literals_with_sqlglot",
]
