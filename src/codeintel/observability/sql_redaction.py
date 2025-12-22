"""SQL statement redaction utilities for telemetry."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal, overload

from codeintel.storage.sqlglot_tools import fingerprint_sql_duckdb_safe

SQLStatementMode = Literal["full", "hash", "operation", "none"]


_LEADING_WS_RE = re.compile(r"^\s+")
_SQL_BLOCK_COMMENT_RE = re.compile(r"/\*.*?\*/", re.DOTALL)
_SQL_LINE_COMMENT_RE = re.compile(r"--[^\n]*")

@dataclass(frozen=True, slots=True)
class RedactedSQL:
    """Result of applying a redaction policy to a SQL statement."""

    mode: SQLStatementMode
    operation: str
    statement_hash: str | None
    display: str


def _to_text(statement: str | bytes) -> str:
    if isinstance(statement, bytes):
        return statement.decode("utf-8", "replace")
    return statement


def _strip_comments(sql: str) -> str:
    return _SQL_LINE_COMMENT_RE.sub(" ", _SQL_BLOCK_COMMENT_RE.sub(" ", sql))


def _extract_operation(sql: str) -> str:
    sql = _strip_comments(sql)
    sql = _LEADING_WS_RE.sub("", sql)
    head = sql.split(maxsplit=1)
    return head[0] if head else ""


@overload
def redact_sql(
    statement: str,
    *,
    mode: SQLStatementMode = "hash",
    hash_len: int = 16,
) -> RedactedSQL: ...


@overload
def redact_sql(
    statement: bytes,
    *,
    mode: SQLStatementMode = "hash",
    hash_len: int = 16,
) -> RedactedSQL: ...


def redact_sql(
    statement: str | bytes,
    *,
    mode: SQLStatementMode = "hash",
    hash_len: int = 16,
) -> RedactedSQL:
    """Redact a SQL statement for safe telemetry output.

    Returns
    -------
    RedactedSQL
        Redacted statement metadata for telemetry.
    """
    text = _to_text(statement)
    operation = _extract_operation(text)

    if mode == "full":
        return RedactedSQL(mode=mode, operation=operation, statement_hash=None, display=text)

    digest = fingerprint_sql_duckdb_safe(text) if text else None

    if mode == "none":
        return RedactedSQL(mode=mode, operation=operation, statement_hash=digest, display="")

    if mode == "operation":
        return RedactedSQL(mode=mode, operation=operation, statement_hash=digest, display=operation)

    prefix = (digest or "")[: max(0, hash_len)]
    display = f"{operation} [sha256:{prefix}]" if operation and prefix else operation
    return RedactedSQL(mode="hash", operation=operation, statement_hash=digest, display=display)


__all__ = ["RedactedSQL", "SQLStatementMode", "redact_sql"]
