"""SQL statement redaction utilities for telemetry."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Literal, overload

SQLStatementMode = Literal["full", "hash", "operation", "none"]


_LEADING_WS_RE = re.compile(r"^\s+")
_SQL_BLOCK_COMMENT_RE = re.compile(r"/\*.*?\*/", re.DOTALL)
_SQL_LINE_COMMENT_RE = re.compile(r"--[^\n]*")
_SQL_SINGLE_QUOTED_STRING_RE = re.compile(r"'(?:''|[^'])*'")
_SQL_HEX_LITERAL_RE = re.compile(r"\b0x[0-9a-fA-F]+\b")
_SQL_UUID_RE = re.compile(
    r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
)
_SQL_NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?\b")
_WS_RE = re.compile(r"\s+")


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


def _normalize_for_hash(sql: str) -> str:
    sql = _strip_comments(sql)
    sql = _SQL_SINGLE_QUOTED_STRING_RE.sub("?", sql)
    sql = _SQL_HEX_LITERAL_RE.sub("?", sql)
    sql = _SQL_UUID_RE.sub("?", sql)
    sql = _SQL_NUMBER_RE.sub("?", sql)
    return _WS_RE.sub(" ", sql).strip()


def _sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


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

    normalized = _normalize_for_hash(text)
    digest = _sha256_hex(normalized) if normalized else None

    if mode == "none":
        return RedactedSQL(mode=mode, operation=operation, statement_hash=digest, display="")

    if mode == "operation":
        return RedactedSQL(mode=mode, operation=operation, statement_hash=digest, display=operation)

    prefix = (digest or "")[: max(0, hash_len)]
    display = f"{operation} [sha256:{prefix}]" if operation and prefix else operation
    return RedactedSQL(mode="hash", operation=operation, statement_hash=digest, display=display)


__all__ = ["RedactedSQL", "SQLStatementMode", "redact_sql"]
