"""Database tracing helpers for observability."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Literal, cast

from sqlglot import exp, parse_one
from sqlglot.errors import ParseError, SqlglotError

from codeintel.observability.attribute_sanitizer import truncate_str
from codeintel.observability.semconv_keys import (
    CODEINTEL_DB_STATEMENT_SHA256,
    DB_NAMESPACE,
    DB_QUERY_PARAMETER_PREFIX,
    DB_QUERY_SUMMARY,
    DB_QUERY_TEXT,
    DB_STATEMENT,
    DB_SYSTEM_NAME,
)
from codeintel.storage.sqlglot_tools import (
    QuerySummaryConfig,
    fingerprint_sql_duckdb_safe,
    summarize_sql_duckdb,
)

SQLStatementMode = Literal["full", "hash", "operation", "none"]

_DUCKDB_PLACEHOLDER_RE = re.compile(r"(\?)|(\$[0-9]+)|(\$[A-Za-z_][A-Za-z0-9_]*)")
_GENERIC_PLACEHOLDER_RE = re.compile(r"(\?)|(\$[0-9]+)|(:[A-Za-z_][A-Za-z0-9_]*)")
_BLOCK_COMMENT_RE = re.compile(r"/\*.*?\*/", re.DOTALL)
_LINE_COMMENT_RE = re.compile(r"--[^\n]*")
_IN_LIST_RE = re.compile(
    r"\bIN\s*\(\s*(\?\s*,\s*){2,}\?\s*\)",
    flags=re.IGNORECASE,
)
_DUCKDB_NAMED_PARAM_RE = re.compile(r"\$([A-Za-z_][A-Za-z0-9_]*)")

_BOOLEAN_EXPR = cast("type[exp.Expression] | None", getattr(exp, "Boolean", None))
_INTERVAL_EXPR = cast("type[exp.Expression] | None", getattr(exp, "Interval", None))
_HEX_STRING_EXPR = cast("type[exp.Expression] | None", getattr(exp, "HexString", None))
_PLACEHOLDER_EXPR = cast("type[exp.Expression] | None", getattr(exp, "Placeholder", None))


@dataclass(frozen=True, slots=True)
class RedactedSQL:
    """Result of applying a redaction policy to a SQL statement."""

    mode: SQLStatementMode
    operation: str
    statement_hash: str | None
    display: str


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
        Redacted SQL representation for telemetry.
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

    Returns
    -------
    bool
        True when placeholders are detected.
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

    Returns
    -------
    str | None
        Redacted SQL text, or None when parsing fails.
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
    return truncate_str(rendered, config.max_len)


@dataclass(frozen=True, slots=True)
class DbQueryParameterConfig:
    """Configuration for db.query.parameter emission."""

    enabled: bool = False
    allowed_keys: frozenset[str] = frozenset()
    require_key_in_sql: bool = True
    max_string_len: int = 80
    hash_string_values_for_keys: frozenset[str] = frozenset()
    hash_len: int = 16
    disable_on_batch: bool = True

    def is_effectively_enabled(self) -> bool:
        """Return whether parameter emission is configured and allowed.

        Returns
        -------
        bool
            True when emission is enabled and allowlist is non-empty.
        """
        return self.enabled and bool(self.allowed_keys)


def emit_db_query_parameters(
    *,
    sql: str,
    params: object | None,
    db_system_name: str,
    config: DbQueryParameterConfig,
    is_batch: bool = False,
) -> dict[str, str | bool | int | float]:
    """Emit db.query.parameter attributes for allowlisted parameters.

    Returns
    -------
    dict[str, str | bool | int | float]
        Mapping of attribute keys to scalar values.
    """
    if not config.is_effectively_enabled():
        return {}
    if config.disable_on_batch and is_batch:
        return {}

    params_map = _normalize_params(params)
    if params_map is None:
        return {}

    keys_in_sql = _resolve_keys_in_sql(sql, db_system_name=db_system_name, config=config)
    return _build_param_attrs(params_map, keys_in_sql, config=config)


DbQuerySummaryConfig = QuerySummaryConfig


@dataclass(frozen=True, slots=True)
class DbSpanAttributeConfig:
    """Configuration for building DB span attributes."""

    statement_mode: SQLStatementMode = "hash"
    statement_hash_len: int = 16
    query_summary: DbQuerySummaryConfig = field(default_factory=DbQuerySummaryConfig)
    query_text: DbQueryTextConfig = field(default_factory=DbQueryTextConfig)
    query_parameters: DbQueryParameterConfig = field(default_factory=DbQueryParameterConfig)


@dataclass(frozen=True, slots=True)
class DbSpanSpec:
    """Span name and attributes for a DB operation."""

    name: str
    attributes: dict[str, object]


class DbSpanAttributeBuilder:
    """Build span attributes for DB spans with safe defaults."""

    def __init__(self, config: DbSpanAttributeConfig) -> None:
        """Initialize the builder with a span attribute configuration."""
        self._config = config

    def build(
        self,
        *,
        sql: str,
        params: object | None,
        db_system_name: str,
        db_namespace: str | None,
        is_batch: bool = False,
    ) -> DbSpanSpec:
        """Build span name and attributes for a database operation.

        Returns
        -------
        DbSpanSpec
            Span name and attributes for the operation.
        """
        summary = summarize_sql_duckdb(sql, config=self._config.query_summary)
        redacted = redact_sql(
            sql,
            mode=self._config.statement_mode,
            hash_len=self._config.statement_hash_len,
        )

        attrs: dict[str, object] = {
            DB_SYSTEM_NAME: db_system_name,
        }
        if db_namespace:
            attrs[DB_NAMESPACE] = db_namespace
        if summary:
            attrs[DB_QUERY_SUMMARY] = summary
        if redacted.statement_hash:
            attrs[CODEINTEL_DB_STATEMENT_SHA256] = redacted.statement_hash
        if redacted.display:
            attrs[DB_STATEMENT] = redacted.display

        query_text = self._maybe_query_text(
            sql=sql,
            params=params,
            db_system_name=db_system_name,
        )
        if query_text:
            attrs[DB_QUERY_TEXT] = query_text

        params_attrs = emit_db_query_parameters(
            sql=sql,
            params=params,
            db_system_name=db_system_name,
            config=self._config.query_parameters,
            is_batch=is_batch,
        )
        if params_attrs:
            attrs.update(params_attrs)

        span_name = summary or db_system_name
        return DbSpanSpec(name=span_name, attributes=attrs)

    def _maybe_query_text(
        self,
        *,
        sql: str,
        params: object | None,
        db_system_name: str,
    ) -> str | None:
        if not sql:
            return None
        policy = self._config.query_text.policy
        if policy == DbQueryTextPolicy.NEVER:
            return None
        if policy == DbQueryTextPolicy.FULL:
            return sql

        result: str | None = None
        if (
            policy
            in {
                DbQueryTextPolicy.PARAMETERIZED,
                DbQueryTextPolicy.PARAMETERIZED_OR_REDACTED,
            }
            and params is not None
        ):
            if looks_parameterized(sql, db_system_name=db_system_name):
                result = sql
            elif policy == DbQueryTextPolicy.PARAMETERIZED:
                result = None

        if result is None and policy in {
            DbQueryTextPolicy.REDACTED,
            DbQueryTextPolicy.PARAMETERIZED_OR_REDACTED,
        }:
            dialect = _dialect_for_system(db_system_name)
            result = redact_sql_literals_with_sqlglot(
                sql,
                dialect=dialect,
                config=self._config.query_text,
            )

        return result


def _dialect_for_system(db_system_name: str) -> str | None:
    system = (db_system_name or "").lower()
    if system == "duckdb":
        return "duckdb"
    if system in {"postgres", "postgresql", "cockroachdb"}:
        return "postgres"
    if system in {"mysql", "mariadb"}:
        return "mysql"
    if system in {"sqlite", "sqlite3"}:
        return "sqlite"
    return None


def _to_text(statement: str | bytes) -> str:
    if isinstance(statement, bytes):
        return statement.decode("utf-8", "replace")
    return statement


def _strip_comments(sql: str) -> str:
    return _LINE_COMMENT_RE.sub(" ", _BLOCK_COMMENT_RE.sub(" ", sql))


def _extract_operation(sql: str) -> str:
    sql = _strip_comments(sql)
    sql = re.sub(r"^\s+", "", sql)
    head = sql.split(maxsplit=1)
    return head[0] if head else ""


def _collapse_in_lists(sql: str) -> str:
    return _IN_LIST_RE.sub("IN (?)", sql)


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


def _normalize_params(params: object | None) -> Mapping[str, object] | None:
    if not isinstance(params, Mapping):
        return None
    for key in params:
        if not isinstance(key, str):
            return None
    return params


def _resolve_keys_in_sql(
    sql: str,
    *,
    db_system_name: str,
    config: DbQueryParameterConfig,
) -> set[str] | None:
    if not config.require_key_in_sql:
        return None
    keys_in_sql = _extract_named_param_keys(sql, db_system_name=db_system_name)
    if not keys_in_sql:
        return set()
    return keys_in_sql


def _build_param_attrs(
    params: Mapping[str, object],
    keys_in_sql: set[str] | None,
    *,
    config: DbQueryParameterConfig,
) -> dict[str, str | bool | int | float]:
    attrs: dict[str, str | bool | int | float] = {}
    for key in config.allowed_keys:
        if key not in params:
            continue
        if keys_in_sql is not None and key not in keys_in_sql:
            continue
        raw = _coerce_scalar(params[key], max_string_len=config.max_string_len)
        if raw is None:
            continue
        if isinstance(raw, str) and key in config.hash_string_values_for_keys:
            raw = _hash_str(raw, config.hash_len)
        attrs[f"{DB_QUERY_PARAMETER_PREFIX}{key}"] = raw
    return attrs


def _extract_named_param_keys(sql: str, *, db_system_name: str) -> set[str]:
    system = (db_system_name or "").lower()
    if system == "duckdb":
        return set(_DUCKDB_NAMED_PARAM_RE.findall(sql))
    return set()


def _coerce_scalar(value: object, *, max_string_len: int) -> str | bool | int | float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float):
        return value
    if isinstance(value, str):
        return truncate_str(value, max_string_len)
    return None


def _hash_str(value: str, hash_len: int) -> str:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    if hash_len <= 0:
        return ""
    return digest[:hash_len]


__all__ = [
    "DbQueryParameterConfig",
    "DbQuerySummaryConfig",
    "DbQueryTextConfig",
    "DbQueryTextPolicy",
    "DbSpanAttributeBuilder",
    "DbSpanAttributeConfig",
    "DbSpanSpec",
    "RedactedSQL",
    "SQLStatementMode",
    "emit_db_query_parameters",
    "looks_parameterized",
    "redact_sql",
    "redact_sql_literals_with_sqlglot",
]
