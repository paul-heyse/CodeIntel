"""Shared DB span attribute composition."""

from __future__ import annotations

from dataclasses import dataclass, field

from codeintel.observability.db_query_parameters import (
    DbQueryParameterConfig,
    emit_db_query_parameters,
)
from codeintel.observability.db_query_text import (
    DbQueryTextConfig,
    DbQueryTextPolicy,
    looks_parameterized,
    redact_sql_literals_with_sqlglot,
)
from codeintel.observability.sql_redaction import SQLStatementMode, redact_sql
from codeintel.storage.sqlglot_tools import QuerySummaryConfig, summarize_sql_duckdb

DbQuerySummaryConfig = QuerySummaryConfig


@dataclass(frozen=True, slots=True)
class DbSpanAttributeConfig:
    """Configuration for building DB span attributes."""

    statement_mode: SQLStatementMode = "hash"
    statement_hash_len: int = 16
    emit_legacy_attributes: bool = False
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
        """Initialize the builder with a span attribute configuration.

        Parameters
        ----------
        config
            Configuration values for span attribute composition.
        """
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

        Parameters
        ----------
        sql
            SQL statement text.
        params
            Query parameters supplied to the database driver.
        db_system_name
            Database system identifier.
        db_namespace
            Database namespace or name, if available.
        is_batch
            Whether the operation is a batch execution.

        Returns
        -------
        DbSpanSpec
            Span name and attribute mapping for the operation.

        Examples
        --------
        >>> config = DbSpanAttributeConfig()
        >>> builder = DbSpanAttributeBuilder(config)
        >>> spec = builder.build(
        ...     sql="SELECT 1",
        ...     params=None,
        ...     db_system_name="duckdb",
        ...     db_namespace=None,
        ... )
        >>> spec.name == "duckdb"
        True

        Notes
        -----
        The result includes only attributes allowed by the configured policies.
        """
        summary = summarize_sql_duckdb(sql, config=self._config.query_summary)
        redacted = redact_sql(
            sql,
            mode=self._config.statement_mode,
            hash_len=self._config.statement_hash_len,
        )

        attrs: dict[str, object] = {
            "db.system.name": db_system_name,
        }
        if db_namespace:
            attrs["db.namespace"] = db_namespace
        if summary:
            attrs["db.query.summary"] = summary
        if redacted.statement_hash:
            attrs["codeintel.db.statement.sha256"] = redacted.statement_hash
        if redacted.display:
            attrs["db.statement"] = redacted.display

        if self._config.emit_legacy_attributes:
            attrs.setdefault("db.system", db_system_name)
            if db_namespace:
                attrs.setdefault("db.name", db_namespace)

        query_text = self._maybe_query_text(
            sql=sql,
            params=params,
            db_system_name=db_system_name,
        )
        if query_text:
            attrs["db.query.text"] = query_text

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


__all__ = [
    "DbQuerySummaryConfig",
    "DbSpanAttributeBuilder",
    "DbSpanAttributeConfig",
    "DbSpanSpec",
]
