"""DuckDB tracing with SQL statement redaction."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Self, cast

from codeintel.observability.otel import get_observability
from codeintel.observability.sql_redaction import SQLStatementMode, redact_sql

if TYPE_CHECKING:
    from collections.abc import Mapping

    from opentelemetry.trace import Span, SpanKind, Tracer

    from codeintel.storage.gateway.config import StorageConfig
    from codeintel.storage.gateway.protocol import DuckDBConnection


try:
    from opentelemetry.trace import SpanKind as _SpanKind

    _SPAN_KIND_CLIENT: SpanKind | None = _SpanKind.CLIENT
except ImportError:
    _SPAN_KIND_CLIENT = None

SpanAttributeValue = (
    str
    | bool
    | int
    | float
    | Sequence[str]
    | Sequence[bool]
    | Sequence[int]
    | Sequence[float]
)


@dataclass(frozen=True, slots=True)
class _TracingConfig:
    tracer: Tracer
    db_name: str
    attributes: Mapping[str, object]
    mode: SQLStatementMode
    hash_len: int


class _RedactingCursorProxy:
    def __init__(
        self,
        cursor: DuckDBConnection,
        config: _TracingConfig,
    ) -> None:
        self._cursor = cursor
        self._config = config

    def __getattr__(self, name: str) -> object:
        return getattr(self._cursor, name)

    def execute(self, statement: object, *args: object, **kwargs: object) -> object:
        return self._trace_call(statement, self._cursor.execute, *args, **kwargs)

    def executemany(self, statement: object, *args: object, **kwargs: object) -> object:
        return self._trace_call(statement, self._cursor.executemany, *args, **kwargs)

    def __enter__(self) -> Self:
        self._cursor.__enter__()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> object:
        return self._cursor.__exit__(exc_type, exc, tb)

    def _trace_call(
        self,
        statement: object,
        func: Callable[..., object],
        *args: object,
        **kwargs: object,
    ) -> object:
        redacted = redact_sql(
            str(statement),
            mode=self._config.mode,
            hash_len=self._config.hash_len,
        )
        operation = redacted.operation or "duckdb"
        with _start_span(self._config.tracer, operation) as span:
            span.set_attribute("db.system", "duckdb")
            span.set_attribute("db.name", self._config.db_name)
            if redacted.operation:
                span.set_attribute("db.operation", redacted.operation)
            if redacted.statement_hash:
                span.set_attribute("codeintel.db.statement.sha256", redacted.statement_hash)
            if redacted.display:
                span.set_attribute("db.statement", redacted.display)
            for key, value in self._config.attributes.items():
                attr_value = _coerce_attribute_value(value)
                if attr_value is not None:
                    span.set_attribute(key, attr_value)
            result = func(statement, *args, **kwargs)
        if result is self._cursor:
            return self
        return result


class _RedactingConnectionProxy:
    def __init__(
        self,
        connection: DuckDBConnection,
        config: _TracingConfig,
    ) -> None:
        self._connection = connection
        self._config = config

    def __getattr__(self, name: str) -> object:
        return getattr(self._connection, name)

    def execute(self, statement: object, *args: object, **kwargs: object) -> object:
        return self._trace_call(statement, self._connection.execute, *args, **kwargs)

    def executemany(self, statement: object, *args: object, **kwargs: object) -> object:
        return self._trace_call(statement, self._connection.executemany, *args, **kwargs)

    def cursor(self, *args: object, **kwargs: object) -> _RedactingCursorProxy:
        cursor = self._connection.cursor(*args, **kwargs)
        return _RedactingCursorProxy(cursor, self._config)

    def __enter__(self) -> Self:
        self._connection.__enter__()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> object:
        return self._connection.__exit__(exc_type, exc, tb)

    def _trace_call(
        self,
        statement: object,
        func: Callable[..., object],
        *args: object,
        **kwargs: object,
    ) -> object:
        redacted = redact_sql(
            str(statement),
            mode=self._config.mode,
            hash_len=self._config.hash_len,
        )
        operation = redacted.operation or "duckdb"
        with _start_span(self._config.tracer, operation) as span:
            span.set_attribute("db.system", "duckdb")
            span.set_attribute("db.name", self._config.db_name)
            if redacted.operation:
                span.set_attribute("db.operation", redacted.operation)
            if redacted.statement_hash:
                span.set_attribute("codeintel.db.statement.sha256", redacted.statement_hash)
            if redacted.display:
                span.set_attribute("db.statement", redacted.display)
            for key, value in self._config.attributes.items():
                attr_value = _coerce_attribute_value(value)
                if attr_value is not None:
                    span.set_attribute(key, attr_value)
            result = func(statement, *args, **kwargs)
        if result is self._connection:
            return self
        return result


def _coerce_attribute_value(value: object) -> SpanAttributeValue | None:
    if value is None:
        return None
    if isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, (list, tuple)):
        if all(isinstance(item, (str, bool, int, float)) for item in value):
            return list(value)
        return str(value)
    return str(value)


def _start_span(tracer: Tracer, operation: str) -> AbstractContextManager[Span]:
    if _SPAN_KIND_CLIENT is None:
        return tracer.start_as_current_span(operation)
    return tracer.start_as_current_span(operation, kind=_SPAN_KIND_CLIENT)


def maybe_instrument_duckdb_connection(
    con: DuckDBConnection,
    *,
    config: StorageConfig,
) -> DuckDBConnection:
    """Wrap a DuckDB connection with redacted OpenTelemetry spans when enabled.

    Returns
    -------
    DuckDBConnection
        Instrumented connection when tracing is enabled, otherwise the original connection.
    """
    obs = get_observability()
    if not obs.enabled or obs.tracer is None:
        return con
    if not obs.duckdb_tracing_enabled:
        return con

    tracer = obs.tracer
    attributes = {
        "codeintel.repo": config.repo or "",
        "codeintel.commit": config.commit or "",
        "codeintel.storage.read_only": bool(config.read_only),
    }
    db_name = str(config.db_path)

    tracing_config = _TracingConfig(
        tracer=tracer,
        db_name=db_name,
        attributes=attributes,
        mode=cast("SQLStatementMode", obs.duckdb_statement_mode),
        hash_len=obs.duckdb_statement_hash_len,
    )
    wrapped = _RedactingConnectionProxy(con, tracing_config)
    return cast("DuckDBConnection", wrapped)


__all__ = ["maybe_instrument_duckdb_connection"]
