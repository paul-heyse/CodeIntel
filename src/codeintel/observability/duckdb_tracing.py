"""DuckDB tracing with SQL statement redaction."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Self, cast

from codeintel.observability.context import get_correlation_id
from codeintel.observability.db_query_parameters import DbQueryParameterConfig
from codeintel.observability.db_query_text import DbQueryTextConfig, DbQueryTextPolicy
from codeintel.observability.db_span_attributes import (
    DbQuerySummaryConfig,
    DbSpanAttributeBuilder,
    DbSpanAttributeConfig,
)
from codeintel.observability.otel import get_observability
from codeintel.observability.sql_redaction import SQLStatementMode

if TYPE_CHECKING:
    from opentelemetry.trace import Span, SpanKind, Tracer

    from codeintel.observability.otel import ObservabilityRuntime
    from codeintel.storage.gateway.config import StorageConfig
    from codeintel.storage.gateway.protocol import DuckDBConnection


try:
    from opentelemetry import trace as otel_trace
    from opentelemetry.trace import SpanKind as _SpanKind
    from opentelemetry.trace.status import Status, StatusCode

    _SPAN_KIND_CLIENT: SpanKind | None = _SpanKind.CLIENT
except ImportError:
    otel_trace = None
    Status = None
    StatusCode = None
    _SPAN_KIND_CLIENT = None

SpanAttributeValue = (
    str | bool | int | float | Sequence[str] | Sequence[bool] | Sequence[int] | Sequence[float]
)


@dataclass(frozen=True, slots=True)
class _TracingConfig:
    tracer: Tracer
    db_name: str
    attributes: Mapping[str, object]
    span_builder: DbSpanAttributeBuilder
    require_parent_span: bool


@dataclass(frozen=True, slots=True)
class _TraceCall:
    statement: object
    func: Callable[..., object]
    args: tuple[object, ...]
    kwargs: Mapping[str, object]
    is_batch: bool
    chain_target: object


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
        call = _TraceCall(
            statement=statement,
            func=self._cursor.execute,
            args=args,
            kwargs=kwargs,
            is_batch=False,
            chain_target=self,
        )
        return self._trace_call(call)

    def executemany(self, statement: object, *args: object, **kwargs: object) -> object:
        call = _TraceCall(
            statement=statement,
            func=self._cursor.executemany,
            args=args,
            kwargs=kwargs,
            is_batch=True,
            chain_target=self,
        )
        return self._trace_call(call)

    def __enter__(self) -> Self:
        self._cursor.__enter__()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> object:
        return self._cursor.__exit__(exc_type, exc, tb)

    def _trace_call(self, call: _TraceCall) -> object:
        return _trace_db_call(call, self._config)


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
        call = _TraceCall(
            statement=statement,
            func=self._connection.execute,
            args=args,
            kwargs=kwargs,
            is_batch=False,
            chain_target=self,
        )
        return self._trace_call(call)

    def executemany(self, statement: object, *args: object, **kwargs: object) -> object:
        call = _TraceCall(
            statement=statement,
            func=self._connection.executemany,
            args=args,
            kwargs=kwargs,
            is_batch=True,
            chain_target=self,
        )
        return self._trace_call(call)

    def cursor(self, *args: object, **kwargs: object) -> _RedactingCursorProxy:
        cursor = self._connection.cursor(*args, **kwargs)
        return _RedactingCursorProxy(cursor, self._config)

    def __enter__(self) -> Self:
        self._connection.__enter__()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> object:
        return self._connection.__exit__(exc_type, exc, tb)

    def _trace_call(self, call: _TraceCall) -> object:
        return _trace_db_call(call, self._config)


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


def _trace_db_call(call: _TraceCall, config: _TracingConfig) -> object:
    if config.require_parent_span and not _has_parent_span():
        return call.func(call.statement, *call.args, **call.kwargs)

    sql_text = _coerce_statement(call.statement)
    params = _extract_params(call.args, call.kwargs)
    spec = config.span_builder.build(
        sql=sql_text,
        params=params,
        db_system_name="duckdb",
        db_namespace=config.db_name,
        is_batch=call.is_batch,
    )

    with _start_span(config.tracer, spec.name) as span:
        _set_span_attributes(span, spec.attributes)
        correlation_id = get_correlation_id()
        if correlation_id:
            span.set_attribute("codeintel.correlation_id", correlation_id)
        _set_span_attributes(span, config.attributes)
        try:
            result = call.func(call.statement, *call.args, **call.kwargs)
        except Exception as exc:  # pragma: no cover - surface via tests
            _record_span_error(span, exc)
            raise

    if result is getattr(call.chain_target, "_cursor", None) or result is getattr(
        call.chain_target, "_connection", None
    ):
        return call.chain_target
    return result


def _set_span_attributes(span: Span, attrs: Mapping[str, object]) -> None:
    for key, value in attrs.items():
        attr_value = _coerce_attribute_value(value)
        if attr_value is not None:
            span.set_attribute(key, attr_value)


def _record_span_error(span: Span, exc: Exception) -> None:
    span.record_exception(exc)
    if Status is not None and StatusCode is not None:
        span.set_status(Status(StatusCode.ERROR))


def _has_parent_span() -> bool:
    if otel_trace is None:
        return False
    span = otel_trace.get_current_span()
    context = span.get_span_context()
    return context.is_valid


def _extract_params(args: tuple[object, ...], kwargs: Mapping[str, object]) -> object | None:
    if "params" in kwargs:
        return kwargs["params"]
    if "parameters" in kwargs:
        return kwargs["parameters"]
    if args:
        return args[0]
    return None


def _coerce_statement(statement: object) -> str:
    if isinstance(statement, bytes):
        return statement.decode("utf-8", "replace")
    return str(statement)


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
        span_builder=_build_span_builder(obs),
        require_parent_span=obs.duckdb_require_parent_span,
    )
    wrapped = _RedactingConnectionProxy(con, tracing_config)
    return cast("DuckDBConnection", wrapped)


def _build_span_builder(obs: ObservabilityRuntime) -> DbSpanAttributeBuilder:
    query_text_policy = _coerce_query_text_policy(obs.duckdb_query_text_policy)
    query_text_config = DbQueryTextConfig(
        policy=query_text_policy,
        max_len=obs.duckdb_query_text_max_len,
        strip_comments=obs.duckdb_query_text_strip_comments,
        collapse_in_lists=obs.duckdb_query_text_collapse_in_lists,
    )
    query_param_config = DbQueryParameterConfig(
        enabled=obs.duckdb_query_parameter_enabled,
        allowed_keys=frozenset(obs.duckdb_query_parameter_keys),
        require_key_in_sql=obs.duckdb_query_parameter_require_in_sql,
        max_string_len=obs.duckdb_query_parameter_max_str_len,
        hash_string_values_for_keys=frozenset(obs.duckdb_query_parameter_hash_keys),
    )
    span_config = DbSpanAttributeConfig(
        statement_mode=cast("SQLStatementMode", obs.duckdb_statement_mode),
        statement_hash_len=obs.duckdb_statement_hash_len,
        emit_legacy_attributes=obs.duckdb_emit_legacy_db_attributes,
        query_summary=DbQuerySummaryConfig(
            max_len=obs.duckdb_query_summary_max_len,
            max_targets=obs.duckdb_query_summary_max_targets,
        ),
        query_text=query_text_config,
        query_parameters=query_param_config,
    )
    return DbSpanAttributeBuilder(span_config)


def _coerce_query_text_policy(value: str) -> DbQueryTextPolicy:
    normalized = value.strip().lower() if value else "never"
    for policy in DbQueryTextPolicy:
        if policy.value == normalized:
            return policy
    return DbQueryTextPolicy.NEVER


__all__ = ["maybe_instrument_duckdb_connection"]
