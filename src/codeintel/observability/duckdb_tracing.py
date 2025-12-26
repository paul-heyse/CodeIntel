"""DuckDB tracing with SQL statement redaction."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Self, cast

from codeintel.observability.db_span_emitter import DbSpanEmitter, DbSpanEmitterConfig
from codeintel.observability.db_tracing import (
    DbQueryParameterConfig,
    DbQuerySummaryConfig,
    DbQueryTextConfig,
    DbQueryTextPolicy,
    DbSpanAttributeBuilder,
    DbTracingPolicy,
    SqlRedactionPolicy,
    SQLStatementMode,
)
from codeintel.observability.runtime import get_observability
from codeintel.observability.semconv_keys import (
    CODEINTEL_COMMIT,
    CODEINTEL_REPO,
    CODEINTEL_STORAGE_READ_ONLY,
)

if TYPE_CHECKING:
    from codeintel.observability.runtime import DbTracingConfig
    from codeintel.storage.gateway.config import StorageConfig
    from codeintel.storage.gateway.protocol import DuckDBConnection


@dataclass(frozen=True, slots=True)
class _TracingConfig:
    emitter: DbSpanEmitter


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


def _trace_db_call(call: _TraceCall, config: _TracingConfig) -> object:
    sql_text = _coerce_statement(call.statement)
    params = _extract_params(call.args, call.kwargs)
    result = config.emitter.trace_call(
        sql=sql_text,
        params=params,
        is_batch=call.is_batch,
        call=lambda: call.func(call.statement, *call.args, **call.kwargs),
    )

    if result is getattr(call.chain_target, "_cursor", None) or result is getattr(
        call.chain_target, "_connection", None
    ):
        return call.chain_target
    return result


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
    if not obs.db_tracing.enabled:
        return con

    tracer = obs.tracer
    policy = _policy_from_config(obs.db_tracing)
    attributes = {
        CODEINTEL_REPO: config.repo or "",
        CODEINTEL_COMMIT: config.commit or "",
        CODEINTEL_STORAGE_READ_ONLY: bool(config.read_only),
    }
    db_name = str(config.db_path)

    emitter_config = DbSpanEmitterConfig(
        tracer=tracer,
        db_system_name="duckdb",
        db_namespace=db_name,
        attributes=attributes,
        span_builder=_build_span_builder(policy),
        require_parent_span=policy.require_parent_span,
        policy=obs.policy,
    )
    tracing_config = _TracingConfig(emitter=DbSpanEmitter(emitter_config))
    wrapped = _RedactingConnectionProxy(con, tracing_config)
    return cast("DuckDBConnection", wrapped)


def _build_span_builder(policy: DbTracingPolicy) -> DbSpanAttributeBuilder:
    span_config = policy.redaction.to_span_attribute_config(summary=policy.query_summary)
    return DbSpanAttributeBuilder(span_config)


def _policy_from_config(db_config: DbTracingConfig) -> DbTracingPolicy:
    query_text_policy = _coerce_query_text_policy(db_config.query_text_policy)
    query_text_config = DbQueryTextConfig(
        policy=query_text_policy,
        max_len=db_config.query_text_max_len,
        strip_comments=db_config.query_text_strip_comments,
        collapse_in_lists=db_config.query_text_collapse_in_lists,
    )
    query_param_config = DbQueryParameterConfig(
        enabled=db_config.query_parameter_enabled,
        allowed_keys=frozenset(db_config.query_parameter_keys),
        require_key_in_sql=db_config.query_parameter_require_in_sql,
        max_string_len=db_config.query_parameter_max_str_len,
        hash_string_values_for_keys=frozenset(db_config.query_parameter_hash_keys),
        hash_len=db_config.statement_hash_len,
    )
    query_summary_config = DbQuerySummaryConfig(
        max_len=db_config.query_summary_max_len,
        max_targets=db_config.query_summary_max_targets,
        emit_ellipsis=db_config.query_summary_emit_ellipsis,
        hash_suspicious_targets=db_config.query_summary_hash_suspicious_targets,
        hash_target_len=db_config.query_summary_hash_len,
        hash_target_min_len=db_config.query_summary_hash_min_len,
        include_subquery_operations=db_config.query_summary_include_subquery_operations,
        include_multi_statement=db_config.query_summary_include_multi_statement,
    )
    redaction = SqlRedactionPolicy(
        statement_mode=cast("SQLStatementMode", db_config.statement_mode),
        statement_hash_len=db_config.statement_hash_len,
        query_text=query_text_config,
        query_parameters=query_param_config,
    )
    return DbTracingPolicy(
        enabled=db_config.enabled,
        require_parent_span=db_config.require_parent_span,
        query_summary=query_summary_config,
        redaction=redaction,
    )


def _coerce_query_text_policy(value: str) -> DbQueryTextPolicy:
    normalized = value.strip().lower() if value else "never"
    for policy in DbQueryTextPolicy:
        if policy.value == normalized:
            return policy
    return DbQueryTextPolicy.NEVER


__all__ = ["maybe_instrument_duckdb_connection"]
