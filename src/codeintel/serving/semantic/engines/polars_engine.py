"""Polars adapter utilities for DuckDB-backed relation plans."""

from __future__ import annotations

import inspect
import logging
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, cast

import pyarrow as pa

from codeintel.core.columnar.compute_helpers import combine_table_chunks
from codeintel.core.columnar.conversion import record_batch_reader_from_iterable
from codeintel.core.columnar.normalization import normalize_table_for_compute
from codeintel.core.columnar.readers import empty_reader_from_schema
from codeintel.serving.semantic.engines.protocol import QueryExplain
from codeintel.serving.semantic.guardrails import warn_eager_materialization
from codeintel.storage.duckdb_explain import normalize_explain_output

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator
    from typing import Literal

    from duckdb import DuckDBPyRelation
    from polars import DataFrame, LazyFrame, QueryOptFlags

    from codeintel.serving.settings import ServingSettings

    type PolarsDataFrame = DataFrame
    type PolarsLazyFrame = LazyFrame
    type PolarsQueryOptFlags = QueryOptFlags
    type PolarsEngineType = Literal["auto", "in-memory", "streaming", "gpu"]
else:
    type PolarsDataFrame = object
    type PolarsLazyFrame = object
    type PolarsQueryOptFlags = object
    type PolarsEngineType = str

try:
    import polars as pl
except ImportError:  # pragma: no cover
    pl = None
try:
    from polars.exceptions import PolarsError
except ImportError:  # pragma: no cover
    PolarsError = Exception

LOG = logging.getLogger(__name__)


_DEFAULT_ENGINE: PolarsEngineType = "auto"
_STREAMING_ENGINE: PolarsEngineType = "streaming"


@dataclass(frozen=True, slots=True)
class _PolarsExecutionConfig:
    engine: PolarsEngineType
    batch_size: int
    streaming: bool
    streaming_fallback: bool
    maintain_order: bool
    sink_batches: bool
    collect_all: bool
    profile: bool
    inspect: bool
    collect_schema: bool
    unify_dictionaries: bool
    query_opt_flags: PolarsQueryOptFlags | None


class PolarsQueryBuilderError(ValueError):
    """Raised when Polars query construction fails."""


@dataclass(frozen=True, slots=True)
class PolarsExecutablePlan:
    """Executable Polars plan wrapper."""

    relation: DuckDBPyRelation
    lazyframe: PolarsLazyFrame
    execution: _PolarsExecutionConfig

    def to_reader(self, *, batch_size: int) -> pa.RecordBatchReader:
        """Return an Arrow RecordBatchReader for the plan results.

        Parameters
        ----------
        batch_size
            Max batches per chunk in the returned reader.

        Returns
        -------
        pyarrow.RecordBatchReader
            Reader over the plan output.

        Raises
        ------
        PolarsQueryBuilderError
            If Polars is unavailable.
        """
        if pl is None:  # pragma: no cover
            msg = "polars is required for Polars query execution"
            raise PolarsQueryBuilderError(msg)
        execution = self.execution
        batches = _collect_frames(
            self.lazyframe,
            batch_size=batch_size,
            execution=execution,
        )
        schema = self.lazyframe.collect_schema()
        if execution.collect_schema:
            _log_schema(schema)
        arrow_schema = schema.to_arrow()
        record_batches = _record_batches_from_frames(
            batches,
            unify_dictionaries=execution.unify_dictionaries,
        )
        reader = record_batch_reader_from_iterable(record_batches, empty_policy="none")
        if reader is None:
            return empty_reader_from_schema(arrow_schema)
        return reader

    def explain(self) -> QueryExplain:
        """Return the DuckDB relation explain plan.

        Returns
        -------
        QueryExplain
            Explain payload with DuckDB SQL and plan text.

        Raises
        ------
        PolarsQueryBuilderError
            If Polars is unavailable.
        """
        if pl is None:  # pragma: no cover
            msg = "polars is required for Polars query execution"
            raise PolarsQueryBuilderError(msg)
        plan = normalize_explain_output(self.relation.explain())
        return QueryExplain(sql=self.relation.sql_query(), plan=plan)

    @staticmethod
    def cleanup() -> None:
        """Release temporary resources after execution."""
        return


def _record_batches_from_frames(
    frames: Iterable[PolarsDataFrame],
    *,
    unify_dictionaries: bool,
) -> Iterator[pa.RecordBatch]:
    for frame in frames:
        table = frame.to_arrow()
        table = combine_table_chunks(table)
        if unify_dictionaries:
            table = normalize_table_for_compute(table)
        yield from table.to_batches()


def _execution_config(
    *,
    settings: ServingSettings,
    query_opt_flags: PolarsQueryOptFlags | None,
) -> _PolarsExecutionConfig:
    engine = _STREAMING_ENGINE if settings.polars_streaming else _DEFAULT_ENGINE
    return _PolarsExecutionConfig(
        engine=engine,
        batch_size=settings.export_batch_size,
        streaming=settings.polars_streaming,
        streaming_fallback=settings.polars_streaming_fallback,
        maintain_order=settings.polars_maintain_order,
        sink_batches=settings.polars_sink_batches,
        collect_all=settings.polars_collect_all,
        profile=settings.polars_profile,
        inspect=settings.polars_inspect,
        collect_schema=settings.polars_collect_schema,
        unify_dictionaries=settings.polars_unify_dictionaries,
        query_opt_flags=query_opt_flags,
    )


def _fallback_execution(execution: _PolarsExecutionConfig) -> _PolarsExecutionConfig:
    if not execution.streaming:
        return execution
    return replace(execution, streaming=False, engine=_DEFAULT_ENGINE)


def _collect_frames(
    lazyframe: PolarsLazyFrame,
    *,
    batch_size: int,
    execution: _PolarsExecutionConfig,
) -> Iterable[PolarsDataFrame]:
    if execution.streaming:
        try:
            return _collect_batches(
                lazyframe,
                batch_size=batch_size,
                execution=execution,
            )
        except PolarsError as exc:
            if execution.streaming_fallback:
                LOG.warning(
                    "Polars streaming collect_batches failed; falling back to eager: %s",
                    exc,
                )
                warn_eager_materialization(
                    engine="polars",
                    context="collect_batches_fallback",
                )
                fallback = _fallback_execution(execution)
                return (_collect_frame(lazyframe, execution=fallback),)
            raise
    return (_collect_frame(lazyframe, execution=execution),)


def _collect_batches(
    lazyframe: PolarsLazyFrame,
    *,
    batch_size: int,
    execution: _PolarsExecutionConfig,
) -> Iterable[PolarsDataFrame]:
    _log_plan_diagnostics(lazyframe, execution=execution)
    if execution.sink_batches:
        return _sink_batches(
            lazyframe,
            batch_size=batch_size,
            execution=execution,
        )
    return _collect_batches_direct(
        lazyframe,
        batch_size=batch_size,
        execution=execution,
    )


def _collect_batches_direct(
    lazyframe: PolarsLazyFrame,
    *,
    batch_size: int,
    execution: _PolarsExecutionConfig,
) -> Iterable[PolarsDataFrame]:
    kwargs = _collect_batch_kwargs(
        lazyframe.collect_batches,
        batch_size=batch_size,
        execution=execution,
    )
    collect_batches = cast("Callable[..., object]", lazyframe.collect_batches)
    result = collect_batches(**kwargs)
    return cast("Iterable[PolarsDataFrame]", result)


def _collect_frame(
    lazyframe: PolarsLazyFrame,
    *,
    execution: _PolarsExecutionConfig,
) -> PolarsDataFrame:
    _log_plan_diagnostics(lazyframe, execution=execution)
    if execution.collect_all and execution.profile:
        LOG.warning("polars_collect_all ignored because polars_profile is enabled")
    if execution.collect_all and pl is not None and not execution.profile:
        collect_all = getattr(pl, "collect_all", None)
        if callable(collect_all):
            kwargs = _collect_all_kwargs(
                collect_all,
                execution=execution,
            )
            result = collect_all([lazyframe], **kwargs)
            if isinstance(result, list) and result:
                return cast("PolarsDataFrame", result[0])
    kwargs = _collect_kwargs(
        lazyframe.collect,
        execution=execution,
        profile=execution.profile,
    )
    collect = cast("Callable[..., object]", lazyframe.collect)
    result = collect(**kwargs)
    return _unwrap_profile_result(result)


def _unwrap_profile_result(result: object) -> PolarsDataFrame:
    if isinstance(result, tuple) and len(result) == _PROFILE_TUPLE_SIZE:
        frame, profile = result
        _log_profile(profile)
        return cast("PolarsDataFrame", frame)
    return cast("PolarsDataFrame", result)


def _log_profile(profile: object) -> None:
    profile_repr = _maybe_to_string(profile)
    if profile_repr is None:
        return
    LOG.info("polars_profile %s", profile_repr)


def _log_plan_diagnostics(
    lazyframe: PolarsLazyFrame,
    *,
    execution: _PolarsExecutionConfig,
) -> None:
    if not execution.inspect:
        return
    _maybe_inspect(lazyframe)
    explain = _polars_explain(lazyframe, execution=execution)
    if explain is not None:
        LOG.debug("polars_explain %s", explain)


def _polars_explain(
    lazyframe: PolarsLazyFrame,
    *,
    execution: _PolarsExecutionConfig,
) -> str | None:
    explain_fn = getattr(lazyframe, "explain", None)
    if not callable(explain_fn):
        return None
    kwargs = _plan_kwargs(
        explain_fn,
        execution=execution,
        optimized=True,
    )
    try:
        result = explain_fn(**kwargs)
    except PolarsError:
        return None
    return result if isinstance(result, str) else None


def _plan_kwargs(
    func: object,
    *,
    execution: _PolarsExecutionConfig,
    optimized: bool,
) -> dict[str, object]:
    signature = _signature(func)
    if signature is None:
        return {}
    kwargs: dict[str, object] = {}
    if "optimized" in signature.parameters:
        kwargs["optimized"] = optimized
    _apply_engine_kwargs(
        signature,
        kwargs=kwargs,
        engine=execution.engine,
        streaming=execution.streaming,
    )
    _apply_query_opt_kwargs(
        signature,
        kwargs=kwargs,
        query_opt_flags=execution.query_opt_flags,
    )
    return kwargs


def _maybe_inspect(lazyframe: PolarsLazyFrame) -> None:
    inspect_fn = getattr(lazyframe, "inspect", None)
    if not callable(inspect_fn):
        return
    try:
        inspect_fn()
    except PolarsError:
        LOG.warning("Polars inspect failed; continuing without inspect.")


def _collect_batch_kwargs(
    func: object,
    *,
    batch_size: int,
    execution: _PolarsExecutionConfig,
) -> dict[str, object]:
    signature = _signature(func)
    if signature is None:
        return {}
    kwargs: dict[str, object] = {}
    batch_param = _first_supported_param(signature, ("chunk_size", "batch_size"))
    if batch_param is not None:
        kwargs[batch_param] = batch_size
    _apply_engine_kwargs(
        signature,
        kwargs=kwargs,
        engine=execution.engine,
        streaming=execution.streaming,
    )
    _apply_query_opt_kwargs(
        signature,
        kwargs=kwargs,
        query_opt_flags=execution.query_opt_flags,
    )
    _apply_maintain_order_kwargs(
        signature,
        kwargs=kwargs,
        maintain_order=execution.maintain_order,
    )
    return kwargs


def _collect_kwargs(
    func: object,
    *,
    execution: _PolarsExecutionConfig,
    profile: bool,
) -> dict[str, object]:
    signature = _signature(func)
    if signature is None:
        return {}
    kwargs: dict[str, object] = {}
    _apply_engine_kwargs(
        signature,
        kwargs=kwargs,
        engine=execution.engine,
        streaming=execution.streaming,
    )
    _apply_query_opt_kwargs(
        signature,
        kwargs=kwargs,
        query_opt_flags=execution.query_opt_flags,
    )
    if profile and "profile" in signature.parameters:
        kwargs["profile"] = True
    return kwargs


def _collect_all_kwargs(
    func: object,
    *,
    execution: _PolarsExecutionConfig,
) -> dict[str, object]:
    signature = _signature(func)
    if signature is None:
        return {}
    kwargs: dict[str, object] = {}
    _apply_engine_kwargs(
        signature,
        kwargs=kwargs,
        engine=execution.engine,
        streaming=execution.streaming,
    )
    _apply_query_opt_kwargs(
        signature,
        kwargs=kwargs,
        query_opt_flags=execution.query_opt_flags,
    )
    return kwargs


def _sink_batches(
    lazyframe: PolarsLazyFrame,
    *,
    batch_size: int,
    execution: _PolarsExecutionConfig,
) -> Iterable[PolarsDataFrame]:
    sinker = getattr(lazyframe, "sink_batches", None)
    if not callable(sinker):
        return _collect_batches_direct(
            lazyframe,
            batch_size=batch_size,
            execution=execution,
        )
    batches: list[PolarsDataFrame] = []

    def _callback(batch: PolarsDataFrame) -> bool | None:
        batches.append(batch)
        return None

    kwargs = _collect_batch_kwargs(
        sinker,
        batch_size=batch_size,
        execution=execution,
    )
    sinker(_callback, **kwargs)
    return batches


def _signature(func: object) -> inspect.Signature | None:
    try:
        return inspect.signature(func)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _first_supported_param(
    signature: inspect.Signature,
    names: tuple[str, ...],
) -> str | None:
    for name in names:
        if name in signature.parameters:
            return name
    return None


def _apply_engine_kwargs(
    signature: inspect.Signature,
    *,
    kwargs: dict[str, object],
    engine: PolarsEngineType,
    streaming: bool,
) -> None:
    if "engine" in signature.parameters:
        kwargs["engine"] = engine
        return
    if "streaming" in signature.parameters:
        kwargs["streaming"] = streaming


def _apply_query_opt_kwargs(
    signature: inspect.Signature,
    *,
    kwargs: dict[str, object],
    query_opt_flags: PolarsQueryOptFlags | None,
) -> None:
    if query_opt_flags is None:
        return
    opt_param = _first_supported_param(
        signature,
        ("optimizations", "optimization_flags", "query_opt_flags"),
    )
    if opt_param is not None:
        kwargs[opt_param] = query_opt_flags


def _apply_maintain_order_kwargs(
    signature: inspect.Signature,
    *,
    kwargs: dict[str, object],
    maintain_order: bool,
) -> None:
    if "maintain_order" in signature.parameters:
        kwargs["maintain_order"] = maintain_order


def _resolve_query_opt_flags(flags: tuple[str, ...]) -> PolarsQueryOptFlags | None:
    if pl is None or not flags:
        return None
    opt_flags = getattr(pl, "QueryOptFlags", None)
    if opt_flags is None:
        return None
    resolved: PolarsQueryOptFlags | None = None
    for raw_flag in flags:
        name = raw_flag.upper()
        candidate = getattr(opt_flags, name, None)
        if candidate is None:
            candidate = getattr(opt_flags, raw_flag, None)
        if candidate is None:
            LOG.debug("Unknown Polars QueryOptFlags value: %s", raw_flag)
            continue
        resolved_flag = cast("PolarsQueryOptFlags", candidate)
        if resolved is None:
            resolved = resolved_flag
            continue
        or_fn = getattr(resolved, "__or__", None)
        if callable(or_fn):
            resolved = cast("PolarsQueryOptFlags", or_fn(resolved_flag))
        else:
            resolved = resolved_flag
    return resolved


def _maybe_to_string(value: object) -> str | None:
    if value is None:
        return None
    text_fn = getattr(value, "to_string", None)
    if callable(text_fn):
        text = text_fn()
        return text if isinstance(text, str) else str(text)
    return str(value)


def _log_schema(schema: object) -> None:
    schema_repr = _maybe_to_string(schema)
    if schema_repr is None:
        return
    LOG.info("polars_schema %s", schema_repr)


def _relation_to_lazyframe(relation: DuckDBPyRelation) -> PolarsLazyFrame:
    if pl is None:  # pragma: no cover
        msg = "polars is required for Polars query execution"
        raise PolarsQueryBuilderError(msg)
    to_polars = getattr(relation, "pl", None)
    if not callable(to_polars):
        msg = "DuckDB relation does not support Polars conversion"
        raise PolarsQueryBuilderError(msg)
    try:
        result = to_polars(lazy=True)
    except TypeError:
        result = to_polars()
    if isinstance(result, pl.LazyFrame):
        return result
    if isinstance(result, pl.DataFrame):
        return result.lazy()
    msg = f"Unexpected Polars relation conversion type: {type(result)}"
    raise PolarsQueryBuilderError(msg)


@dataclass(frozen=True, slots=True)
class PolarsPlanAdapter:
    """Adapter that converts DuckDB relations into Polars execution plans."""

    settings: ServingSettings

    def build(self, *, relation: DuckDBPyRelation) -> PolarsExecutablePlan:
        """Wrap a DuckDB relation in a PolarsExecutablePlan.

        Parameters
        ----------
        relation
            DuckDB relation to adapt for Polars execution.

        Returns
        -------
        PolarsExecutablePlan
            Polars plan backed by the DuckDB relation.
        """
        query_opt_flags = _resolve_query_opt_flags(self.settings.polars_query_opt_flags)
        execution = _execution_config(settings=self.settings, query_opt_flags=query_opt_flags)
        lazyframe = _relation_to_lazyframe(relation)
        return PolarsExecutablePlan(
            relation=relation,
            lazyframe=lazyframe,
            execution=execution,
        )


_PROFILE_TUPLE_SIZE = 2


__all__ = ["PolarsExecutablePlan", "PolarsPlanAdapter"]
