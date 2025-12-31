"""Polars adapter utilities for DuckDB-backed relation plans."""

from __future__ import annotations

import inspect
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import pyarrow as pa

from codeintel.serving.semantic.engines.protocol import QueryExplain
from codeintel.serving.semantic.guardrails import warn_eager_materialization

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


_STREAMING_ENGINE: PolarsEngineType = "streaming"


class PolarsQueryBuilderError(ValueError):
    """Raised when Polars query construction fails."""


@dataclass(frozen=True, slots=True)
class PolarsExecutablePlan:
    """Executable Polars plan wrapper."""

    relation: DuckDBPyRelation
    lazyframe: PolarsLazyFrame
    settings: ServingSettings
    query_opt_flags: PolarsQueryOptFlags | None = None

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
        batches = _collect_batches(
            self.lazyframe,
            batch_size=batch_size,
            settings=self.settings,
            query_opt_flags=self.query_opt_flags,
        )
        schema = self.lazyframe.collect_schema().to_arrow()
        record_batches = _record_batches_from_frames(
            batches,
            unify_dictionaries=self.settings.polars_unify_dictionaries,
        )
        return pa.RecordBatchReader.from_batches(schema, record_batches)

    def to_table(self) -> pa.Table:
        """Execute the plan and return a fully materialized Arrow table.

        Returns
        -------
        pyarrow.Table
            Materialized Arrow table.

        Raises
        ------
        PolarsQueryBuilderError
            If Polars is unavailable.
        """
        warn_eager_materialization(engine="polars", context="polars_executable_plan")
        if pl is None:  # pragma: no cover
            msg = "polars is required for Polars query execution"
            raise PolarsQueryBuilderError(msg)
        frame = _collect_frame(
            self.lazyframe,
            settings=self.settings,
            query_opt_flags=self.query_opt_flags,
        )
        table = frame.to_arrow()
        if self.settings.polars_unify_dictionaries:
            table = _unify_dictionaries(table)
        return table

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
        return QueryExplain(sql=self.relation.sql_query(), plan=self.relation.explain())

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
        if unify_dictionaries:
            table = _unify_dictionaries(table)
        yield from table.to_batches()


def _collect_batches(
    lazyframe: PolarsLazyFrame,
    *,
    batch_size: int,
    settings: ServingSettings,
    query_opt_flags: PolarsQueryOptFlags | None,
) -> Iterable[PolarsDataFrame]:
    def _collect(*, streaming: bool) -> Iterable[PolarsDataFrame]:
        _maybe_inspect(lazyframe, settings=settings)
        kwargs = _collect_batch_kwargs(
            lazyframe.collect_batches,
            batch_size=batch_size,
            streaming=streaming,
            query_opt_flags=query_opt_flags,
        )
        collect_batches = cast("Callable[..., object]", lazyframe.collect_batches)
        result = collect_batches(**kwargs)
        return cast("Iterable[PolarsDataFrame]", result)

    if settings.polars_streaming:
        try:
            return _collect(streaming=True)
        except PolarsError as exc:
            if settings.polars_streaming_fallback:
                LOG.warning(
                    "Polars streaming collect_batches failed; falling back to eager: %s",
                    exc,
                )
                warn_eager_materialization(
                    engine="polars",
                    context="collect_batches_fallback",
                )
                return _collect(streaming=False)
            raise
    return _collect(streaming=False)


def _collect_frame(
    lazyframe: PolarsLazyFrame,
    *,
    settings: ServingSettings,
    query_opt_flags: PolarsQueryOptFlags | None,
) -> PolarsDataFrame:
    def _collect(*, streaming: bool) -> PolarsDataFrame:
        _maybe_inspect(lazyframe, settings=settings)
        kwargs = _collect_kwargs(
            lazyframe.collect,
            streaming=streaming,
            query_opt_flags=query_opt_flags,
            profile=settings.polars_profile,
        )
        collect = cast("Callable[..., object]", lazyframe.collect)
        result = collect(**kwargs)
        return _unwrap_profile_result(result)

    if settings.polars_streaming:
        try:
            return _collect(streaming=True)
        except PolarsError as exc:
            if settings.polars_streaming_fallback:
                LOG.warning(
                    "Polars streaming collect failed; falling back to eager: %s",
                    exc,
                )
                warn_eager_materialization(
                    engine="polars",
                    context="collect_fallback",
                )
                return _collect(streaming=False)
            raise
    return _collect(streaming=False)


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


def _maybe_inspect(lazyframe: PolarsLazyFrame, *, settings: ServingSettings) -> None:
    if not settings.polars_inspect:
        return
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
    streaming: bool,
    query_opt_flags: PolarsQueryOptFlags | None,
) -> dict[str, object]:
    signature = _signature(func)
    if signature is None:
        return {}
    kwargs: dict[str, object] = {}
    if "chunk_size" in signature.parameters:
        kwargs["chunk_size"] = batch_size
    elif "batch_size" in signature.parameters:
        kwargs["batch_size"] = batch_size
    if "engine" in signature.parameters:
        if streaming:
            kwargs["engine"] = _STREAMING_ENGINE
    elif "streaming" in signature.parameters:
        kwargs["streaming"] = streaming
    if query_opt_flags is not None:
        if "optimization_flags" in signature.parameters:
            kwargs["optimization_flags"] = query_opt_flags
        elif "query_opt_flags" in signature.parameters:
            kwargs["query_opt_flags"] = query_opt_flags
    return kwargs


def _collect_kwargs(
    func: object,
    *,
    streaming: bool,
    query_opt_flags: PolarsQueryOptFlags | None,
    profile: bool,
) -> dict[str, object]:
    signature = _signature(func)
    if signature is None:
        return {}
    kwargs: dict[str, object] = {}
    if "engine" in signature.parameters:
        if streaming:
            kwargs["engine"] = _STREAMING_ENGINE
    elif "streaming" in signature.parameters:
        kwargs["streaming"] = streaming
    if query_opt_flags is not None:
        if "optimization_flags" in signature.parameters:
            kwargs["optimization_flags"] = query_opt_flags
        elif "query_opt_flags" in signature.parameters:
            kwargs["query_opt_flags"] = query_opt_flags
    if profile and "profile" in signature.parameters:
        kwargs["profile"] = True
    return kwargs


def _signature(func: object) -> inspect.Signature | None:
    try:
        return inspect.signature(func)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


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


def _unify_dictionaries(table: pa.Table) -> pa.Table:
    unify = getattr(table, "unify_dictionaries", None)
    if not callable(unify):
        return table
    try:
        return unify()
    except pa.ArrowInvalid:
        return table


def _maybe_to_string(value: object) -> str | None:
    if value is None:
        return None
    text_fn = getattr(value, "to_string", None)
    if callable(text_fn):
        text = text_fn()
        return text if isinstance(text, str) else str(text)
    return str(value)


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
        lazyframe = _relation_to_lazyframe(relation)
        return PolarsExecutablePlan(
            relation=relation,
            lazyframe=lazyframe,
            settings=self.settings,
            query_opt_flags=query_opt_flags,
        )


_PROFILE_TUPLE_SIZE = 2


__all__ = ["PolarsExecutablePlan", "PolarsPlanAdapter"]
