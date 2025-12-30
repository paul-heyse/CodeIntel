"""Polars-based semantic query engine."""

from __future__ import annotations

import inspect
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import pyarrow as pa

from codeintel.core.columnar.tabular_adapter import to_lazyframe
from codeintel.core.iceberg.guardrails import iceberg_enforced_table, require_iceberg_read
from codeintel.serving.semantic.engines.protocol import EngineContext, ExecutablePlan, QueryExplain
from codeintel.serving.semantic.guardrails import (
    warn_eager_materialization,
    warn_schema_drift_observed,
)
from codeintel.serving.semantic.iceberg_scans import (
    IcebergScanError,
    IcebergScanRequest,
    iceberg_scan_for_query,
    iceberg_table_exists,
)
from codeintel.serving.semantic.polars_query_builder import (
    PolarsQueryBuilderError,
    apply_query_ast,
    can_apply_query_ast,
)
from codeintel.serving.semantic.query_ast import ServingQuery
from codeintel.serving.semantic.routing import ast_supports_polars
from codeintel.serving.semantic.view_registry import ViewInputs
from codeintel.storage.duckdb_types import DuckDBError
from codeintel.storage.helpers.table_key import split_table_key
from codeintel.storage.schema import arrow_schema_for_table_key

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Mapping, Sequence
    from typing import Literal

    from polars import DataFrame, LazyFrame, QueryOptFlags

    from codeintel.core.schemas.primitives import ColumnType
    from codeintel.serving.semantic.models import FilterSpec
    from codeintel.serving.semantic.specs import SemanticQuerySpec
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


@dataclass(frozen=True, slots=True)
class PolarsExecutablePlan:
    """Executable Polars plan wrapper."""

    lazyframe: PolarsLazyFrame
    settings: ServingSettings
    explain_plan: str | None = None
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
        """Return the Polars explain plan.

        Returns
        -------
        QueryExplain
            Explain payload with the Polars plan text.

        Raises
        ------
        PolarsQueryBuilderError
            If Polars is unavailable.
        """
        if pl is None:  # pragma: no cover
            msg = "polars is required for Polars query execution"
            raise PolarsQueryBuilderError(msg)
        optimized = not self.settings.polars_inspect
        plan = self.explain_plan or _explain_plan(
            self.lazyframe,
            optimized=optimized,
            query_opt_flags=self.query_opt_flags,
        )
        return QueryExplain(sql=None, plan=plan)

    def cleanup(self) -> None:
        """Release temporary resources after execution."""
        if self.explain_plan is not None:
            return


@dataclass(frozen=True, slots=True)
class _PolarsSource:
    lazyframe: PolarsLazyFrame
    iceberg_snapshot_id: int | None = None


@dataclass(frozen=True, slots=True)
class _IcebergScanInputs:
    table_key: str
    columns: Sequence[str]
    filters: list[FilterSpec]
    order_by: Sequence[str]
    column_types: Mapping[str, ColumnType] | None
    primary_key: Sequence[str]


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


def _explain_plan(
    lazyframe: PolarsLazyFrame,
    *,
    optimized: bool,
    query_opt_flags: PolarsQueryOptFlags | None,
) -> str | None:
    explain_fn = getattr(lazyframe, "explain", None)
    if not callable(explain_fn):
        return None
    signature = _signature(explain_fn)
    if signature is None:
        return None
    kwargs: dict[str, object] = {}
    if "optimized" in signature.parameters:
        kwargs["optimized"] = optimized
    if query_opt_flags is not None:
        if "optimization_flags" in signature.parameters:
            kwargs["optimization_flags"] = query_opt_flags
        elif "query_opt_flags" in signature.parameters:
            kwargs["query_opt_flags"] = query_opt_flags
    try:
        result = explain_fn(**kwargs)
    except PolarsError:
        return None
    return cast("str | None", result)


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


@dataclass(frozen=True, slots=True)
class PolarsQueryEngine:
    """Polars query engine for semantic specs."""

    name: str = "polars"

    def can_run(self, query: ServingQuery, *, ctx: EngineContext) -> bool:
        """Return True when Polars can satisfy the query.

        Parameters
        ----------
        query
            Serving query bundle with AST/spec data.
        ctx
            Engine context with view and dataset registries.

        Returns
        -------
        bool
            True if Polars can execute the query.
        """
        if pl is None or self.name.lower() != "polars":
            return False
        if not ast_supports_polars(query.ast):
            return False
        spec = query.spec
        if not can_apply_query_ast(
            ast=query.ast,
            allowed_columns=spec.allowed_columns,
            column_types=spec.column_types,
        ):
            return False
        if ctx.view_registry.get(spec.table_key) is not None:
            return True
        return iceberg_table_exists(settings=ctx.settings.iceberg, table_key=spec.table_key)

    def compile(self, query: ServingQuery, *, ctx: EngineContext) -> ExecutablePlan:
        """Compile a serving query into a Polars execution plan.

        Parameters
        ----------
        query
            Serving query bundle with AST/spec data.
        ctx
            Engine context with data sources.

        Returns
        -------
        ExecutablePlan
            Executable Polars plan wrapper.

        Raises
        ------
        PolarsQueryBuilderError
            If Polars is unavailable or the query is invalid.
        """
        if pl is None:  # pragma: no cover
            msg = "polars is required for Polars query execution"
            raise PolarsQueryBuilderError(msg)
        spec = query.spec
        source = self._resolve_source(spec, ctx=ctx)
        query_opt_flags = _resolve_query_opt_flags(ctx.settings.polars_query_opt_flags)
        try:
            lazyframe = apply_query_ast(
                source.lazyframe,
                ast=query.ast,
                allowed_columns=spec.allowed_columns,
                column_types=spec.column_types,
            )
        except PolarsQueryBuilderError:
            raise
        except Exception as exc:  # pragma: no cover
            msg = f"Failed to build Polars query for {spec.table_key}"
            raise PolarsQueryBuilderError(msg) from exc
        explain_plan = None
        if (
            ctx.settings.polars_profile
            or ctx.settings.polars_inspect
            or query_opt_flags is not None
        ):
            explain_plan = _explain_plan(
                lazyframe,
                optimized=not ctx.settings.polars_inspect,
                query_opt_flags=query_opt_flags,
            )
        return PolarsExecutablePlan(
            lazyframe=lazyframe,
            settings=ctx.settings,
            explain_plan=explain_plan,
            query_opt_flags=query_opt_flags,
        )

    def _resolve_source(self, spec: SemanticQuerySpec, *, ctx: EngineContext) -> _PolarsSource:
        if pl is None:  # pragma: no cover
            msg = "polars is required for Polars query execution"
            raise PolarsQueryBuilderError(msg)
        view_spec = ctx.view_registry.get(spec.table_key)
        if view_spec is not None:
            inputs = ViewInputs(
                loader=lambda key, idx: self._scan_table(
                    ctx,
                    key,
                    idx,
                )
            )
            lazyframe = view_spec.builder(inputs)
            if not isinstance(lazyframe, pl.LazyFrame):
                msg = f"View builder for {spec.table_key} did not return a LazyFrame"
                raise PolarsQueryBuilderError(msg)
            return _PolarsSource(lazyframe=lazyframe)
        enforced = iceberg_enforced_table(
            settings=ctx.settings.iceberg,
            table_key=spec.table_key,
        )
        if enforced:
            require_iceberg_read(
                settings=ctx.settings.iceberg,
                table_key=spec.table_key,
            )
            if not iceberg_table_exists(
                settings=ctx.settings.iceberg,
                table_key=spec.table_key,
            ):
                msg = f"Iceberg table missing for enforced table: {spec.table_key}"
                raise PolarsQueryBuilderError(msg)
        if ctx.settings.iceberg.read_enabled:
            primary_key = _primary_key_for_table(
                ctx, table_key=spec.table_key, view_id=spec.view_id
            )
            iceberg_source = self._scan_iceberg(
                ctx,
                inputs=_IcebergScanInputs(
                    table_key=spec.table_key,
                    columns=spec.columns,
                    filters=spec.filters,
                    order_by=spec.order_by,
                    column_types=spec.column_types,
                    primary_key=primary_key,
                ),
                enforced=enforced,
            )
            if iceberg_source is not None:
                return iceberg_source
            if enforced:
                msg = f"Iceberg scan failed for enforced table: {spec.table_key}"
                raise PolarsQueryBuilderError(msg)
        msg = f"Iceberg scan unavailable for {spec.table_key}"
        raise PolarsQueryBuilderError(msg)

    def _scan_table(
        self,
        ctx: EngineContext,
        table_key: str,
        row_index: str | None,
    ) -> PolarsLazyFrame:
        enforced = iceberg_enforced_table(
            settings=ctx.settings.iceberg,
            table_key=table_key,
        )
        if enforced:
            require_iceberg_read(
                settings=ctx.settings.iceberg,
                table_key=table_key,
            )
            if not iceberg_table_exists(
                settings=ctx.settings.iceberg,
                table_key=table_key,
            ):
                msg = f"Iceberg table missing for enforced table: {table_key}"
                raise PolarsQueryBuilderError(msg)
        if ctx.settings.iceberg.read_enabled:
            primary_key = _primary_key_for_table(ctx, table_key=table_key, view_id=None)
            iceberg_source = self._scan_iceberg(
                ctx,
                inputs=_IcebergScanInputs(
                    table_key=table_key,
                    columns=_columns_for_table(ctx, table_key=table_key),
                    filters=[],
                    order_by=[],
                    column_types=None,
                    primary_key=primary_key,
                ),
                enforced=enforced,
            )
            if iceberg_source is not None:
                lazyframe = iceberg_source.lazyframe
                if row_index:
                    lazyframe = lazyframe.with_row_index(name=row_index)
                return lazyframe
            if enforced:
                msg = f"Iceberg scan failed for enforced table: {table_key}"
                raise PolarsQueryBuilderError(msg)
        msg = f"Iceberg scan unavailable for {table_key}"
        raise PolarsQueryBuilderError(msg)

    @staticmethod
    def _scan_iceberg(
        ctx: EngineContext,
        *,
        inputs: _IcebergScanInputs,
        enforced: bool = False,
    ) -> _PolarsSource | None:
        try:
            scan_result = iceberg_scan_for_query(
                request=IcebergScanRequest(
                    table_key=inputs.table_key,
                    columns=inputs.columns,
                    filters=inputs.filters,
                    order_by=inputs.order_by,
                    column_types=inputs.column_types,
                    pointer=ctx.pointer,
                    settings=ctx.settings.iceberg,
                    batch_size=ctx.settings.export_batch_size,
                )
            )
        except IcebergScanError as exc:
            if enforced:
                msg = f"Iceberg scan failed for enforced table: {inputs.table_key}"
                raise PolarsQueryBuilderError(msg) from exc
            LOG.warning("Iceberg scan failed for %s: %s", inputs.table_key, exc)
            return None
        lazyframe = to_lazyframe(scan_result.scan)
        lazyframe = _apply_iceberg_tombstones(
            lazyframe,
            ctx=ctx,
            table_key=inputs.table_key,
            primary_key=inputs.primary_key,
            snapshot_id=scan_result.snapshot_id,
        )
        return _PolarsSource(
            lazyframe=lazyframe,
            iceberg_snapshot_id=scan_result.snapshot_id,
        )


def _apply_iceberg_tombstones(
    lazyframe: PolarsLazyFrame,
    *,
    ctx: EngineContext,
    table_key: str,
    primary_key: Sequence[str],
    snapshot_id: int | None,
) -> PolarsLazyFrame:
    result = lazyframe
    if (
        not ctx.settings.iceberg.tombstones_enabled
        or not primary_key
        or snapshot_id is None
        or pl is None  # pragma: no cover
    ):
        return result
    tombstone_key = _tombstone_table_key(table_key)
    try:
        tombstone_scan = iceberg_scan_for_query(
            request=IcebergScanRequest(
                table_key=tombstone_key,
                columns=(*tuple(primary_key), "snapshot_id"),
                filters=[],
                order_by=[],
                column_types=None,
                pointer=ctx.pointer,
                settings=ctx.settings.iceberg,
                batch_size=ctx.settings.export_batch_size,
            )
        )
    except IcebergScanError as exc:
        LOG.warning("Tombstone scan failed for %s: %s", tombstone_key, exc)
        return result
    tombstones = to_lazyframe(tombstone_scan.scan)
    try:
        tombstones = tombstones.filter(pl.col("snapshot_id") <= snapshot_id)
        joined = result.join(tombstones, on=list(primary_key), how="anti")
    except PolarsError as exc:
        LOG.warning("Polars tombstone anti-join failed: %s", exc)
        return result
    if isinstance(joined, pl.LazyFrame):
        result = joined
    return result


def _tombstone_table_key(table_key: str) -> str:
    schema, table = split_table_key(table_key)
    return f"{schema}.{table}__tombstones"


def _columns_for_table(ctx: EngineContext, *, table_key: str) -> list[str]:
    schema = ctx.inventory.get(table_key)
    if schema is None:
        return []
    return [column.name for column in schema.columns]


def _primary_key_for_table(
    ctx: EngineContext,
    *,
    table_key: str,
    view_id: str | None,
) -> tuple[str, ...]:
    schema = ctx.inventory.get(table_key)
    if schema is not None and schema.primary_key:
        return tuple(schema.primary_key)
    if view_id is None:
        return ()
    try:
        view = ctx.registry.by_id(view_id)
    except KeyError:
        return ()
    if view.primary_key:
        return tuple(view.primary_key)
    return ()


_PROFILE_TUPLE_SIZE = 2

def _contract_schema_for_table(ctx: EngineContext, *, table_key: str) -> pa.Schema | None:
    if ctx.warehouse is None:
        return None
    _log_drift_if_present(ctx, table_key=table_key)
    try:
        return arrow_schema_for_table_key(
            ctx.warehouse.gateway.con,
            table_key=table_key,
            repo=ctx.pointer.repo,
            commit=ctx.pointer.commit,
        )
    except (DuckDBError, RuntimeError, TypeError, ValueError):
        return None


def _log_drift_if_present(ctx: EngineContext, *, table_key: str) -> None:
    if ctx.warehouse is None:
        return
    try:
        schemas = getattr(ctx.warehouse.gateway, "schemas", None)
        if schemas is None:
            return
        observation = schemas.load_latest_schema_observation(table_key=table_key)
    except (DuckDBError, RuntimeError, TypeError, ValueError):
        return
    if observation is None or observation.drift_summary is None:
        return
    warn_schema_drift_observed(table_key=table_key, drift_summary=observation.drift_summary)


__all__ = ["PolarsExecutablePlan", "PolarsQueryEngine"]
