"""Polars-based semantic query engine."""

from __future__ import annotations

import inspect
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import pyarrow as pa
import pyarrow.dataset as ds

from codeintel.serving.semantic.datasets import (
    DatasetScannerOptions,
    dataset_filter_expression,
    dataset_for_entry,
    dataset_scanner_for_entry,
)
from codeintel.serving.semantic.engines.protocol import EngineContext, ExecutablePlan, QueryExplain
from codeintel.serving.semantic.guardrails import (
    warn_eager_materialization,
    warn_schema_drift_observed,
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
from codeintel.storage.schema import arrow_schema_for_table_key

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Mapping
    from typing import Literal

    from polars import DataFrame, LazyFrame, QueryOptFlags

    from codeintel.core.schemas.primitives import ColumnType
    from codeintel.serving.semantic.datasets import DatasetManifestEntry
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


def _scan_arrow_dataset(
    entry: DatasetManifestEntry,
    *,
    filter_expression: ds.Expression | None,
    settings: ServingSettings,
    contract_schema: pa.Schema | None,
) -> PolarsLazyFrame | None:
    if pl is None:  # pragma: no cover
        msg = "polars is required for Polars query execution"
        raise PolarsQueryBuilderError(msg)
    dataset = dataset_for_entry(entry)
    options = DatasetScannerOptions(
        batch_size=settings.export_batch_size,
        fragment_readahead=settings.dataset_fragment_readahead,
        filter_expression=filter_expression,
        metrics_enabled=settings.dataset_scan_metrics_enabled,
        schema=contract_schema,
    )
    scanner = dataset_scanner_for_entry(entry, options=options)
    scan_pyarrow_dataset = getattr(pl, "scan_pyarrow_dataset", None)
    if not callable(scan_pyarrow_dataset):
        LOG.debug("Polars scan_pyarrow_dataset unavailable; falling back to scan_parquet.")
        return None
    try:
        scan = cast("Callable[..., PolarsLazyFrame]", scan_pyarrow_dataset)
        return scan(scanner)
    except TypeError:
        try:
            scan = cast("Callable[..., PolarsLazyFrame]", scan_pyarrow_dataset)
            return scan(dataset)
        except TypeError:
            return None


def _apply_sortedness(
    lazyframe: PolarsLazyFrame,
    *,
    entry: DatasetManifestEntry,
    settings: ServingSettings,
) -> PolarsLazyFrame:
    if not settings.polars_set_sorted:
        return lazyframe
    sort_keys = _manifest_sort_keys(entry)
    if not sort_keys:
        return lazyframe
    set_sorted = getattr(lazyframe, "set_sorted", None)
    if not callable(set_sorted):
        return lazyframe
    try:
        result = set_sorted(sort_keys)
    except PolarsError:
        LOG.debug("Polars set_sorted failed; continuing without sortedness.")
        return lazyframe
    if pl is not None and isinstance(result, pl.LazyFrame):
        return result
    return lazyframe


def _manifest_sort_keys(entry: DatasetManifestEntry) -> tuple[str, ...] | None:
    stats = entry.manifest.stats or {}
    raw = stats.get("sort_keys")
    if not raw:
        return None
    if isinstance(raw, tuple):
        return tuple(str(value) for value in raw)
    if isinstance(raw, list):
        return tuple(str(value) for value in raw)
    return None


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
        return ctx.dataset_manifests.get(spec.table_key) is not None

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
                source,
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

    def _resolve_source(self, spec: SemanticQuerySpec, *, ctx: EngineContext) -> PolarsLazyFrame:
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
            return lazyframe
        entry = ctx.dataset_manifests.get(spec.table_key)
        if entry is None:
            msg = f"No dataset manifest found for {spec.table_key}"
            raise PolarsQueryBuilderError(msg)
        contract_schema = _contract_schema_for_table(ctx, table_key=spec.table_key)
        return self._scan_entry(
            entry,
            filters=spec.filters,
            column_types=spec.column_types,
            settings=ctx.settings,
            contract_schema=contract_schema,
        )

    @staticmethod
    def _scan_entry(
        entry: DatasetManifestEntry,
        *,
        filters: list[FilterSpec] | None,
        column_types: Mapping[str, ColumnType] | None,
        settings: ServingSettings,
        contract_schema: pa.Schema | None,
    ) -> PolarsLazyFrame:
        if pl is None:  # pragma: no cover
            msg = "polars is required for Polars query execution"
            raise PolarsQueryBuilderError(msg)
        filter_expression = dataset_filter_expression(
            filters=filters or [],
            column_types=column_types,
        )
        lazyframe = None
        if settings.polars_use_arrow_scanner:
            lazyframe = _scan_arrow_dataset(
                entry,
                filter_expression=filter_expression,
                settings=settings,
                contract_schema=contract_schema,
            )
        hive_partitioning = bool(entry.manifest.partition_columns)
        if lazyframe is None:
            if entry.manifest.files:
                paths = [str(entry.dataset_dir / path) for path in entry.manifest.files]
                lazyframe = _scan_parquet(
                    paths,
                    hive_partitioning=hive_partitioning,
                    use_pyarrow=settings.polars_use_arrow_scanner,
                )
            else:
                glob = str(entry.dataset_dir / "**" / "*.parquet")
                lazyframe = _scan_parquet(
                    glob,
                    hive_partitioning=hive_partitioning,
                    use_pyarrow=settings.polars_use_arrow_scanner,
                )
        return _apply_sortedness(
            lazyframe,
            entry=entry,
            settings=settings,
        )

    def _scan_table(
        self,
        ctx: EngineContext,
        table_key: str,
        row_index: str | None,
    ) -> PolarsLazyFrame:
        entry = ctx.dataset_manifests.get(table_key)
        if entry is None:
            msg = f"No dataset manifest found for {table_key}"
            raise PolarsQueryBuilderError(msg)
        contract_schema = _contract_schema_for_table(ctx, table_key=table_key)
        lazyframe = self._scan_entry(
            entry,
            filters=None,
            column_types=None,
            settings=ctx.settings,
            contract_schema=contract_schema,
        )
        if row_index:
            lazyframe = lazyframe.with_row_index(name=row_index)
        return lazyframe


_PROFILE_TUPLE_SIZE = 2


def _scan_parquet(
    paths: list[str] | str,
    *,
    hive_partitioning: bool,
    use_pyarrow: bool,
) -> PolarsLazyFrame:
    if pl is None:  # pragma: no cover
        msg = "polars is required for Polars query execution"
        raise PolarsQueryBuilderError(msg)
    scan_parquet = cast("Callable[..., PolarsLazyFrame]", pl.scan_parquet)
    kwargs: dict[str, object] = {}
    if hive_partitioning:
        kwargs["hive_partitioning"] = True
    if use_pyarrow:
        kwargs["use_pyarrow"] = True
    signature = _signature(scan_parquet)
    if signature is not None:
        kwargs = {key: value for key, value in kwargs.items() if key in signature.parameters}
    try:
        return scan_parquet(paths, **kwargs)
    except TypeError:
        return scan_parquet(paths)


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
