"""Parquet scan helpers shared across build and storage."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pyarrow as pa
import pyarrow.dataset as ds

from codeintel.core.columnar.arrowdsl import ExecutionPlan
from codeintel.core.columnar.conversion import reader_to_table
from codeintel.core.columnar.execution_context import ExecutionContext
from codeintel.core.columnar.finalize_ops import (
    FinalizeMode,
    finalize_spec_for_table,
    finalize_table,
)
from codeintel.core.columnar.masks import equal_expr
from codeintel.core.columnar.normalization import normalize_table_for_compute
from codeintel.core.columnar.plan_ops import QueryPlanOptions, build_query_plan_for_context
from codeintel.core.columnar.queryspec import QuerySpec, projection_spec_from_columns
from codeintel.core.columnar.streaming import (
    DatasetScanOptions,
    build_scanner_for_queryspec_ctx,
    configure_arrow_threading_for_context,
    scan_options_for_queryspec_ctx,
)
from codeintel.core.constants import (
    DEFAULT_ARROW_BATCH_READAHEAD,
    DEFAULT_ARROW_BATCH_SIZE,
    DEFAULT_ARROW_CACHE_METADATA,
    DEFAULT_ARROW_FRAGMENT_READAHEAD,
    DEFAULT_ARROW_PARQUET_BUFFER_SIZE,
    DEFAULT_ARROW_PARQUET_PRE_BUFFER,
    DEFAULT_ARROW_PARQUET_USE_BUFFERED_STREAM,
    DEFAULT_ARROW_PROVENANCE_COLUMNS,
    DEFAULT_ARROW_USE_THREADS,
)
from codeintel.core.datasets.arrow_store import scan_dataset

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

LOG = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ParquetScanOptions:
    """Options for snapshot-scoped parquet scans."""

    columns: Sequence[str] | Mapping[str, ds.Expression] | None = None
    provenance_columns: Sequence[str] = ()
    repo: str | None = None
    commit: str | None = None
    batch_size: int = DEFAULT_ARROW_BATCH_SIZE
    batch_readahead: int | None = DEFAULT_ARROW_BATCH_READAHEAD
    fragment_readahead: int | None = DEFAULT_ARROW_FRAGMENT_READAHEAD
    use_threads: bool | None = DEFAULT_ARROW_USE_THREADS
    cache_metadata: bool | None = DEFAULT_ARROW_CACHE_METADATA
    parquet_pre_buffer: bool | None = DEFAULT_ARROW_PARQUET_PRE_BUFFER
    parquet_use_buffered_stream: bool | None = DEFAULT_ARROW_PARQUET_USE_BUFFERED_STREAM
    parquet_buffer_size: int | None = DEFAULT_ARROW_PARQUET_BUFFER_SIZE
    implicit_ordering: bool | None = None
    require_sequenced_output: bool | None = None
    metrics_enabled: bool = False
    finalize_mode: FinalizeMode | None = None
    execution_ctx: ExecutionContext | None = None


@dataclass(frozen=True, slots=True)
class ParquetScanTelemetry:
    """Telemetry collected during a dataset scan plan."""

    table_key: str
    snapshot_id: str
    fragment_count: int | None
    row_count: int | None
    filter_expression: ds.Expression | None
    projection_columns: tuple[str, ...] = ()
    provenance_columns: tuple[str, ...] = ()

    def to_mapping(self) -> dict[str, object]:
        """Return a mapping representation for telemetry logging.

        Returns
        -------
        dict[str, object]
            Mapping payload suitable for logs or metrics sinks.
        """
        payload: dict[str, object] = {
            "table_key": self.table_key,
            "snapshot_id": self.snapshot_id,
        }
        if self.fragment_count is not None:
            payload["fragment_count"] = self.fragment_count
        if self.row_count is not None:
            payload["row_count"] = self.row_count
        if self.filter_expression is not None:
            payload["filter_expression"] = str(self.filter_expression)
        if self.projection_columns:
            payload["projection_columns"] = list(self.projection_columns)
        if self.provenance_columns:
            payload["provenance_columns"] = list(self.provenance_columns)
        return payload


@dataclass(frozen=True, slots=True)
class PreparedParquetScan:
    """Prepared dataset scan metadata."""

    dataset: ds.Dataset
    query_spec: QuerySpec
    scan_options: DatasetScanOptions
    use_threads: bool
    execution_ctx: ExecutionContext | None


def scan_parquet_dataset(
    *,
    dataset_root: Path,
    table_key: str,
    snapshot_id: str,
    options: ParquetScanOptions | None = None,
) -> pa.RecordBatchReader | None:
    """Return a RecordBatchReader for a parquet dataset snapshot.

    Returns
    -------
    pa.RecordBatchReader | None
        RecordBatchReader when a dataset snapshot is available, otherwise None.
    """
    resolved = options or ParquetScanOptions()
    if resolved.metrics_enabled:
        reader, telemetry = scan_parquet_dataset_with_telemetry(
            dataset_root=dataset_root,
            table_key=table_key,
            snapshot_id=snapshot_id,
            options=resolved,
        )
        if telemetry is not None:
            LOG.debug("Parquet scan telemetry: %s", telemetry.to_mapping())
        return reader
    prepared = _prepare_parquet_dataset(
        dataset_root=dataset_root,
        table_key=table_key,
        snapshot_id=snapshot_id,
        options=resolved,
    )
    if prepared is None:
        return None
    plan_reader = _plan_scan_reader(prepared)
    if plan_reader is not None:
        return plan_reader
    scanner = build_scanner_for_queryspec_ctx(
        prepared.dataset,
        spec=prepared.query_spec,
        ctx=prepared.execution_ctx,
        options=prepared.scan_options,
    )
    return scanner.to_reader()


def scan_parquet_dataset_with_telemetry(
    *,
    dataset_root: Path,
    table_key: str,
    snapshot_id: str,
    options: ParquetScanOptions | None = None,
) -> tuple[pa.RecordBatchReader | None, ParquetScanTelemetry | None]:
    """Return a parquet dataset reader with scan telemetry.

    Returns
    -------
    tuple[pa.RecordBatchReader | None, ParquetScanTelemetry | None]
        Reader plus scan telemetry, or (None, None) when unavailable.
    """
    prepared = _prepare_parquet_dataset(
        dataset_root=dataset_root,
        table_key=table_key,
        snapshot_id=snapshot_id,
        options=options,
    )
    if prepared is None:
        return None, None
    telemetry = collect_parquet_scan_telemetry(
        dataset=prepared.dataset,
        table_key=table_key,
        snapshot_id=snapshot_id,
        scan_options=prepared.scan_options,
    )
    plan_reader = _plan_scan_reader(prepared)
    if plan_reader is not None:
        return plan_reader, telemetry
    scanner = build_scanner_for_queryspec_ctx(
        prepared.dataset,
        spec=prepared.query_spec,
        ctx=prepared.execution_ctx,
        options=prepared.scan_options,
    )
    return scanner.to_reader(), telemetry


def _prepare_parquet_dataset(
    *,
    dataset_root: Path,
    table_key: str,
    snapshot_id: str,
    options: ParquetScanOptions | None,
) -> PreparedParquetScan | None:
    resolved = options or ParquetScanOptions()
    configure_arrow_threading_for_context(ctx=resolved.execution_ctx)
    try:
        dataset = scan_dataset(
            dataset_root=dataset_root,
            table_key=table_key,
            snapshot_id=snapshot_id,
        )
    except FileNotFoundError:
        LOG.warning("Dataset snapshot missing for %s@%s", table_key, snapshot_id)
        return None
    except (OSError, ValueError, pa.ArrowInvalid) as exc:
        LOG.warning("Dataset scan failed for %s@%s: %s", table_key, snapshot_id, exc)
        return None

    names = set(dataset.schema.names)
    expression: ds.Expression | None = None
    if resolved.repo is not None and "repo" in names:
        expression = cast("ds.Expression", equal_expr("repo", resolved.repo))
    if resolved.commit is not None and "commit" in names:
        commit_expr = cast("ds.Expression", equal_expr("commit", resolved.commit))
        if expression is None:
            expression = commit_expr
        else:
            expression = cast("ds.Expression", expression & commit_expr)

    provenance_columns = _resolve_provenance_columns(resolved, ctx=resolved.execution_ctx)
    query_spec = _query_spec_for_scan(
        columns=resolved.columns,
        provenance_columns=provenance_columns,
        predicate=expression,
    )
    base_options = _base_scan_options(resolved, provenance_columns=provenance_columns)
    scan_options = scan_options_for_queryspec_ctx(
        query_spec,
        ctx=resolved.execution_ctx,
        options=base_options,
    )
    use_threads = _resolve_use_threads(resolved, ctx=resolved.execution_ctx)
    return PreparedParquetScan(
        dataset=dataset,
        query_spec=query_spec,
        scan_options=scan_options,
        use_threads=use_threads,
        execution_ctx=resolved.execution_ctx,
    )


def _plan_scan_reader(
    prepared: PreparedParquetScan,
) -> pa.RecordBatchReader | None:
    query_plan_options = QueryPlanOptions(
        implicit_ordering=prepared.scan_options.implicit_ordering,
        require_sequenced_output=prepared.scan_options.require_sequenced_output,
    )
    try:
        plan = build_query_plan_for_context(
            prepared.dataset,
            spec=prepared.query_spec,
            ctx=prepared.execution_ctx,
            options=query_plan_options,
        )
        execution_ctx = prepared.execution_ctx
        if execution_ctx is None:
            execution_ctx = ExecutionContext(use_threads=prepared.use_threads)
        return ExecutionPlan.from_plan(plan).to_reader(ctx=execution_ctx)
    except (
        pa.ArrowInvalid,
        pa.ArrowNotImplementedError,
        pa.ArrowTypeError,
        TypeError,
        ValueError,
    ):
        return None


def _query_spec_for_scan(
    *,
    columns: Sequence[str] | Mapping[str, ds.Expression] | None,
    provenance_columns: Sequence[str],
    predicate: ds.Expression | None,
) -> QuerySpec:
    projection = projection_spec_from_columns(
        columns,
        provenance_columns=provenance_columns,
    )
    return QuerySpec(
        predicate=predicate,
        pushdown_predicate=predicate,
        projection=projection,
    )


def _base_scan_options(
    options: ParquetScanOptions,
    *,
    provenance_columns: Sequence[str],
) -> DatasetScanOptions:
    return DatasetScanOptions(
        batch_size=options.batch_size,
        batch_readahead=options.batch_readahead,
        fragment_readahead=options.fragment_readahead,
        cache_metadata=options.cache_metadata,
        use_threads=options.use_threads,
        parquet_pre_buffer=options.parquet_pre_buffer,
        parquet_use_buffered_stream=options.parquet_use_buffered_stream,
        parquet_buffer_size=options.parquet_buffer_size,
        provenance_columns=tuple(provenance_columns),
        implicit_ordering=options.implicit_ordering,
        require_sequenced_output=options.require_sequenced_output,
        metrics_enabled=options.metrics_enabled,
        unify_schemas=True,
    )


def _resolve_use_threads(
    options: ParquetScanOptions,
    *,
    ctx: ExecutionContext | None,
) -> bool:
    default_threads = options.use_threads
    if default_threads is None:
        default_threads = DEFAULT_ARROW_USE_THREADS
    if ctx is None:
        return default_threads
    return ctx.resolve_use_threads()


def scan_parquet_table(
    *,
    dataset_root: Path,
    table_key: str,
    snapshot_id: str,
    options: ParquetScanOptions | None = None,
) -> pa.Table | None:
    """Return a materialized Arrow Table for a parquet dataset snapshot.

    Returns
    -------
    pa.Table | None
        Materialized Arrow table when available, otherwise None.
    """
    reader = scan_parquet_dataset(
        dataset_root=dataset_root,
        table_key=table_key,
        snapshot_id=snapshot_id,
        options=options,
    )
    if reader is None:
        return None
    resolved = options or ParquetScanOptions()
    provenance_columns = _resolve_provenance_columns(resolved, ctx=resolved.execution_ctx)
    table = normalize_table_for_compute(reader_to_table(reader))
    if resolved.finalize_mode is None or resolved.columns is not None:
        return table
    finalized = finalize_table(
        table,
        spec=finalize_spec_for_table(
            table_key,
            mode=resolved.finalize_mode,
            context_fields=provenance_columns,
        ),
    )
    return finalized.good


def collect_parquet_scan_telemetry(
    *,
    dataset: ds.Dataset,
    table_key: str,
    snapshot_id: str,
    scan_options: DatasetScanOptions,
) -> ParquetScanTelemetry:
    """Collect scan telemetry for a dataset scan plan.

    Returns
    -------
    ParquetScanTelemetry
        Telemetry summary for the dataset scan.
    """
    return _collect_parquet_scan_telemetry(
        dataset=dataset,
        table_key=table_key,
        snapshot_id=snapshot_id,
        scan_options=scan_options,
    )


def _collect_parquet_scan_telemetry(
    *,
    dataset: ds.Dataset,
    table_key: str,
    snapshot_id: str,
    scan_options: DatasetScanOptions,
) -> ParquetScanTelemetry:
    projection_columns = _projection_column_names(scan_options.projection_columns())
    return ParquetScanTelemetry(
        table_key=table_key,
        snapshot_id=snapshot_id,
        fragment_count=_count_fragments(dataset, scan_options.filter_expression),
        row_count=_count_rows(dataset, scan_options.filter_expression),
        filter_expression=scan_options.filter_expression,
        projection_columns=projection_columns,
        provenance_columns=tuple(scan_options.provenance_columns),
    )


def _count_fragments(
    dataset: ds.Dataset,
    filter_expression: ds.Expression | None,
) -> int | None:
    get_fragments = getattr(dataset, "get_fragments", None)
    if not callable(get_fragments):
        return None
    try:
        if filter_expression is None:
            fragments = get_fragments()
        else:
            fragments = get_fragments(filter=filter_expression)
    except (TypeError, ValueError, pa.ArrowInvalid):
        return None
    try:
        iterable = cast("Iterable[ds.Fragment]", fragments)
        return len(tuple(iterable))
    except TypeError:
        return None


def _count_rows(
    dataset: ds.Dataset,
    filter_expression: ds.Expression | None,
) -> int | None:
    count: int | None = None
    counter = getattr(dataset, "count_rows", None)
    if callable(counter):
        try:
            if filter_expression is None:
                count = _coerce_int(counter())
            else:
                count = _coerce_int(counter(filter=filter_expression))
        except (TypeError, ValueError, pa.ArrowInvalid):
            count = None
    if count is not None:
        return count
    try:
        if filter_expression is None:
            scanner = dataset.scanner()
        else:
            scanner = dataset.scanner(filter=filter_expression)
    except (TypeError, ValueError, pa.ArrowInvalid):
        return None
    scanner_counter = getattr(scanner, "count_rows", None)
    if callable(scanner_counter):
        try:
            count = _coerce_int(scanner_counter())
        except (TypeError, ValueError, pa.ArrowInvalid):
            count = None
    return count


def _coerce_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def _projection_column_names(
    columns: Sequence[str] | Mapping[str, ds.Expression] | None,
) -> tuple[str, ...]:
    if columns is None:
        return ()
    if isinstance(columns, Mapping):
        return tuple(columns.keys())
    return tuple(columns)


def _resolve_provenance_columns(
    options: ParquetScanOptions,
    *,
    ctx: ExecutionContext | None,
) -> tuple[str, ...]:
    if options.provenance_columns:
        return tuple(options.provenance_columns)
    if _resolve_provenance_enabled(options, ctx=ctx):
        return DEFAULT_ARROW_PROVENANCE_COLUMNS
    return ()


def _resolve_provenance_enabled(
    options: ParquetScanOptions,
    *,
    ctx: ExecutionContext | None,
) -> bool:
    enabled = options.metrics_enabled or options.finalize_mode is not None
    if ctx is None:
        return enabled
    provenance = ctx.provenance or enabled
    profile = ctx.runtime_profile
    if profile is not None:
        provenance = profile.resolve_provenance(default=provenance)
    return provenance


__all__ = [
    "ParquetScanOptions",
    "ParquetScanTelemetry",
    "collect_parquet_scan_telemetry",
    "scan_parquet_dataset",
    "scan_parquet_dataset_with_telemetry",
    "scan_parquet_table",
]
