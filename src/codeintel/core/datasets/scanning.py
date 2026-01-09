"""Parquet scan helpers shared across build and storage."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pyarrow as pa
import pyarrow.dataset as ds

from codeintel.core.columnar.conversion import reader_to_table
from codeintel.core.columnar.finalize_ops import FinalizeMode, FinalizeSpec, finalize_table
from codeintel.core.columnar.masks import equal_expr
from codeintel.core.columnar.normalization import normalize_table_for_compute
from codeintel.core.columnar.plan_ops import ScanPlanOptions, build_scan_plan
from codeintel.core.columnar.streaming import DatasetScanOptions
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
from codeintel.core.datasets.scanner_ops import build_scanner

LOG = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence


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
    dataset, scan_options = prepared
    plan_reader = _plan_scan_reader(dataset, scan_options)
    if plan_reader is not None:
        return plan_reader
    scanner = build_scanner(dataset, options=scan_options)
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
    dataset, scan_options = prepared
    telemetry = collect_parquet_scan_telemetry(
        dataset=dataset,
        table_key=table_key,
        snapshot_id=snapshot_id,
        scan_options=scan_options,
    )
    plan_reader = _plan_scan_reader(dataset, scan_options)
    if plan_reader is not None:
        return plan_reader, telemetry
    scanner = build_scanner(dataset, options=scan_options)
    return scanner.to_reader(), telemetry


def _prepare_parquet_dataset(
    *,
    dataset_root: Path,
    table_key: str,
    snapshot_id: str,
    options: ParquetScanOptions | None,
) -> tuple[ds.Dataset, DatasetScanOptions] | None:
    resolved = options or ParquetScanOptions()
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

    scan_options = DatasetScanOptions(
        batch_size=resolved.batch_size,
        batch_readahead=resolved.batch_readahead,
        fragment_readahead=resolved.fragment_readahead,
        filter_expression=expression,
        cache_metadata=resolved.cache_metadata,
        use_threads=resolved.use_threads,
        parquet_pre_buffer=resolved.parquet_pre_buffer,
        parquet_use_buffered_stream=resolved.parquet_use_buffered_stream,
        parquet_buffer_size=resolved.parquet_buffer_size,
        columns=resolved.columns,
        provenance_columns=_resolve_provenance_columns(resolved),
        implicit_ordering=resolved.implicit_ordering,
        require_sequenced_output=resolved.require_sequenced_output,
        metrics_enabled=resolved.metrics_enabled,
        unify_schemas=True,
    )
    return dataset, scan_options


def _plan_scan_reader(
    dataset: ds.Dataset,
    scan_options: DatasetScanOptions,
) -> pa.RecordBatchReader | None:
    use_threads = scan_options.use_threads
    resolved_use_threads = use_threads if use_threads is not None else True
    try:
        plan = build_scan_plan(
            dataset,
            options=ScanPlanOptions(
                columns=scan_options.projection_columns(),
                filter_expr=scan_options.filter_expression,
                implicit_ordering=scan_options.implicit_ordering,
                require_sequenced_output=scan_options.require_sequenced_output,
            ),
        )
        return plan.to_reader(use_threads=resolved_use_threads)
    except (
        pa.ArrowInvalid,
        pa.ArrowNotImplementedError,
        pa.ArrowTypeError,
        TypeError,
        ValueError,
    ):
        return None


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
    provenance_columns = _resolve_provenance_columns(resolved)
    table = normalize_table_for_compute(reader_to_table(reader))
    if resolved.finalize_mode is None or resolved.columns is not None:
        return table
    finalized = finalize_table(
        table,
        spec=FinalizeSpec(
            table_key=table_key,
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


def _resolve_provenance_columns(options: ParquetScanOptions) -> tuple[str, ...]:
    if options.provenance_columns:
        return tuple(options.provenance_columns)
    if options.metrics_enabled or options.finalize_mode is not None:
        return DEFAULT_ARROW_PROVENANCE_COLUMNS
    return ()


__all__ = [
    "ParquetScanOptions",
    "ParquetScanTelemetry",
    "collect_parquet_scan_telemetry",
    "scan_parquet_dataset",
    "scan_parquet_dataset_with_telemetry",
    "scan_parquet_table",
]
