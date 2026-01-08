"""Parquet scan helpers shared across build and storage."""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import pyarrow as pa
import pyarrow.dataset as ds

from codeintel.core.columnar.conversion import reader_to_table
from codeintel.core.columnar.masks import equal_expr
from codeintel.core.columnar.streaming import DatasetScanOptions
from codeintel.core.constants import (
    DEFAULT_ARROW_BATCH_READAHEAD,
    DEFAULT_ARROW_BATCH_SIZE,
    DEFAULT_ARROW_CACHE_METADATA,
    DEFAULT_ARROW_FRAGMENT_READAHEAD,
    DEFAULT_ARROW_PARQUET_BUFFER_SIZE,
    DEFAULT_ARROW_PARQUET_PRE_BUFFER,
    DEFAULT_ARROW_PARQUET_USE_BUFFERED_STREAM,
    DEFAULT_ARROW_USE_THREADS,
)
from codeintel.core.datasets.arrow_store import scan_dataset
from codeintel.core.datasets.scanner_ops import build_scanner

LOG = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ParquetScanOptions:
    """Options for snapshot-scoped parquet scans."""

    columns: Sequence[str] | Mapping[str, ds.Expression] | None = None
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


@dataclass(frozen=True, slots=True)
class ParquetScanTelemetry:
    """Telemetry collected during a dataset scan plan."""

    table_key: str
    snapshot_id: str
    fragment_count: int | None
    row_count: int | None
    filter_expression: ds.Expression | None


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
    prepared = _prepare_parquet_dataset(
        dataset_root=dataset_root,
        table_key=table_key,
        snapshot_id=snapshot_id,
        options=options,
    )
    if prepared is None:
        return None
    dataset, scan_options = prepared
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
    telemetry = _collect_parquet_scan_telemetry(
        dataset=dataset,
        table_key=table_key,
        snapshot_id=snapshot_id,
        filter_expression=scan_options.filter_expression,
    )
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
        implicit_ordering=resolved.implicit_ordering,
        require_sequenced_output=resolved.require_sequenced_output,
        unify_schemas=True,
    )
    return dataset, scan_options


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
    return reader_to_table(reader)


def _collect_parquet_scan_telemetry(
    *,
    dataset: ds.Dataset,
    table_key: str,
    snapshot_id: str,
    filter_expression: ds.Expression | None,
) -> ParquetScanTelemetry:
    return ParquetScanTelemetry(
        table_key=table_key,
        snapshot_id=snapshot_id,
        fragment_count=_count_fragments(dataset, filter_expression),
        row_count=_count_rows(dataset, filter_expression),
        filter_expression=filter_expression,
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


__all__ = [
    "ParquetScanOptions",
    "ParquetScanTelemetry",
    "scan_parquet_dataset",
    "scan_parquet_dataset_with_telemetry",
    "scan_parquet_table",
]
