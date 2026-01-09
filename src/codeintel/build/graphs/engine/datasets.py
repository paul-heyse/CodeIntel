"""Parquet dataset helpers for graph engines and validation."""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING

import pyarrow as pa
import pyarrow.dataset as ds

from codeintel.build.graphs.assembly import iter_normalized_tuples
from codeintel.build.scopes.snapshot import SnapshotScanContext
from codeintel.core.columnar.finalize_ops import FinalizeSpec, finalize_reader
from codeintel.core.datasets.arrow_store import scan_dataset
from codeintel.core.datasets.parquet_metadata import DatasetMetadataContext
from codeintel.core.datasets.paths import SnapshotIdError, dataset_snapshot_dir
from codeintel.core.datasets.scanner_ops import build_scanner
from codeintel.core.runtime.loader import load_runtime_settings

if TYPE_CHECKING:
    from codeintel.config.primitives import SnapshotRef

LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class SnapshotScanRequest:
    """Scan request for dataset snapshots."""

    dataset_root: Path
    table_key: str
    snapshot_id: str
    columns: tuple[str, ...] | Mapping[str, ds.Expression] | None = None
    repo: str | None = None
    commit: str | None = None
    batch_size: int | None = None
    batch_readahead: int | None = None
    fragment_readahead: int | None = None
    use_threads: bool | None = None
    cache_metadata: bool | None = None
    parquet_pre_buffer: bool | None = None
    parquet_use_buffered_stream: bool | None = None
    parquet_buffer_size: int | None = None
    unify_schemas: bool = True
    scan_context: SnapshotScanContext | None = None
    apply_filter: bool = True
    implicit_ordering: bool | None = True
    require_sequenced_output: bool | None = True
    metrics_enabled: bool = True


@dataclass(frozen=True, slots=True)
class GraphViewScanOptions:
    """Overrides for snapshot scan behavior in graph views."""

    apply_filter: bool = True
    implicit_ordering: bool | None = True
    require_sequenced_output: bool | None = True
    metrics_enabled: bool = True


@dataclass(frozen=True, slots=True)
class GraphViewFactory:
    """Factory for graph views backed by snapshot datasets."""

    dataset_root: Path
    snapshot_id: str
    scan_context: SnapshotScanContext

    @classmethod
    def for_snapshot(
        cls,
        dataset_root: Path,
        *,
        repo: str | None,
        commit: str,
    ) -> GraphViewFactory:
        """Build a graph view factory aligned to a snapshot.

        Parameters
        ----------
        dataset_root
            Root directory for Parquet dataset snapshots.
        repo
            Repository identifier anchoring the view.
        commit
            Commit hash anchoring the snapshot.

        Returns
        -------
        GraphViewFactory
            Factory configured for the snapshot.
        """
        scan_context = SnapshotScanContext(
            repo=repo,
            commit=commit,
            settings=load_runtime_settings().build.arrow_scan,
        )
        return cls(dataset_root=dataset_root, snapshot_id=commit, scan_context=scan_context)

    def load_reader(
        self,
        *,
        table_key: str,
        columns: Sequence[str] | Mapping[str, ds.Expression] | None = None,
        scan_options: GraphViewScanOptions | None = None,
    ) -> pa.RecordBatchReader | None:
        """Return a record batch reader for a snapshot table.

        Parameters
        ----------
        table_key
            Dataset table key.
        columns
            Optional column selection for the scan.
        scan_options
            Optional scan overrides (filter, ordering, metrics).

        Returns
        -------
        pyarrow.RecordBatchReader | None
            Reader for the dataset snapshot or None when missing.
        """
        resolved_scan_options = scan_options or GraphViewScanOptions()
        resolved_columns: tuple[str, ...] | Mapping[str, ds.Expression] | None
        if isinstance(columns, Mapping):
            resolved_columns = columns
        elif columns is None:
            resolved_columns = None
        else:
            resolved_columns = tuple(columns)
        request = SnapshotScanRequest(
            dataset_root=self.dataset_root,
            table_key=table_key,
            snapshot_id=self.snapshot_id,
            columns=resolved_columns,
            repo=self.scan_context.repo,
            commit=self.scan_context.commit,
            scan_context=self.scan_context,
            apply_filter=resolved_scan_options.apply_filter,
            implicit_ordering=resolved_scan_options.implicit_ordering,
            require_sequenced_output=resolved_scan_options.require_sequenced_output,
            metrics_enabled=resolved_scan_options.metrics_enabled,
        )
        return scan_snapshot_reader(request)

    @staticmethod
    def iter_tuples(
        reader: pa.RecordBatchReader,
        *,
        columns: Sequence[str] | None = None,
    ) -> Iterable[tuple[object, ...]]:
        """Yield normalized row tuples from a record batch reader.

        Parameters
        ----------
        reader
            Reader supplying record batches.
        columns
            Optional column selection for tuple materialization.

        Yields
        ------
        tuple[object, ...]
            Row tuples in column order after normalization.
        """
        yield from iter_normalized_tuples(reader, columns=columns)


def resolve_dataset_root(
    _snapshot: SnapshotRef,
    dataset_root_dir: Path | None,
) -> Path | None:
    """Resolve the dataset root directory for a snapshot.

    Parameters
    ----------
    _snapshot
        Snapshot reference for repository context.
    dataset_root_dir
        Optional explicit dataset root directory.

    Returns
    -------
    pathlib.Path | None
        Resolved dataset root directory or None when not found.
    """
    return dataset_root_dir


def dataset_snapshot_exists(
    dataset_root: Path | None,
    table_key: str,
    snapshot_id: str,
) -> bool:
    """Return True when a dataset snapshot directory exists.

    Parameters
    ----------
    dataset_root
        Root directory for datasets.
    table_key
        Dataset table key.
    snapshot_id
        Snapshot identifier for the dataset.

    Returns
    -------
    bool
        True when the snapshot directory exists, otherwise False.
    """
    if dataset_root is None:
        return False
    try:
        snapshot_dir = dataset_snapshot_dir(
            dataset_root,
            table_key=table_key,
            snapshot_id=snapshot_id,
        )
    except SnapshotIdError as exc:
        LOG.warning("Invalid snapshot_id for %s: %s", table_key, exc)
        return False
    return snapshot_dir.is_dir()


def _metadata_schema_for_request(request: SnapshotScanRequest) -> pa.Schema | None:
    try:
        snapshot_dir = dataset_snapshot_dir(
            request.dataset_root,
            table_key=request.table_key,
            snapshot_id=request.snapshot_id,
        )
    except SnapshotIdError as exc:
        LOG.warning("Invalid snapshot_id for %s: %s", request.table_key, exc)
        return None
    metadata_ctx = DatasetMetadataContext(
        dataset_root=snapshot_dir,
        table_key=request.table_key,
    )
    return metadata_ctx.read_schema()


def scan_snapshot_reader(
    request: SnapshotScanRequest,
) -> pa.RecordBatchReader | None:
    """Return a RecordBatchReader for a dataset snapshot or None when missing.

    Parameters
    ----------
    request
        Snapshot scan request describing the dataset and filters.

    Returns
    -------
    pyarrow.RecordBatchReader | None
        Reader for the dataset snapshot or None when missing.
    """
    dataset = _scan_dataset(request.dataset_root, request.table_key, request.snapshot_id)
    if dataset is None:
        return None
    scan_ctx = request.scan_context or SnapshotScanContext(
        repo=request.repo,
        commit=request.commit,
        settings=load_runtime_settings().build.arrow_scan,
    )
    filter_expression = scan_ctx.filter_expr(dataset.schema) if request.apply_filter else None
    resolved_columns = _resolve_columns(dataset, request.columns)
    if resolved_columns is None and request.columns is not None:
        return None
    options = scan_ctx.scan_options(
        columns=resolved_columns,
        batch_size=request.batch_size,
    )
    metadata_schema = _metadata_schema_for_request(request)
    options = replace(
        options,
        batch_readahead=request.batch_readahead
        if request.batch_readahead is not None
        else options.batch_readahead,
        fragment_readahead=request.fragment_readahead
        if request.fragment_readahead is not None
        else options.fragment_readahead,
        filter_expression=filter_expression,
        cache_metadata=request.cache_metadata
        if request.cache_metadata is not None
        else options.cache_metadata,
        use_threads=(
            request.use_threads if request.use_threads is not None else options.use_threads
        ),
        parquet_pre_buffer=request.parquet_pre_buffer
        if request.parquet_pre_buffer is not None
        else options.parquet_pre_buffer,
        parquet_use_buffered_stream=request.parquet_use_buffered_stream
        if request.parquet_use_buffered_stream is not None
        else options.parquet_use_buffered_stream,
        parquet_buffer_size=request.parquet_buffer_size
        if request.parquet_buffer_size is not None
        else options.parquet_buffer_size,
        columns=resolved_columns,
        schema=metadata_schema if metadata_schema is not None else options.schema,
        unify_schemas=request.unify_schemas,
        implicit_ordering=request.implicit_ordering,
        require_sequenced_output=request.require_sequenced_output,
        metrics_enabled=request.metrics_enabled,
    )
    if request.metrics_enabled:
        _log_scan_telemetry(
            dataset,
            table_key=request.table_key,
            snapshot_id=request.snapshot_id,
            filter_expression=filter_expression,
        )
    scanner = build_scanner(dataset, options=options)
    return scanner.to_reader()


def scan_snapshot_reader_with_columns(
    request: SnapshotScanRequest,
    *,
    columns: tuple[str, ...] | Mapping[str, ds.Expression] | None,
) -> pa.RecordBatchReader | None:
    """Return a RecordBatchReader for a dataset snapshot with selected columns.

    Returns
    -------
    pyarrow.RecordBatchReader | None
        Reader for the dataset snapshot or None when missing.
    """
    updated = replace(request, columns=columns)
    return scan_snapshot_reader(updated)


def scan_snapshot_table(
    request: SnapshotScanRequest,
) -> pa.Table | None:
    """Return a materialized Arrow Table for a dataset snapshot.

    Returns
    -------
    pyarrow.Table | None
        Arrow table for the dataset snapshot or None when missing.
    """
    reader = scan_snapshot_reader(request)
    if reader is None:
        return None
    result = finalize_reader(
        reader,
        spec=FinalizeSpec(table_key=request.table_key, mode="tolerant"),
    )
    return result.good


def _scan_dataset(dataset_root: Path, table_key: str, snapshot_id: str) -> ds.Dataset | None:
    try:
        return scan_dataset(
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


def _resolve_columns(
    dataset: ds.Dataset,
    columns: tuple[str, ...] | Mapping[str, ds.Expression] | None,
) -> tuple[str, ...] | Mapping[str, ds.Expression] | None:
    if columns is None:
        return None
    if isinstance(columns, Mapping):
        return columns
    available = set(dataset.schema.names)
    missing = [name for name in columns if name not in available]
    if missing:
        LOG.warning(
            "Dataset columns missing: %s (table=%s)",
            ", ".join(missing),
            dataset.schema,
        )
        return None
    return columns


def _log_scan_telemetry(
    dataset: ds.Dataset,
    *,
    table_key: str,
    snapshot_id: str,
    filter_expression: ds.Expression | None,
) -> None:
    fragments = _count_fragments(dataset, filter_expression)
    rows = _count_rows(dataset, filter_expression)
    LOG.debug(
        "Dataset scan telemetry table=%s snapshot=%s fragments=%s rows=%s filter=%s",
        table_key,
        snapshot_id,
        fragments,
        rows,
        filter_expression,
    )


def _count_fragments(
    dataset: ds.Dataset,
    filter_expression: ds.Expression | None,
) -> int | None:
    get_fragments = getattr(dataset, "get_fragments", None)
    if not callable(get_fragments):
        return None
    try:
        fragments = (
            get_fragments(filter=filter_expression)
            if filter_expression is not None
            else get_fragments()
        )
    except (TypeError, ValueError, pa.ArrowInvalid):
        return None
    if not isinstance(fragments, Iterable):
        return None
    try:
        return len(tuple(fragments))
    except TypeError:
        return None


def _count_rows(
    dataset: ds.Dataset,
    filter_expression: ds.Expression | None,
) -> int | None:
    counter = getattr(dataset, "count_rows", None)
    if callable(counter):
        try:
            count = counter() if filter_expression is None else counter(filter=filter_expression)
        except (TypeError, ValueError, pa.ArrowInvalid):
            count = None
        if isinstance(count, int):
            return count
    try:
        scanner = (
            dataset.scanner(filter=filter_expression)
            if filter_expression is not None
            else dataset.scanner()
        )
    except (TypeError, ValueError, pa.ArrowInvalid):
        return None
    count_rows = getattr(scanner, "count_rows", None)
    if callable(count_rows):
        try:
            count = count_rows()
        except (TypeError, ValueError, pa.ArrowInvalid):
            return None
        return count if isinstance(count, int) else None
    return None


__all__ = [
    "GraphViewFactory",
    "SnapshotScanRequest",
    "dataset_snapshot_exists",
    "resolve_dataset_root",
    "scan_snapshot_reader",
    "scan_snapshot_table",
]
