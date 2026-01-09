"""Parquet dataset helpers for graph engines and validation."""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds

from codeintel.build.graphs.assembly import iter_normalized_tuples
from codeintel.build.scopes.snapshot import SnapshotScanContext
from codeintel.core.columnar.finalize_ops import FinalizeSpec, finalize_reader
from codeintel.core.columnar.plan_ops import QueryPlanOptions, build_query_plan
from codeintel.core.columnar.queryspec import ProjectionSpec, QuerySpec
from codeintel.core.columnar.streaming import scan_telemetry
from codeintel.core.datasets.arrow_store import scan_dataset
from codeintel.core.datasets.paths import SnapshotIdError, dataset_snapshot_dir
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
    columns: tuple[str, ...] | Mapping[str, pc.Expression] | None = None
    provenance: bool = False
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
    provenance: bool = False


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
        columns: Sequence[str] | Mapping[str, pc.Expression] | None = None,
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
        resolved_columns: tuple[str, ...] | Mapping[str, pc.Expression] | None
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
            provenance=resolved_scan_options.provenance,
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
    if request.metrics_enabled:
        _log_scan_telemetry(
            dataset,
            table_key=request.table_key,
            snapshot_id=request.snapshot_id,
            filter_expression=filter_expression,
        )
    resolved_columns = _resolve_columns(dataset, request.columns)
    if resolved_columns is None and request.columns is not None:
        return None
    query_spec = _query_spec_for_request(
        dataset,
        columns=resolved_columns,
        predicate=filter_expression,
    )
    options = QueryPlanOptions(
        provenance=request.provenance,
        implicit_ordering=request.implicit_ordering,
        require_sequenced_output=request.require_sequenced_output,
    )
    plan = build_query_plan(dataset, spec=query_spec, options=options)
    use_threads = request.use_threads if request.use_threads is not None else True
    return plan.to_reader(use_threads=use_threads)


def scan_snapshot_reader_with_columns(
    request: SnapshotScanRequest,
    *,
    columns: tuple[str, ...] | Mapping[str, pc.Expression] | None,
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
    columns: tuple[str, ...] | Mapping[str, pc.Expression] | None,
) -> tuple[str, ...] | Mapping[str, pc.Expression] | None:
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


def _query_spec_for_request(
    dataset: ds.Dataset,
    *,
    columns: tuple[str, ...] | Mapping[str, pc.Expression] | None,
    predicate: pc.Expression | None,
) -> QuerySpec:
    projection = _projection_spec_for_columns(dataset, columns)
    return QuerySpec(
        predicate=predicate,
        pushdown_predicate=predicate,
        projection=projection,
    )


def _projection_spec_for_columns(
    dataset: ds.Dataset,
    columns: tuple[str, ...] | Mapping[str, pc.Expression] | None,
) -> ProjectionSpec:
    if columns is None:
        return ProjectionSpec(base_cols=tuple(dataset.schema.names))
    if isinstance(columns, Mapping):
        return ProjectionSpec(base_cols=(), computed=tuple(columns.items()))
    return ProjectionSpec(base_cols=tuple(columns))


def _log_scan_telemetry(
    dataset: ds.Dataset,
    *,
    table_key: str,
    snapshot_id: str,
    filter_expression: pc.Expression | None,
) -> None:
    telemetry = scan_telemetry(dataset, filter_expression=filter_expression)
    LOG.debug(
        "Dataset scan telemetry table=%s snapshot=%s fragments=%s rows=%s filter=%s",
        table_key,
        snapshot_id,
        telemetry.fragment_count,
        telemetry.estimated_rows,
        filter_expression,
    )


__all__ = [
    "GraphViewFactory",
    "SnapshotScanRequest",
    "dataset_snapshot_exists",
    "resolve_dataset_root",
    "scan_snapshot_reader",
    "scan_snapshot_table",
]
