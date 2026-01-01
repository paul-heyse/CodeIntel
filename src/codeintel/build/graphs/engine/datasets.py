"""Parquet dataset helpers for graph engines and validation."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl
import pyarrow as pa
import pyarrow.dataset as ds

from codeintel.core.constants import DEFAULT_ARROW_BATCH_SIZE
from codeintel.core.datasets.arrow_store import scan_dataset
from codeintel.core.datasets.paths import SnapshotIdError, dataset_snapshot_dir
from codeintel.core.datasets.scanning import DatasetScanOptions, build_scanner

if TYPE_CHECKING:
    from codeintel.config.primitives import SnapshotRef

LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class SnapshotScanRequest:
    """Scan request for dataset snapshots."""

    dataset_root: Path
    table_key: str
    snapshot_id: str
    columns: tuple[str, ...] | None = None
    repo: str | None = None
    commit: str | None = None
    batch_size: int = DEFAULT_ARROW_BATCH_SIZE


def resolve_dataset_root(
    snapshot: SnapshotRef,
    dataset_root_dir: Path | None,
) -> Path | None:
    """Resolve the dataset root directory for a snapshot.

    Parameters
    ----------
    snapshot
        Snapshot reference for repository context.
    dataset_root_dir
        Optional explicit dataset root directory.

    Returns
    -------
    pathlib.Path | None
        Resolved dataset root directory or None when not found.
    """
    if dataset_root_dir is not None:
        return dataset_root_dir
    candidate = snapshot.repo_root / "Document Output" / "datasets"
    if candidate.is_dir():
        return candidate
    return None


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
    filter_expression = _snapshot_filter_expression(
        dataset,
        repo=request.repo,
        commit=request.commit,
    )
    resolved_columns = _resolve_columns(dataset, request.columns)
    if resolved_columns is None and request.columns is not None:
        return None
    options = DatasetScanOptions(
        batch_size=request.batch_size,
        filter_expression=filter_expression,
        columns=resolved_columns,
        unify_schemas=True,
    )
    scanner = build_scanner(dataset, options=options)
    return scanner.to_reader()


def scan_snapshot_lazyframe(
    request: SnapshotScanRequest,
) -> pl.LazyFrame | None:
    """Return a LazyFrame for a dataset snapshot or None when missing.

    Parameters
    ----------
    request
        Snapshot scan request describing the dataset and filters.

    Returns
    -------
    polars.LazyFrame | None
        LazyFrame for the dataset snapshot or None when missing.
    """
    dataset = _scan_dataset(request.dataset_root, request.table_key, request.snapshot_id)
    if dataset is None:
        return None
    frame = pl.scan_pyarrow_dataset(dataset, batch_size=request.batch_size)
    frame = _filter_frame(
        frame,
        dataset.schema,
        repo=request.repo,
        commit=request.commit,
    )
    if request.columns is not None:
        resolved_columns = _resolve_columns(dataset, request.columns)
        if resolved_columns is None:
            return None
        frame = frame.select(list(resolved_columns))
    return frame


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


def _snapshot_filter_expression(
    dataset: ds.Dataset,
    *,
    repo: str | None,
    commit: str | None,
) -> ds.Expression | None:
    names = set(dataset.schema.names)
    expression: ds.Expression | None = None
    if repo is not None and "repo" in names:
        expression = ds.field("repo") == repo
    if commit is not None and "commit" in names:
        commit_expr = ds.field("commit") == commit
        expression = commit_expr if expression is None else expression & commit_expr
    return expression


def _resolve_columns(
    dataset: ds.Dataset,
    columns: tuple[str, ...] | None,
) -> tuple[str, ...] | None:
    if columns is None:
        return None
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


def _filter_frame(
    frame: pl.LazyFrame,
    schema: pa.Schema,
    *,
    repo: str | None,
    commit: str | None,
) -> pl.LazyFrame:
    names = set(schema.names)
    if repo is not None and "repo" in names:
        frame = frame.filter(pl.col("repo") == repo)
    if commit is not None and "commit" in names:
        frame = frame.filter(pl.col("commit") == commit)
    return frame


__all__ = [
    "SnapshotScanRequest",
    "dataset_snapshot_exists",
    "resolve_dataset_root",
    "scan_snapshot_lazyframe",
    "scan_snapshot_reader",
]
