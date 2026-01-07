"""Parquet dataset helpers for graph engines and validation."""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING

import pyarrow as pa
import pyarrow.dataset as ds

from codeintel.build.tabular.compute_masks import equal_expr
from codeintel.build.tabular.conversion import reader_to_table
from codeintel.core.columnar.streaming import DatasetScanOptions
from codeintel.core.constants import DEFAULT_ARROW_BATCH_SIZE
from codeintel.core.datasets.arrow_store import scan_dataset
from codeintel.core.datasets.paths import SnapshotIdError, dataset_snapshot_dir
from codeintel.core.datasets.scanner_ops import build_scanner

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
    _ = _snapshot
    if dataset_root_dir is not None:
        return dataset_root_dir
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


def scan_snapshot_reader_with_columns(
    request: SnapshotScanRequest,
    *,
    columns: tuple[str, ...] | None,
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
    return reader_to_table(reader)


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
        expression = equal_expr("repo", repo)
    if commit is not None and "commit" in names:
        commit_expr = equal_expr("commit", commit)
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


__all__ = [
    "SnapshotScanRequest",
    "dataset_snapshot_exists",
    "resolve_dataset_root",
    "scan_snapshot_reader",
    "scan_snapshot_table",
]
