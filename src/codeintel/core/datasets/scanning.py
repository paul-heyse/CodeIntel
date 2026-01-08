"""Parquet scan helpers shared across build and storage."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc

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

    columns: Sequence[str] | None = None
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
    expression: pc.Expression | None = None
    if resolved.repo is not None and "repo" in names:
        expression = equal_expr("repo", resolved.repo)
    if resolved.commit is not None and "commit" in names:
        commit_expr = equal_expr("commit", resolved.commit)
        expression = commit_expr if expression is None else expression & commit_expr

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
        columns=tuple(resolved.columns) if resolved.columns is not None else None,
        unify_schemas=True,
    )
    scanner = build_scanner(dataset, options=scan_options)
    return scanner.to_reader()


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


__all__ = [
    "ParquetScanOptions",
    "scan_parquet_dataset",
    "scan_parquet_table",
]
