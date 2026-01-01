"""Arrow dataset scanner utilities."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

import polars as pl
import pyarrow as pa
import pyarrow.dataset as ds

from codeintel.core.constants import DEFAULT_ARROW_BATCH_SIZE


def empty_reader_from_schema(schema: pa.Schema) -> pa.RecordBatchReader:
    """Return an empty reader with the provided schema.

    Parameters
    ----------
    schema
        Arrow schema to attach to the empty reader.

    Returns
    -------
    pyarrow.RecordBatchReader
        Empty reader matching the schema.
    """
    return pa.RecordBatchReader.from_batches(schema, [])


def scan_dataset_reader(
    dataset_dir: Path,
    *,
    columns: Sequence[str] | None = None,
    batch_size: int = DEFAULT_ARROW_BATCH_SIZE,
    fragment_readahead: int | None = None,
) -> pa.RecordBatchReader | None:
    """Return a streaming reader for a dataset directory.

    Parameters
    ----------
    dataset_dir
        Directory containing dataset files.
    columns
        Optional column projection for the scan. When omitted, scan all columns.
    batch_size
        Batch size for Arrow scanner.
    fragment_readahead
        Optional fragment readahead value.

    Returns
    -------
    pyarrow.RecordBatchReader | None
        Reader when the dataset is available, otherwise None.
    """
    if not dataset_dir.is_dir():
        return None
    scan_kwargs: dict[str, object] = {"batch_size": batch_size}
    if columns is not None:
        scan_kwargs["columns"] = list(columns)
    if fragment_readahead is not None:
        scan_kwargs["fragment_readahead"] = fragment_readahead
    try:
        dataset = ds.dataset(str(dataset_dir), format="parquet")
        scanner = dataset.scanner(**scan_kwargs)
        return scanner.to_reader()
    except (OSError, ValueError, pa.ArrowInvalid):
        return None


def scan_dataset_lazyframe(
    dataset_dir: Path,
    *,
    batch_size: int = DEFAULT_ARROW_BATCH_SIZE,
) -> pl.LazyFrame | None:
    """Return a Polars LazyFrame for a dataset directory.

    Parameters
    ----------
    dataset_dir
        Directory containing dataset files.
    batch_size
        Batch size for Arrow-backed scanning.

    Returns
    -------
    polars.LazyFrame | None
        Lazy frame when the dataset is available, otherwise None.
    """
    if not dataset_dir.is_dir():
        return None
    try:
        dataset = ds.dataset(str(dataset_dir), format="parquet")
        return pl.scan_pyarrow_dataset(dataset, batch_size=batch_size)
    except (OSError, ValueError, pa.ArrowInvalid):
        return None


def sample_reader(
    reader: pa.RecordBatchReader,
    *,
    max_rows: int,
) -> pa.RecordBatchReader:
    """Return a reader truncated to a maximum number of rows.

    Parameters
    ----------
    reader
        Reader to sample.
    max_rows
        Maximum number of rows to return.

    Returns
    -------
    pyarrow.RecordBatchReader
        Sampled reader with up to max_rows.
    """
    if max_rows <= 0:
        return empty_reader_from_schema(reader.schema)

    def _iter_batches() -> Iterable[pa.RecordBatch]:
        remaining = max_rows
        for batch in reader:
            if remaining <= 0:
                break
            current = batch
            if current.num_rows > remaining:
                current = current.slice(0, remaining)
            remaining -= current.num_rows
            yield current

    return pa.RecordBatchReader.from_batches(reader.schema, _iter_batches())


__all__ = [
    "empty_reader_from_schema",
    "sample_reader",
    "scan_dataset_lazyframe",
    "scan_dataset_reader",
]
