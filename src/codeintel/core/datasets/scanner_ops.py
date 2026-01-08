"""Dataset scanner helpers for Arrow datasets."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import pyarrow as pa
import pyarrow.dataset as ds

from codeintel.core.columnar.schema_ops import DEFAULT_SCHEMA_PROMOTE_OPTIONS, SchemaPromoteOptions
from codeintel.core.columnar.streaming import DatasetScanOptions
from codeintel.core.columnar.streaming import build_scanner as _build_scanner
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


@dataclass(frozen=True, slots=True)
class ScannerParams:
    """Convenience parameters for dataset scanning."""

    columns: Sequence[str] | Mapping[str, ds.Expression] | None = None
    filter_expression: ds.Expression | None = None
    batch_size: int | None = None
    batch_readahead: int | None = DEFAULT_ARROW_BATCH_READAHEAD
    fragment_readahead: int | None = DEFAULT_ARROW_FRAGMENT_READAHEAD
    use_threads: bool | None = DEFAULT_ARROW_USE_THREADS
    cache_metadata: bool | None = DEFAULT_ARROW_CACHE_METADATA
    parquet_pre_buffer: bool | None = DEFAULT_ARROW_PARQUET_PRE_BUFFER
    parquet_use_buffered_stream: bool | None = DEFAULT_ARROW_PARQUET_USE_BUFFERED_STREAM
    parquet_buffer_size: int | None = DEFAULT_ARROW_PARQUET_BUFFER_SIZE
    memory_pool: pa.MemoryPool | None = None
    schema: pa.Schema | None = None
    implicit_ordering: bool | None = None
    require_sequenced_output: bool | None = None
    unify_schemas: bool = False
    schema_promote_options: SchemaPromoteOptions = DEFAULT_SCHEMA_PROMOTE_OPTIONS
    metrics_enabled: bool = False

    def to_options(self) -> DatasetScanOptions:
        """Return DatasetScanOptions derived from the parameters.

        Returns
        -------
        DatasetScanOptions
            Dataset scan options object.
        """
        resolved_batch_size = self.batch_size or DEFAULT_ARROW_BATCH_SIZE
        return DatasetScanOptions(
            batch_size=resolved_batch_size,
            batch_readahead=self.batch_readahead,
            fragment_readahead=self.fragment_readahead,
            filter_expression=self.filter_expression,
            use_threads=self.use_threads,
            cache_metadata=self.cache_metadata,
            parquet_pre_buffer=self.parquet_pre_buffer,
            parquet_use_buffered_stream=self.parquet_use_buffered_stream,
            parquet_buffer_size=self.parquet_buffer_size,
            memory_pool=self.memory_pool,
            schema=self.schema,
            columns=self.columns,
            implicit_ordering=self.implicit_ordering,
            require_sequenced_output=self.require_sequenced_output,
            unify_schemas=self.unify_schemas,
            schema_promote_options=self.schema_promote_options,
            metrics_enabled=self.metrics_enabled,
        )


def build_scanner(
    dataset: ds.Dataset,
    *,
    options: DatasetScanOptions | None = None,
    params: ScannerParams | None = None,
) -> ds.Scanner:
    """Build a dataset scanner from options or convenience parameters.

    Parameters
    ----------
    dataset
        Dataset to scan.
    options
        Optional DatasetScanOptions to use directly.
    params
        Optional convenience parameters for scanner construction.

    Returns
    -------
    pyarrow.dataset.Scanner
        Configured dataset scanner.
    """
    if options is None:
        options = (params or ScannerParams()).to_options()
    return _build_scanner(dataset, options=options)


__all__ = [
    "ScannerParams",
    "build_scanner",
]
