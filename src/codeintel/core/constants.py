"""Shared constants for core/build/serving layers."""

from __future__ import annotations

__all__ = [
    "DEFAULT_ARROW_BATCH_READAHEAD",
    "DEFAULT_ARROW_BATCH_SIZE",
    "DEFAULT_ARROW_CACHE_METADATA",
    "DEFAULT_ARROW_CPU_COUNT",
    "DEFAULT_ARROW_FRAGMENT_READAHEAD",
    "DEFAULT_ARROW_IO_THREAD_COUNT",
    "DEFAULT_ARROW_IO_THREAD_MULTIPLIER",
    "DEFAULT_ARROW_MIN_IO_THREADS",
    "DEFAULT_ARROW_PARQUET_BUFFER_SIZE",
    "DEFAULT_ARROW_PARQUET_PRE_BUFFER",
    "DEFAULT_ARROW_PARQUET_USE_BUFFERED_STREAM",
    "DEFAULT_ARROW_PROVENANCE_COLUMNS",
    "DEFAULT_ARROW_USE_THREADS",
    "DUCKDB_DIALECT",
    "SCHEMAS",
]

DUCKDB_DIALECT = "duckdb"
"""SQLGlot dialect identifier for DuckDB SQL generation."""

SCHEMAS = ("build", "core", "graph", "analytics", "docs")
"""Database schema names used in the CodeIntel data warehouse."""

DEFAULT_ARROW_BATCH_SIZE = 131_072
"""Default rows per Arrow record batch for streaming exports."""

DEFAULT_ARROW_BATCH_READAHEAD = 64
"""Default record batch readahead for Arrow dataset scans."""

DEFAULT_ARROW_FRAGMENT_READAHEAD = 16
"""Default fragment readahead for Arrow dataset scans."""

DEFAULT_ARROW_CACHE_METADATA = True
"""Default metadata caching toggle for Arrow dataset scans."""

DEFAULT_ARROW_USE_THREADS = True
"""Default threading setting for Arrow dataset scans."""

DEFAULT_ARROW_PARQUET_PRE_BUFFER = True
"""Default pre-buffer toggle for Parquet fragment scans."""

DEFAULT_ARROW_PARQUET_USE_BUFFERED_STREAM = False
"""Default buffered stream toggle for Parquet fragment scans."""

DEFAULT_ARROW_PARQUET_BUFFER_SIZE = 8_388_608
"""Default buffered stream size (bytes) for Parquet fragment scans."""

DEFAULT_ARROW_CPU_COUNT: int | None = None
"""Default CPU thread count for Arrow compute (None uses detected CPU count)."""

DEFAULT_ARROW_IO_THREAD_COUNT: int | None = None
"""Default IO thread count for Arrow dataset reads (None uses a CPU-scaled default)."""

DEFAULT_ARROW_IO_THREAD_MULTIPLIER = 2
"""Multiplier applied to CPU count to derive Arrow IO threads when unset."""

DEFAULT_ARROW_MIN_IO_THREADS = 8
"""Minimum IO threads when deriving Arrow IO thread count from CPU."""

DEFAULT_ARROW_PROVENANCE_COLUMNS = ("__filename", "__fragment_index", "__batch_index")
"""Default provenance columns for dataset scan debugging."""
