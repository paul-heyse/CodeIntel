"""Shared constants for core/build/serving layers."""

from __future__ import annotations

__all__ = [
    "DEFAULT_ARROW_BATCH_READAHEAD",
    "DEFAULT_ARROW_BATCH_SIZE",
    "DEFAULT_ARROW_CPU_COUNT",
    "DEFAULT_ARROW_FRAGMENT_READAHEAD",
    "DEFAULT_ARROW_IO_THREAD_COUNT",
    "DEFAULT_ARROW_IO_THREAD_MULTIPLIER",
    "DEFAULT_ARROW_MIN_IO_THREADS",
    "DEFAULT_ARROW_USE_THREADS",
    "DUCKDB_DIALECT",
    "SCHEMAS",
]

DUCKDB_DIALECT = "duckdb"
"""SQLGlot dialect identifier for DuckDB SQL generation."""

SCHEMAS = ("build", "core", "graph", "analytics", "docs")
"""Database schema names used in the CodeIntel data warehouse."""

DEFAULT_ARROW_BATCH_SIZE = 10_000
"""Default rows per Arrow record batch for streaming exports."""

DEFAULT_ARROW_BATCH_READAHEAD = 16
"""Default record batch readahead for Arrow dataset scans."""

DEFAULT_ARROW_FRAGMENT_READAHEAD = 8
"""Default fragment readahead for Arrow dataset scans."""

DEFAULT_ARROW_USE_THREADS = True
"""Default threading setting for Arrow dataset scans."""

DEFAULT_ARROW_CPU_COUNT: int | None = None
"""Default CPU thread count for Arrow compute (None uses detected CPU count)."""

DEFAULT_ARROW_IO_THREAD_COUNT: int | None = None
"""Default IO thread count for Arrow dataset reads (None uses a CPU-scaled default)."""

DEFAULT_ARROW_IO_THREAD_MULTIPLIER = 2
"""Multiplier applied to CPU count to derive Arrow IO threads when unset."""

DEFAULT_ARROW_MIN_IO_THREADS = 8
"""Minimum IO threads when deriving Arrow IO thread count from CPU."""
