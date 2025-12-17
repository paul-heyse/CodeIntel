"""Materializers and I/O adapters for Hamilton build graphs.

This package provides Hamilton-native I/O building blocks (DataSavers/DataLoaders)
that make side effects (DuckDB writes, file outputs) visible in the Hamilton DAG.
"""

from __future__ import annotations

from codeintel.build.hamilton.materializers.duckdb_saver import DuckDBIbisTableSaver

__all__ = [
    "DuckDBIbisTableSaver",
]
