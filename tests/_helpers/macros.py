"""Shared helpers for ingest macro coverage and performance tests."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import pytest

from codeintel.ingestion.adapters.duckdb_storage import DuckDBStorageAdapter
from codeintel.ingestion.infrastructure.macros import INGEST_MACRO_TABLES, macro_exists
from codeintel.storage.gateway import StorageGateway
from codeintel.storage.gateway.protocol import DuckDBConnection
from codeintel.storage.macros import list_ingest_macros
from codeintel.storage.metadata import INGEST_MACROS
from codeintel.storage.sql.builder import prepared_statements_dynamic
from tests._helpers.assertions import expect_true


@dataclass(frozen=True)
class MacroPerfResult:
    """Measured ingest timings for macro vs prepared statements."""

    table_key: str
    macro_elapsed: float
    prepared_elapsed: float
    rows_written: int


def assert_all_ingest_macros(con: DuckDBConnection) -> None:
    """Assert that all ingest macros from metadata are registered on the connection."""
    macros = list_ingest_macros(con)
    missing = {macro.lower() for macro in INGEST_MACROS.values() if macro.lower() not in macros}
    if missing:
        pytest.fail(f"Missing ingest macros: {sorted(missing)}")


def assert_ingest_macros_registered(con: DuckDBConnection) -> None:
    """Assert macros exist for every table key in INGEST_MACRO_TABLES."""
    for table_key in sorted(INGEST_MACRO_TABLES):
        if not macro_exists(con, table_key):
            _, table_name = table_key.split(".", maxsplit=1)
            macro_name = f"metadata.ingest_{table_name}"
            pytest.fail(f"Missing ingest macro {macro_name} for {table_key}")


def measure_ingest_perf(
    gateway: StorageGateway,
    table_key: str,
    rows: list[tuple[object, ...]],
) -> MacroPerfResult:
    """
    Measure macro vs prepared insert performance for a table.

    Returns
    -------
    MacroPerfResult
        Timings and row counts for macro and prepared inserts.

    Raises
    ------
    ValueError
        If the table_key is not supported for perf measurement.
    """
    con = gateway.con
    adapter = DuckDBStorageAdapter(gateway)

    if table_key == "analytics.function_metrics":
        delete_sql = "DELETE FROM analytics.function_metrics WHERE 1=1"
    elif table_key == "analytics.function_effects":
        delete_sql = "DELETE FROM analytics.function_effects WHERE 1=1"
    else:
        message = f"Unsupported table for ingest perf: {table_key}"
        raise ValueError(message)

    con.execute(delete_sql)
    start_macro = perf_counter()
    result = adapter.write_batch(table_key, rows)
    macro_elapsed = perf_counter() - start_macro

    con.execute(delete_sql)
    stmts = prepared_statements_dynamic(con, table_key)
    start_prepared = perf_counter()
    con.executemany(stmts.insert_sql, rows)
    prepared_elapsed = perf_counter() - start_prepared

    return MacroPerfResult(
        table_key=table_key,
        macro_elapsed=macro_elapsed,
        prepared_elapsed=prepared_elapsed,
        rows_written=result.rows_written,
    )


def assert_macro_perf(
    result: MacroPerfResult, *, slowdown_factor: float = 10.0, slack: float = 0.05
) -> None:
    """Assert macro ingest is within acceptable bound of prepared inserts."""
    expect_true(
        result.rows_written > 0,
        message=f"No rows written for {result.table_key}",
    )
    allowed = result.prepared_elapsed * slowdown_factor + slack
    if result.macro_elapsed > allowed:
        pytest.fail(
            f"Macro ingest slower than expected for {result.table_key}: "
            f"macro={result.macro_elapsed:.6f}s prepared={result.prepared_elapsed:.6f}s "
            f"allowed<={allowed:.6f}s"
        )


__all__ = [
    "MacroPerfResult",
    "assert_all_ingest_macros",
    "assert_ingest_macros_registered",
    "assert_macro_perf",
    "measure_ingest_perf",
]
