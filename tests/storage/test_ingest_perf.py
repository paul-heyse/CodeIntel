"""Sanity checks for ingest macro performance versus prepared inserts."""

from __future__ import annotations

from time import perf_counter

import pytest

from codeintel.config.schemas.sql_builder import prepared_statements_dynamic
from codeintel.ingestion.ingest_service import ingest_via_macro
from codeintel.storage.gateway import DuckDBConnection, StorageGateway


def _sample_rows(con: DuckDBConnection, table_key: str, count: int) -> list[tuple[object, ...]]:
    cols = con.execute(
        """
        SELECT column_name, data_type, is_nullable
        FROM information_schema.columns
        WHERE table_schema = ? AND table_name = ?
        ORDER BY ordinal_position
        """,
        table_key.split(".", maxsplit=1),
    ).fetchall()

    def sample(i: int) -> tuple[object, ...]:
        values: list[object] = []
        for _, data_type, is_nullable in cols:
            nullable = str(is_nullable).upper() == "YES"
            dtype = str(data_type).upper()
            if nullable:
                values.append(None)
                continue
            if "INT" in dtype:
                values.append(i)
            elif "DOUBLE" in dtype or "DECIMAL" in dtype or "FLOAT" in dtype:
                values.append(float(i))
            elif "BOOLEAN" in dtype:
                values.append(False)
            elif "DATE" in dtype:
                values.append("1970-01-01")
            elif "TIME" in dtype:
                values.append("1970-01-01 00:00:00")
            else:
                values.append(f"value_{i}")
        return tuple(values)

    return [sample(i) for i in range(count)]


def test_ingest_macro_perf_reasonable(fresh_gateway: StorageGateway) -> None:
    """Macro-based ingest should be within a reasonable factor of prepared inserts."""
    con = fresh_gateway.con
    table_keys = ["analytics.function_metrics", "analytics.function_effects"]
    for table_key in table_keys:
        rows = _sample_rows(con, table_key, count=15)

        con.execute(
            "DELETE FROM analytics.function_metrics WHERE 1=1"
            if table_key == "analytics.function_metrics"
            else "DELETE FROM analytics.function_effects WHERE 1=1"
        )
        start_macro = perf_counter()
        macro_inserted = ingest_via_macro(con, table_key, rows)
        macro_elapsed = perf_counter() - start_macro

        con.execute(
            "DELETE FROM analytics.function_metrics WHERE 1=1"
            if table_key == "analytics.function_metrics"
            else "DELETE FROM analytics.function_effects WHERE 1=1"
        )
        stmts = prepared_statements_dynamic(con, table_key)
        start_prepared = perf_counter()
        con.executemany(stmts.insert_sql, rows)
        prepared_elapsed = perf_counter() - start_prepared

        if macro_inserted != len(rows):
            pytest.fail(f"Inserted rows {macro_inserted} != expected {len(rows)} for {table_key}")
        # Allow a generous factor to avoid flakiness while catching regressions.
        if macro_elapsed > prepared_elapsed * 5 + 0.05:
            pytest.fail(
                f"Macro ingest slower than expected for {table_key}: macro={macro_elapsed:.6f}s "
                f"prepared={prepared_elapsed:.6f}s"
            )
