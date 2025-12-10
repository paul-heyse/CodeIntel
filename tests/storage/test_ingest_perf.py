"""Sanity checks for ingest macro performance versus prepared inserts."""

from __future__ import annotations

import pytest

from codeintel.storage.gateway import DuckDBConnection, StorageGateway
from tests._helpers.macros import assert_macro_perf, measure_ingest_perf


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


@pytest.mark.parametrize(
    "table_key",
    ["analytics.function_metrics", "analytics.function_effects"],
)
def test_ingest_macro_perf_reasonable(macro_gateway: StorageGateway, table_key: str) -> None:
    """Macro-based ingest should be within a reasonable factor of prepared inserts."""
    rows = _sample_rows(macro_gateway.con, table_key, count=15)
    result = measure_ingest_perf(macro_gateway, table_key, rows)
    assert_macro_perf(result)
