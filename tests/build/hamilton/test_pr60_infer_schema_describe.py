"""PR-60: DuckDB DESCRIBE-based schema inference."""

from __future__ import annotations

import duckdb
import ibis
import pytest

from codeintel.build.schemas.infer_duckdb import infer_table_schema_from_ibis
from tests._helpers.schemas import ensure_production_schemas


def test_pr60_infer_schema_describe_preserves_decimal_and_timestamptz() -> None:
    """Inferred schemas should preserve DECIMAL(38,0) and TIMESTAMPTZ."""
    con = duckdb.connect(":memory:")
    try:
        ensure_production_schemas(con)
        con.execute(
            """
            CREATE TABLE analytics.input (
                goid DECIMAL(38,0),
                event_time TIMESTAMPTZ,
                name VARCHAR,
                count INTEGER
            )
            """
        )

        ibis_con = ibis.duckdb.from_connection(con)
        table = ibis_con.table("input", database="analytics")
        expr = table.select(
            goid=table.goid,
            event_time=table.event_time,
            name=table.name,
            count=table["count"],
        )

        inferred = infer_table_schema_from_ibis(expr=expr, con=con, table_key="analytics.output")

        names = [c.name for c in inferred.columns]
        expected_names = ["goid", "event_time", "name", "count"]
        if names != expected_names:
            pytest.fail(f"Unexpected inferred column names: {names} != {expected_names}")

        types = [c.type for c in inferred.columns]
        expected_types = ["DECIMAL(38,0)", "TIMESTAMPTZ", "VARCHAR", "INTEGER"]
        if types != expected_types:
            pytest.fail(f"Unexpected inferred column types: {types} != {expected_types}")
    finally:
        con.close()
