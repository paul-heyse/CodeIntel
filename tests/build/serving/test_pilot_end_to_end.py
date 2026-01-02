"""End-to-end pilot coverage for build → serving snapshots."""

from __future__ import annotations

import duckdb

from tests._helpers.assertions import assert_target_ok, expect_true
from tests._helpers.harnesses.serving_harness import ServingTargetHarness


def _count_rows(con: duckdb.DuckDBPyConnection, sql: str) -> int:
    row = con.execute(sql).fetchone()
    if row is None or row[0] is None:
        return 0
    return int(row[0])


def test_pilot_end_to_end_function_types_snapshot(
    serving_target_harness: ServingTargetHarness,
) -> None:
    """Pilot path should produce serving snapshot data for function types."""
    records = serving_target_harness.run_targets(["function_types", "serving_artifacts"])
    assert_target_ok(records["function_types"])
    assert_target_ok(records["serving_artifacts"])

    manifest = serving_target_harness.publish_snapshot(run_id="pilot-end-to-end")
    con = duckdb.connect(manifest.db_path, read_only=True)
    try:
        metrics_count = _count_rows(con, "SELECT COUNT(*) FROM analytics.function_types")
        expect_true(
            metrics_count > 0,
            message="Expected analytics.function_types rows in serving snapshot.",
        )
        search_count = _count_rows(
            con,
            "SELECT COUNT(*) FROM docs.search_documents WHERE kind = 'function'",
        )
        expect_true(
            search_count > 0,
            message="Expected function search documents in serving snapshot.",
        )
    finally:
        con.close()
