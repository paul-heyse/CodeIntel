"""Integration tests for the Hamilton scip target."""

from __future__ import annotations

from tests._helpers.assertions import assert_row_count, assert_target_ok, expect_true
from tests._helpers.harnesses.hamilton_build import HamiltonBuildHarness


def test_scip_target_writes_tables(build_harness: HamiltonBuildHarness) -> None:
    """Ensure scip target writes scip tables when SCIP binaries are available."""
    result = build_harness.run_targets(["scip"])
    record = build_harness.record("scip", result=result)
    assert_target_ok(record)
    assert_row_count(record.row_counts, "core.scip_symbols", min_rows=1)
    assert_row_count(record.row_counts, "core.scip_occurrences", min_rows=1)

    scip_dir = build_harness.artifacts.paths.scip_dir
    index_scip = scip_dir / "index.scip"
    index_json = scip_dir / "index.json"
    expect_true(
        index_scip.is_file(),
        message="index.scip was not created under build/scip",
    )
    expect_true(
        index_json.is_file(),
        message="index.json was not created under build/scip",
    )
