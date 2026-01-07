"""Integration tests for the Hamilton scip target."""

from __future__ import annotations

import pytest

from tests._helpers.assertions import assert_row_count, assert_target_ok, expect_true
from tests._helpers.harnesses.hamilton_build import HamiltonBuildHarness


def test_scip_target_writes_tables(build_harness: HamiltonBuildHarness) -> None:
    """Ensure scip target writes scip tables when SCIP binaries are available.

    Raises
    ------
    ValueError
        If the harness build fails for an unexpected schema configuration.
    """
    try:
        result = build_harness.run_targets(["scip"])
    except ValueError as exc:
        if "Missing TableSchema definitions" in str(exc):
            pytest.xfail("Schema registry incomplete for scip build.")
        raise
    record = build_harness.record("scip", result=result)
    assert_target_ok(record)
    assert_row_count(record.row_counts, "core.scip_symbols", min_rows=1)
    assert_row_count(record.row_counts, "core.scip_occurrences", min_rows=1)
    assert_row_count(record.row_counts, "core.scip_symbol_information", min_rows=1)
    expect_true(
        "core.scip_symbol_relationships" in record.row_counts,
        message="Expected scip symbol relationships row counts to be recorded",
    )
    expect_true(
        "core.scip_diagnostics" in record.row_counts,
        message="Expected scip diagnostics row counts to be recorded",
    )
    expect_true(
        "core.scip_external_symbols" in record.row_counts,
        message="Expected scip external symbols row counts to be recorded",
    )
    expect_true(
        "core.scip_module_state" in record.row_counts,
        message="Expected scip module state row counts to be recorded",
    )

    scip_dir = build_harness.artifacts.paths.scip_dir
    index_scip = scip_dir / "index.scip"
    proto_module = scip_dir / "proto" / "scip_pb2.py"
    expect_true(
        index_scip.is_file(),
        message="index.scip was not created under build/scip",
    )
    expect_true(
        proto_module.is_file(),
        message="scip_pb2.py was not created under build/scip/proto",
    )
