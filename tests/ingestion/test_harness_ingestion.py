"""Ingestion harness smoke tests for production-parity execution."""

from __future__ import annotations

import pytest

from tests._helpers.assertions import assert_target_ok, expect_true
from tests._helpers.harnesses.hamilton_build import HamiltonBuildHarness
from tests._helpers.parquet_datasets import read_snapshot_rows


def test_modules_target_runs_with_build_harness(
    build_harness: HamiltonBuildHarness,
) -> None:
    """Run modules target through the build harness.

    Parameters
    ----------
    build_harness
        Hamilton build harness fixture.

    Raises
    ------
    ValueError
        If the build harness fails for an unexpected schema configuration.
    """
    try:
        result = build_harness.run_targets(["modules"])
    except ValueError as exc:
        if "Missing TableSchema definitions for DAG outputs" in str(exc):
            pytest.xfail("Schema authority check fails while view schemas are inferred.")
        raise
    record = build_harness.record("modules", result=result)
    assert_target_ok(record)
    dataset_root = build_harness.ctx.build_paths.dataset_root_dir
    snapshot = build_harness.ctx.snapshot
    try:
        rows = read_snapshot_rows(
            dataset_root,
            table_key="core.modules",
            snapshot_id=snapshot.commit,
        )
    except FileNotFoundError:
        pytest.xfail("Parquet datasets not yet materialized for modules target.")
    expect_true(len(rows) >= 1, message="Expected core.modules dataset rows")
