"""Ingestion harness smoke tests for production-parity execution."""

from __future__ import annotations

from tests._helpers.assertions import assert_table_has_rows, assert_target_ok
from tests._helpers.harnesses.hamilton_build import HamiltonBuildHarness


def test_modules_target_runs_with_build_harness(
    build_harness: HamiltonBuildHarness,
) -> None:
    """Run modules target through the build harness.

    Parameters
    ----------
    build_harness
        Hamilton build harness fixture.
    """
    result = build_harness.run_targets(["modules"])
    record = build_harness.record("modules", result=result)
    assert_target_ok(record)
    assert_table_has_rows(build_harness.ctx.gateway, "core.modules", min_rows=1)
