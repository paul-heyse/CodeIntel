"""Subsystem inference tests covering clustering and risk aggregation."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest

from codeintel.analytics.subsystems import build_subsystems
from codeintel.config import ConfigBuilder, SnapshotInit
from tests._helpers.assertions import expect_equal, expect_in, expect_length
from tests._helpers.context import TestContext
from tests._helpers.scenarios import TestScenario
from tests._helpers.seeds.subsystems_analytics import (
    SubsystemAnalyticsPack,
)

EXPECTED_SUBSYSTEMS = 2
EXPECTED_MEMBERSHIP_COUNT = 6
EXPECTED_HIGH_RISK_COUNT = 1
EXPECTED_MODULES = {
    "pkg.api",
    "pkg.core",
    "pkg.misc",
    "pkg.mod_a",
    "pkg.mod_b",
    "pkg.mod_c",
}


@pytest.fixture
def subsystem_ctx(tmp_path: Path) -> Iterator[TestContext]:
    """Create a test context seeded with subsystem analytics data.

    Yields
    ------
    Iterator[TestContext]
        Context seeded with subsystem analytics data.
    """
    ctx = TestScenario.minimal().with_seeds(SubsystemAnalyticsPack()).build(tmp_path)
    try:
        yield ctx
    finally:
        ctx.close()


def test_subsystems_cluster_and_risk_aggregation(subsystem_ctx: TestContext) -> None:
    """Cluster modules and aggregate risk across subsystems using seeded pack."""
    cfg = ConfigBuilder.from_snapshot(
        snapshot=SnapshotInit(
            repo=subsystem_ctx.repo,
            commit=subsystem_ctx.commit,
            repo_root=subsystem_ctx.repo_root,
        ),
    ).analytics.subsystems(
        max_subsystems=2,
        min_modules=1,
    )
    build_subsystems(subsystem_ctx.gateway, cfg)

    subsystems = subsystem_ctx.query(
        """
        SELECT subsystem_id, modules_json, risk_level, high_risk_function_count
        FROM analytics.subsystems
        """
    )
    expect_length(subsystems, EXPECTED_SUBSYSTEMS)

    subs_by_id = {
        str(row.subsystem_id): (
            set(json.loads(str(row.modules_json))),
            str(row.risk_level),
            int(str(row.high_risk_function_count))
            if row.high_risk_function_count is not None
            else 0,
        )
        for row in subsystems
    }

    # Identify clusters by risk level rather than ID (clustering assigns new IDs)
    high_cluster = next(
        (modules, risk, high_count)
        for modules, risk, high_count in subs_by_id.values()
        if risk == "high"
    )
    low_cluster = next(
        (modules, risk, high_count)
        for modules, risk, high_count in subs_by_id.values()
        if risk == "low"
    )

    high_modules, _high_risk, high_count = high_cluster
    low_modules, _low_risk, low_count = low_cluster

    expect_in("pkg.core", high_modules)
    expect_equal(high_count, EXPECTED_HIGH_RISK_COUNT)
    expect_in("pkg.misc", low_modules)
    expect_equal(low_count, 0)

    memberships = subsystem_ctx.query(
        "SELECT subsystem_id, module FROM analytics.subsystem_modules"
    )
    expect_length(memberships, EXPECTED_MEMBERSHIP_COUNT)
    assigned_modules = {str(row.module) for row in memberships}
    expect_equal(assigned_modules, EXPECTED_MODULES)
