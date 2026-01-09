"""Subsystem inference tests covering clustering and risk aggregation."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import duckdb
import polars as pl
import pytest

from codeintel.build.analytics.subsystems.materialize import (
    SubsystemBuildInputs,
    SubsystemOptions,
    build_subsystem_rows,
)
from codeintel.config.primitives import SnapshotRef
from codeintel.core.serialization.payload import decode_payload
from tests._helpers.assertions import expect_equal, expect_length
from tests._helpers.scenarios import TestScenario
from tests._helpers.seeds.subsystems_analytics import (
    SubsystemAnalyticsPack,
)

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from tests._helpers.context import TestContext

EXPECTED_SUBSYSTEMS = 2
EXPECTED_MEMBERSHIP_COUNT = 6
EXPECTED_HIGH_RISK_COUNT = 0
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


def _frame_from_table(ctx: TestContext, table_key: str) -> pl.DataFrame:
    try:
        relation = ctx.gateway.relation_from_table_key(table_key)
    except duckdb.Error:
        return pl.DataFrame()
    frame = pl.from_arrow(relation.arrow())
    return frame if isinstance(frame, pl.DataFrame) else pl.DataFrame()


def test_subsystems_cluster_and_risk_aggregation(subsystem_ctx: TestContext) -> None:
    """Cluster modules and aggregate risk across subsystems using seeded pack."""
    snapshot = SnapshotRef(
        repo=subsystem_ctx.repo,
        commit=subsystem_ctx.commit,
        repo_root=subsystem_ctx.repo_root,
    )
    options = SubsystemOptions(
        max_subsystems=2,
        min_modules=1,
    )
    rows = build_subsystem_rows(
        snapshot,
        SubsystemBuildInputs(
            modules_frame=_frame_from_table(subsystem_ctx, "core.modules"),
            import_graph_edges_frame=_frame_from_table(subsystem_ctx, "graph.import_graph_edges"),
            symbol_use_edges_frame=_frame_from_table(subsystem_ctx, "graph.symbol_use_edges"),
            config_values_frame=_frame_from_table(subsystem_ctx, "analytics.config_values"),
            options=options,
        ),
    )
    backend = subsystem_ctx.gateway.policy
    backend.delete_for_snapshot(
        "analytics.subsystems",
        repo=snapshot.repo,
        commit=snapshot.commit,
    )
    backend.delete_for_snapshot(
        "analytics.subsystem_modules",
        repo=snapshot.repo,
        commit=snapshot.commit,
    )
    backend.bulk_insert("analytics.subsystems", rows.subsystem_rows)
    backend.bulk_insert("analytics.subsystem_modules", rows.membership_rows)

    subsystems = subsystem_ctx.query(
        """
        SELECT
            subsystem_id,
            struct_extract(extras, 'modules') AS modules_json,
            risk_level,
            high_risk_function_count
        FROM analytics.subsystems
        """
    )
    expect_length(subsystems, EXPECTED_SUBSYSTEMS)

    subs_by_id = {
        str(row.subsystem_id): (
            set(
                cast(
                    "list[str]",
                    decode_payload(row.modules_json),
                )
            ),
            str(row.risk_level),
            int(str(row.high_risk_function_count))
            if row.high_risk_function_count is not None
            else 0,
        )
        for row in subsystems
    }

    for _, risk_level, high_count in subs_by_id.values():
        expect_equal(risk_level, "low")
        expect_equal(high_count, EXPECTED_HIGH_RISK_COUNT)

    memberships = subsystem_ctx.query(
        "SELECT subsystem_id, module FROM analytics.subsystem_modules"
    )
    expect_length(memberships, EXPECTED_MEMBERSHIP_COUNT)
    assigned_modules = {str(row.module) for row in memberships}
    expect_equal(assigned_modules, EXPECTED_MODULES)
