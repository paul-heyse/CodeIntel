"""Tests for subsystem cache row builders."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

import polars as pl
import pytest

from codeintel.build.analytics.subsystems.cache import (
    build_subsystem_profile_cache_rows,
)
from tests._helpers import TestScenario

if TYPE_CHECKING:
    from pathlib import Path

    from tests._helpers import TestContext


def _seed_subsystem(ctx: TestContext) -> None:
    """Insert a minimal subsystem row for cache row tests."""
    ctx.gateway.con.execute(
        """
        INSERT INTO analytics.subsystems (
            repo,
            commit,
            subsystem_id,
            name,
            description,
            module_count,
            modules_json,
            entrypoints_json,
            internal_edge_count,
            external_edge_count,
            fan_in,
            fan_out,
            function_count,
            avg_risk_score,
            max_risk_score,
            high_risk_function_count,
            risk_level,
            created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            ctx.repo,
            ctx.commit,
            "subsys-1",
            "demo_subsystem",
            "demo description",
            1,
            ["pkg.mod"],
            None,
            0,
            0,
            0,
            0,
            1,
            0.1,
            0.2,
            0,
            "low",
            datetime.now(UTC),
        ],
    )


def test_subsystem_cache_rows_build(tmp_path: Path) -> None:
    """Building subsystem cache rows should return matching subsystem entries."""
    ctx = TestScenario.minimal().build(tmp_path)
    try:
        _seed_subsystem(ctx)
        subsystems = pl.from_arrow(
            ctx.gateway.relation_from_table_key("analytics.subsystems").arrow()
        )
        metrics = pl.from_arrow(
            ctx.gateway.relation_from_table_key("analytics.subsystem_graph_metrics").arrow()
        )
        subsystems_frame = (
            subsystems.lazy() if isinstance(subsystems, pl.DataFrame) else pl.DataFrame().lazy()
        )
        metrics_frame = (
            metrics.lazy() if isinstance(metrics, pl.DataFrame) else pl.DataFrame().lazy()
        )
        profile_rows = build_subsystem_profile_cache_rows(
            ctx.snapshot,
            subsystems_frame,
            metrics_frame,
        )
    finally:
        ctx.close()

    if not profile_rows:
        pytest.fail("Expected subsystem profile cache rows to be built")
    if profile_rows[0]["subsystem_id"] != "subsys-1":
        pytest.fail("Expected profile cache row to reference seeded subsystem")
