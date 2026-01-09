"""Tests for subsystem cache row builders."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

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
            extras,
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
            {"modules": ["pkg.mod"], "entrypoints": []},
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
        subsystems = (
            ctx.gateway.con.execute("SELECT * FROM analytics.subsystems").arrow().read_all()
        )
        metrics = (
            ctx.gateway.con.execute("SELECT * FROM analytics.subsystem_graph_metrics")
            .arrow()
            .read_all()
        )
        profile_rows = build_subsystem_profile_cache_rows(
            ctx.snapshot,
            subsystems,
            metrics,
        )
    finally:
        ctx.close()

    if not profile_rows:
        pytest.fail("Expected subsystem profile cache rows to be built")
    if profile_rows[0]["subsystem_id"] != "subsys-1":
        pytest.fail("Expected profile cache row to reference seeded subsystem")
