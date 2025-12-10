"""Benchmark helpers for subsystem cache performance."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from codeintel.analytics.subsystems.materialize import refresh_subsystem_caches
from tests._helpers import TestContext, create_test_context


def _seed_subsystem(ctx: TestContext) -> None:
    """Insert a minimal subsystem row for cache refresh tests."""
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


def test_refresh_and_benchmark_returns_timings(tmp_path: Path) -> None:
    """Refreshing caches with benchmarking enabled should emit timing data."""
    ctx = create_test_context(tmp_path)
    _seed_subsystem(ctx)
    result = refresh_subsystem_caches(
        ctx.gateway,
        repo=ctx.repo,
        commit=ctx.commit,
        benchmark=True,
        benchmark_limit=5,
    )
    ctx.close()
    if result is None:
        pytest.fail("Expected benchmark results when benchmark flag is set")
    if result.profile_view_ms < 0 or result.profile_cache_ms < 0:
        pytest.fail("Profile timing metrics should be non-negative")
    if result.coverage_view_ms < 0 or result.coverage_cache_ms < 0:
        pytest.fail("Coverage timing metrics should be non-negative")
    timings = result.as_dict()
    if "profile_speedup" not in timings or "coverage_speedup" not in timings:
        pytest.fail("Speedup metrics should be present in benchmark output")
