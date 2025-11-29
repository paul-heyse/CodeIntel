"""Benchmark helpers for subsystem cache performance."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from duckdb import DuckDBPyConnection

from codeintel.analytics.subsystems.materialize import refresh_subsystem_caches
from codeintel.storage.gateway import open_memory_gateway


def _seed_subsystem(con: DuckDBPyConnection, *, repo: str, commit: str) -> None:
    con.execute(
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
            repo,
            commit,
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


def test_refresh_and_benchmark_returns_timings() -> None:
    """Refreshing caches with benchmarking enabled should emit timing data."""
    gateway = open_memory_gateway(ensure_views=True, validate_schema=False)
    _seed_subsystem(gateway.con, repo="demo/repo", commit="deadbeef")
    result = refresh_subsystem_caches(
        gateway,
        repo="demo/repo",
        commit="deadbeef",
        benchmark=True,
        benchmark_limit=5,
    )
    if result is None:
        pytest.fail("Expected benchmark results when benchmark flag is set")
    if result.profile_view_ms < 0 or result.profile_cache_ms < 0:
        pytest.fail("Profile timing metrics should be non-negative")
    if result.coverage_view_ms < 0 or result.coverage_cache_ms < 0:
        pytest.fail("Coverage timing metrics should be non-negative")
    timings = result.as_dict()
    if "profile_speedup" not in timings or "coverage_speedup" not in timings:
        pytest.fail("Speedup metrics should be present in benchmark output")
