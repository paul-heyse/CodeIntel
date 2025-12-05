"""Integration tests for analytics.tests_profiles module.

This module consolidates integration tests for test profiles including
coverage input processing and plugin runtime execution.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from codeintel.analytics.core.build_bridge import (
    AnalyticsPlanRequest,
    AnalyticsRunContext,
    plan_analytics_plugin_run,
    run_analytics_plugins,
)
from codeintel.analytics.plugins import TEST_PROFILE_PLUGIN
from codeintel.analytics.testing.coverage import inputs as coverage_inputs
from codeintel.config import BehavioralCoverageStepConfig, ConfigBuilder, TestProfileStepConfig
from codeintel.config.steps_graphs import GraphPluginPolicy, GraphRunScope
from tests._helpers import provisioned_gateway
from tests._helpers.constants import DEFAULT_COMMIT, DEFAULT_REPO
from tests._helpers.factories import make_snapshot

if TYPE_CHECKING:
    from duckdb import DuckDBPyConnection


# =============================================================================
# Shared Test Helpers
# =============================================================================


def _snapshot_cfg() -> tuple[TestProfileStepConfig, BehavioralCoverageStepConfig]:
    """Create test and behavioral coverage configs from a snapshot.

    Returns
    -------
    tuple[TestProfileStepConfig, BehavioralCoverageStepConfig]
        Tuple of test profile and behavioral coverage configs.
    """
    snapshot = make_snapshot()
    return TestProfileStepConfig(snapshot=snapshot), BehavioralCoverageStepConfig(snapshot=snapshot)


def _seed_sample_data(con: DuckDBPyConnection) -> None:
    """Seed sample data into the DuckDB fixture for coverage testing."""
    edges = [
        ("t1", 1, "mod.a", 5, 10, DEFAULT_REPO, DEFAULT_COMMIT, "a.py", "A::t1a"),
        ("t1", 2, "mod.b", 8, 10, DEFAULT_REPO, DEFAULT_COMMIT, "b.py", "B::t1b"),
        ("t2", 2, "mod.b", 6, 10, DEFAULT_REPO, DEFAULT_COMMIT, "b.py", "B::t2"),
    ]
    con.executemany(
        """
        INSERT INTO analytics.test_coverage_edges
        (test_id, function_goid_h128, module, covered_lines, executable_lines, repo, commit, rel_path, qualname)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        edges,
    )
    catalog = [
        ("t1", DEFAULT_REPO, DEFAULT_COMMIT, "passed", 500.0, False),
        ("t2", DEFAULT_REPO, DEFAULT_COMMIT, "failed", 2500.0, True),
    ]
    con.executemany(
        "INSERT INTO analytics.test_catalog VALUES (?, ?, ?, ?, ?, ?)",
        catalog,
    )
    subsystems = [
        ("mod.a", "subA", DEFAULT_REPO, DEFAULT_COMMIT),
        ("mod.b", "subB", DEFAULT_REPO, DEFAULT_COMMIT),
    ]
    con.executemany(
        "INSERT INTO analytics.subsystem_modules VALUES (?, ?, ?, ?)",
        subsystems,
    )
    con.executemany(
        "INSERT INTO analytics.subsystems VALUES (?, ?, ?, ?, ?)",
        [
            ("subA", "Subsystem A", 0.2, DEFAULT_REPO, DEFAULT_COMMIT),
            ("subB", "Subsystem B", 0.8, DEFAULT_REPO, DEFAULT_COMMIT),
        ],
    )


def _aggregate_functions(con: DuckDBPyConnection, repo: str, commit: str) -> dict[str, Any]:
    """Compute function coverage summaries from the fixture tables.

    Parameters
    ----------
    con
        DuckDB connection.
    repo
        Repository identifier.
    commit
        Commit identifier.

    Returns
    -------
    dict[str, Any]
        Mapping of test_id to function coverage summary.
    """
    rows = con.execute(
        """
        SELECT
            test_id,
            function_goid_h128,
            covered_lines,
            executable_lines
        FROM analytics.test_coverage_edges
        WHERE repo = ? AND commit = ?
        """,
        [repo, commit],
    ).fetchall()

    by_test: dict[str, list[tuple[int, int, int]]] = {}
    for test_id, func_goid, covered, executable in rows:
        by_test.setdefault(test_id, []).append((int(func_goid), int(covered), int(executable or 0)))

    primary_threshold = 0.4
    result: dict[str, Any] = {}
    for test_id, funcs in by_test.items():
        functions = [
            {"function_goid_h128": goid, "covered_lines": cov, "executable_lines": exe}
            for goid, cov, exe in funcs
        ]
        primary = [goid for goid, cov, exe in funcs if exe > 0 and (cov / exe) >= primary_threshold]
        result[test_id] = {"functions": functions, "count": len(functions), "primary": primary}
    return result


def _aggregate_subsystems(con: DuckDBPyConnection, repo: str, commit: str) -> dict[str, Any]:
    """Compute subsystem coverage summaries from the fixture tables.

    Parameters
    ----------
    con
        DuckDB connection.
    repo
        Repository identifier.
    commit
        Commit identifier.

    Returns
    -------
    dict[str, Any]
        Mapping of test_id to subsystem coverage summary.
    """
    rows = con.execute(
        """
        SELECT
            e.test_id,
            sm.subsystem_id
        FROM analytics.test_coverage_edges AS e
        LEFT JOIN analytics.subsystem_modules AS sm
          ON sm.module = e.module AND sm.repo = e.repo AND sm.commit = e.commit
        WHERE e.repo = ? AND e.commit = ?
        """,
        [repo, commit],
    ).fetchall()

    by_test: dict[str, list[str | None]] = {}
    for test_id, subsystem_id in rows:
        by_test.setdefault(test_id, []).append(subsystem_id)

    result: dict[str, Any] = {}
    for test_id, subs in by_test.items():
        subsystems = [{"subsystem_id": sid} for sid in subs if sid is not None]
        primary_subsystem = subsystems[0]["subsystem_id"] if subsystems else None
        result[test_id] = {
            "subsystems": subsystems,
            "count": len(subsystems),
            "primary_subsystem_id": primary_subsystem,
            "max_risk_score": None,
        }
    return result


# =============================================================================
# Coverage Input Tests
# =============================================================================


def test_aggregate_test_coverage_by_function_in_memory(
    coverage_profiles_conn: DuckDBPyConnection,
) -> None:
    """Validate function coverage aggregation against small DuckDB fixture."""
    _seed_sample_data(coverage_profiles_conn)
    test_cfg, _ = _snapshot_cfg()
    result = coverage_inputs.aggregate_test_coverage_by_function(
        coverage_profiles_conn, test_cfg, loader=_aggregate_functions
    )
    if set(result.keys()) != {"t1", "t2"}:
        pytest.fail("Expected both tests t1 and t2 in coverage results.")
    t1 = result["t1"]
    t2 = result["t2"]
    expected_t1_count = 2
    expected_t2_count = 1
    if t1.count != expected_t1_count or t2.count != expected_t2_count:
        pytest.fail("Function counts did not match expectations.")
    primary_expected = {1, 2}
    if set(t1.primary) != primary_expected or t2.primary != [2]:
        pytest.fail("Primary function selection did not match expectations.")


def test_aggregate_test_coverage_by_subsystem_in_memory(
    coverage_profiles_conn: DuckDBPyConnection,
) -> None:
    """Validate subsystem coverage aggregation against small DuckDB fixture."""
    _seed_sample_data(coverage_profiles_conn)
    _, beh_cfg = _snapshot_cfg()
    result = coverage_inputs.aggregate_test_coverage_by_subsystem(
        coverage_profiles_conn, beh_cfg, loader=_aggregate_subsystems
    )
    if set(result.keys()) != {"t1", "t2"}:
        pytest.fail("Expected both tests t1 and t2 in subsystem results.")
    t1 = result["t1"]
    t2 = result["t2"]
    expected_t1_count = 2
    expected_t2_count = 1
    if t1.count != expected_t1_count or t2.count != expected_t2_count:
        pytest.fail("Subsystem counts did not match expectations.")
    primary_subs = {"subA", "subB"}
    if t1.primary_subsystem_id not in primary_subs:
        pytest.fail("Primary subsystem for t1 not in expected set.")
    if t2.primary_subsystem_id != "subB":
        pytest.fail("Primary subsystem for t2 did not match expectations.")


# =============================================================================
# Plugin Runtime Tests
# =============================================================================


def test_tests_profile_plugin_runtime(tmp_path: Path) -> None:
    """Execute the test profile plugin through the analytics harness."""
    with provisioned_gateway(tmp_path) as ctx:
        builder = ConfigBuilder.from_snapshot(
            ctx.repo,
            ctx.commit,
            ctx.repo_root,
            build_dir=ctx.build_dir,
            db_path=ctx.db_path,
            document_output_dir=ctx.document_output_dir,
        )
        cfg = builder.test_profile()
        policy = GraphPluginPolicy()
        scope = GraphRunScope()

        plan = plan_analytics_plugin_run(
            AnalyticsPlanRequest(
                plugin_names=(TEST_PROFILE_PLUGIN.metadata.name,),
                policy=policy,
                repo=cfg.repo,
                commit=cfg.commit,
                scope=scope,
                prior_manifest={},
                cfg_options={},
                runtime_options={},
                run_id="test-profile-run",
            )
        )

        report = run_analytics_plugins(
            plan=plan,
            run_context=AnalyticsRunContext(
                gateway=ctx.gateway,
                graph_runtime=None,
                cfgs={"test_profile": cfg},
                extra={},
                catalog_provider=None,
            ),
        )

        if len(report.records) != 1:
            msg = "Expected single run record for test profile plugin."
            pytest.fail(msg)
        rec = report.records[0]
        if rec.name != TEST_PROFILE_PLUGIN.metadata.name:
            msg = "Unexpected plugin recorded."
            pytest.fail(msg)
        if rec.status != "succeeded":
            msg = f"Plugin execution failed with status {rec.status}"
            pytest.fail(msg)
        summary = rec.meta.get("result")
        if not isinstance(summary, dict):
            msg = "Expected summary metadata dictionary."
            pytest.fail(msg)
        if summary.get("profile_rows", 0) < 0:
            msg = "Profile rows count is invalid."
            pytest.fail(msg)
