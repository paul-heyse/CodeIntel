"""Smoke tests for analytics.tests_profiles wrappers."""

from __future__ import annotations

from pathlib import Path

import duckdb

from codeintel.analytics.tests_profiles.behavioral_tags import infer_behavior_tags
from codeintel.analytics.tests_profiles.coverage_inputs import (
    aggregate_test_coverage_by_function,
    aggregate_test_coverage_by_subsystem,
    load_test_graph_metrics,
)
from codeintel.analytics.tests_profiles.importance import (
    compute_flakiness_score,
    compute_importance_score,
)
from codeintel.analytics.tests_profiles.types import ImportanceInputs, IoFlags, TestAstInfo
from codeintel.config.primitives import SnapshotRef
from codeintel.config.steps_analytics import BehavioralCoverageStepConfig, TestProfileStepConfig


def _empty_conn() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(database=":memory:")
    con.execute("CREATE SCHEMA analytics")
    con.execute("CREATE SCHEMA core")
    con.execute(
        """
        CREATE TABLE analytics.test_coverage_edges (
            test_id VARCHAR,
            function_goid_h128 DECIMAL(38,0),
            covered_lines INTEGER,
            executable_lines INTEGER,
            repo VARCHAR,
            commit VARCHAR,
            rel_path VARCHAR,
            qualname VARCHAR
        )
        """
    )
    con.execute(
        """
        CREATE TABLE analytics.test_graph_metrics_tests (
            test_id VARCHAR,
            degree INTEGER,
            weighted_degree DOUBLE,
            proj_degree INTEGER,
            proj_weight DOUBLE,
            proj_clustering DOUBLE,
            proj_betweenness DOUBLE,
            repo VARCHAR,
            commit VARCHAR
        )
        """
    )
    con.execute(
        """
        CREATE TABLE analytics.subsystem_modules (
            module VARCHAR,
            subsystem_id VARCHAR,
            repo VARCHAR,
            commit VARCHAR
        )
        """
    )
    con.execute(
        """
        CREATE TABLE analytics.subsystems (
            subsystem_id VARCHAR,
            name VARCHAR,
            max_risk_score DOUBLE,
            repo VARCHAR,
            commit VARCHAR
        )
        """
    )
    con.execute(
        """
        CREATE TABLE core.goids (
            goid_h128 DECIMAL(38,0),
            urn VARCHAR,
            repo VARCHAR,
            commit VARCHAR,
            rel_path VARCHAR,
            qualname VARCHAR
        )
        """
    )
    con.execute(
        """
        CREATE TABLE core.modules (
            module VARCHAR,
            path VARCHAR,
            repo VARCHAR,
            commit VARCHAR
        )
        """
    )
    return con


def _configs(tmp_path: Path) -> tuple[TestProfileStepConfig, BehavioralCoverageStepConfig]:
    snapshot = SnapshotRef(repo="r", commit="c", repo_root=tmp_path)
    return TestProfileStepConfig(snapshot=snapshot), BehavioralCoverageStepConfig(snapshot=snapshot)


def test_coverage_wrappers_empty(tmp_path: Path) -> None:
    """
    Ensure coverage aggregation wrappers handle empty tables.

    Raises
    ------
    AssertionError
        If any aggregation returns a non-empty result.
    """
    con = _empty_conn()
    test_cfg, beh_cfg = _configs(tmp_path)

    if aggregate_test_coverage_by_function(con, test_cfg, loader=lambda *_: {}) != {}:
        message = "Expected empty function coverage aggregation."
        raise AssertionError(message)
    if aggregate_test_coverage_by_subsystem(con, beh_cfg, loader=lambda *_: {}) != {}:
        message = "Expected empty subsystem coverage aggregation."
        raise AssertionError(message)
    if load_test_graph_metrics(con, test_cfg, loader=lambda *_: {}) != {}:
        message = "Expected empty test graph metrics aggregation."
        raise AssertionError(message)


def test_importance_and_flakiness_scoring() -> None:
    """
    Validate flakiness and importance scoring produce bounded values.

    Raises
    ------
    AssertionError
        If scores fall outside expected ranges.
    """
    io_flags = IoFlags(
        uses_network=True, uses_db=False, uses_filesystem=False, uses_subprocess=False
    )
    flakiness = compute_flakiness_score(
        status="xfail",
        markers=["slow", "network"],
        duration_ms=3000.0,
        io_flags=io_flags,
        slow_test_threshold_ms=2000.0,
    )
    min_expected_flakiness = 0.4
    if not (min_expected_flakiness <= flakiness <= 1.0):
        message = "Flakiness score out of expected range."
        raise AssertionError(message)

    inputs = ImportanceInputs(
        functions_covered_count=2,
        weighted_degree=1.0,
        max_function_count=4,
        max_weighted_degree=2.0,
        subsystem_risk=0.5,
        max_subsystem_risk=1.0,
    )
    importance = compute_importance_score(inputs)
    if importance is None or not (0.0 <= importance <= 1.0):
        message = "Importance score out of expected range."
        raise AssertionError(message)


def test_infer_behavior_tags_basic() -> None:
    """
    Ensure behavior tag inference captures core markers.

    Raises
    ------
    AssertionError
        If expected tags are missing.
    """
    ast_info = TestAstInfo(
        assert_count=0,
        raise_count=0,
        uses_pytest_raises=True,
        uses_concurrency_lib=False,
        has_boundary_asserts=False,
        uses_fixtures=False,
        io_flags=IoFlags(),
    )
    tags = infer_behavior_tags(
        name="test_network_error_path",
        markers=["network", "slow"],
        io_flags=IoFlags(
            uses_network=True, uses_db=False, uses_filesystem=False, uses_subprocess=False
        ),
        ast_info=ast_info,
    )
    if "network_interaction" not in tags or "error_paths" not in tags:
        message = f"Unexpected tags: {tags}"
        raise AssertionError(message)
