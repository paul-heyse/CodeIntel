"""Unit tests for analytics.profiles aggregations."""

from __future__ import annotations

import pytest

from codeintel.analytics.profiles import (
    SLOW_TEST_THRESHOLD_MS,
    build_file_profile,
    build_function_profile,
    build_module_profile,
)
from codeintel.config.models import ProfilesAnalyticsConfig
from codeintel.storage.gateway import DuckDBConnection
from tests._helpers.fixtures import ProvisionedGateway, seed_profile_data

EPSILON = 1e-6
REL_PATH = "pkg/mod.py"
MODULE = "pkg.mod"


def _assert_function_profile(con: DuckDBConnection) -> None:
    row = con.execute(
        """
        SELECT tests_touching, failing_tests, slow_tests, call_fan_in, call_fan_out,
               risk_component_coverage, doc_short, slow_test_threshold_ms
        FROM analytics.function_profile
        WHERE function_goid_h128 = 1
        """
    ).fetchone()
    if row is None:
        pytest.fail("function_profile row missing")
    if row[0] != 1:
        pytest.fail(f"Expected tests_touching=1, got {row[0]}")
    if row[1] != 1:
        pytest.fail(f"Expected failing_tests=1, got {row[1]}")
    if row[2] != 1:
        pytest.fail(f"Expected slow_tests=1, got {row[2]}")
    if row[3] != 1 or row[4] != 1:
        pytest.fail(f"Unexpected call fan-in/out: {row[3]}, {row[4]}")
    if abs(row[5] - 0.2) > EPSILON:
        pytest.fail(f"Unexpected coverage component {row[5]}")
    if row[6] != "Short doc":
        pytest.fail(f"Doc summary mismatch: {row[6]}")
    if row[7] != SLOW_TEST_THRESHOLD_MS:
        pytest.fail(f"Slow threshold mismatch: {row[7]}")


def _assert_file_profile(con: DuckDBConnection) -> None:
    row = con.execute(
        """
        SELECT file_coverage_ratio, high_risk_function_count, module
        FROM analytics.file_profile
        WHERE rel_path = ?
        """,
        [REL_PATH],
    ).fetchone()
    if row is None:
        pytest.fail("file_profile row missing")
    if abs(row[0] - 0.5) > EPSILON:
        pytest.fail(f"Unexpected file coverage ratio {row[0]}")
    if row[1] != 1:
        pytest.fail(f"Expected 1 high-risk function, got {row[1]}")
    if row[2] != MODULE:
        pytest.fail(f"Module mismatch: {row[2]}")


def _assert_module_profile(con: DuckDBConnection) -> None:
    row = con.execute(
        """
        SELECT module_coverage_ratio, import_fan_in, import_fan_out, in_cycle
        FROM analytics.module_profile
        WHERE module = ?
        """,
        [MODULE],
    ).fetchone()
    if row is None:
        pytest.fail("module_profile row missing")
    if abs(row[0] - 1.0) > EPSILON:
        pytest.fail(f"Unexpected module coverage ratio {row[0]}")
    if row[1] != 1 or row[2] != 1:
        pytest.fail(f"Import fan-in/out mismatch: {row[1]}, {row[2]}")
    if row[3] is not True:
        pytest.fail("Expected module to be marked as in_cycle")


def test_profile_builders_aggregate_expected_fields(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Ensure profile builders compose metrics, tests, coverage, and graph data."""
    gateway = provisioned_repo.gateway
    con = gateway.con
    seed_profile_data(
        gateway,
        repo=provisioned_repo.repo,
        commit=provisioned_repo.commit,
        rel_path=REL_PATH,
        module=MODULE,
    )
    cfg = ProfilesAnalyticsConfig(repo=provisioned_repo.repo, commit=provisioned_repo.commit)
    build_function_profile(gateway, cfg)
    build_file_profile(gateway, cfg)
    build_module_profile(gateway, cfg)
    _assert_function_profile(con)
    _assert_file_profile(con)
    _assert_module_profile(con)
