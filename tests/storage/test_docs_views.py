"""Tests for docs view functionality.

This module consolidates tests for:
- Docs view performance indexes
- Docs views as first-class datasets
- Docs view scoping (repo/commit isolation)
- Subsystem docs view schemas and cache behavior

Consolidated from:
- test_docs_view_indexes.py
- test_docs_views_as_datasets.py
- test_views_scoping.py
- test_subsystem_docs_views.py
"""

from __future__ import annotations

from pathlib import Path

import pytest

from codeintel.storage.datasets import load_dataset_registry
from codeintel.storage.gateway import StorageGateway
from codeintel.storage.metadata import bootstrap_metadata_datasets
from codeintel.storage.repositories.datasets import DatasetReadRepository
from codeintel.storage.views import DERIVED_DOCS_VIEWS
from tests._helpers import docs_views_ready_gateway, seed_call_graph_scoping
from tests._helpers.docs_views import list_indexes, seed_subsystem

# Constants
EXPECTED_MODULE_COUNT_42 = 42
EXPECTED_FUNCTION_COUNT_4 = 4
EXPECTED_TEST_COUNT_99 = 99
EXPECTED_FUNCTIONS_COVERED_50 = 50


def _require(*, condition: bool, message: str) -> None:
    """Fail test if condition is not met."""
    if not condition:
        pytest.fail(message)


# =============================================================================
# Performance Index Tests
# =============================================================================


def test_test_profile_has_primary_subsystem_index(docs_views_gateway: StorageGateway) -> None:
    """analytics.test_profile should be indexed for primary_subsystem_id scans."""
    index_names = list_indexes(docs_views_gateway.con, schema="analytics", table="test_profile")
    expected = "idx_analytics_test_profile_primary_subsystem"
    if expected not in index_names:
        pytest.fail(f"Missing index {expected} on analytics.test_profile")


def test_subsystems_has_repo_commit_index(docs_views_gateway: StorageGateway) -> None:
    """analytics.subsystems should be indexed for repo/commit/subsystem lookups."""
    index_names = list_indexes(docs_views_gateway.con, schema="analytics", table="subsystems")
    expected = "idx_analytics_subsystems_repo_commit_id"
    if expected not in index_names:
        pytest.fail(f"Missing index {expected} on analytics.subsystems")


# =============================================================================
# Docs Views as Datasets Tests
# =============================================================================


def test_docs_views_registered_in_metadata(docs_views_gateway: StorageGateway) -> None:
    """Derived docs views should be registered as views in metadata.datasets."""
    bootstrap_metadata_datasets(docs_views_gateway.con)
    rows = docs_views_gateway.con.execute(
        "SELECT table_key, is_view FROM metadata.datasets WHERE table_key LIKE 'docs.%'"
    ).fetchall()
    table_keys = {row[0] for row in rows}
    missing = set(DERIVED_DOCS_VIEWS) - table_keys
    if missing:
        pytest.fail(f"Missing docs views in metadata.datasets: {sorted(missing)}")
    if not all(row[1] for row in rows):
        pytest.fail("Expected all docs entries in metadata.datasets to be marked as views")


def test_docs_view_readable_via_dataset_rows(docs_views_gateway: StorageGateway) -> None:
    """Docs views remain readable through metadata.dataset_rows slices."""
    bootstrap_metadata_datasets(docs_views_gateway.con)
    repo = DatasetReadRepository(
        gateway=docs_views_gateway, repo="demo/repo", commit="deadbeef"
    )
    rows = repo.read_dataset_rows("docs.v_function_summary", limit=5, offset=0)
    if not isinstance(rows, list):
        pytest.fail("Expected list from dataset_rows")


def test_docs_views_expose_capabilities(docs_views_gateway: StorageGateway) -> None:
    """Docs views and caches surface docs/read-only capability flags."""
    bootstrap_metadata_datasets(docs_views_gateway.con)
    registry = load_dataset_registry(docs_views_gateway.con)
    profile_view = registry.by_name["v_subsystem_profile"]
    profile_caps = profile_view.capabilities()
    if not profile_caps["docs_view"]:
        pytest.fail("Expected docs views to be marked with docs_view capability")
    if not profile_caps["read_only"]:
        pytest.fail("Expected docs views to be flagged read_only")
    cache_ds = registry.by_name["subsystem_profile_cache"]
    cache_caps = cache_ds.capabilities()
    if cache_caps["docs_view"]:
        pytest.fail("Cache tables should not be marked as docs views")
    if not cache_caps["can_validate"]:
        pytest.fail("Cache tables should be validation-capable via JSON Schema")
    if cache_caps["read_only"]:
        pytest.fail("Cache tables should allow writes for refreshes")


# =============================================================================
# View Scoping Tests
# =============================================================================


def test_call_graph_view_scopes_edges_to_repo_commit(tmp_path: Path) -> None:
    """Edges from other snapshots should not join into v_call_graph_enriched."""
    ctx = docs_views_ready_gateway(tmp_path / "docs_scoping", repo="r1", commit="c1")
    seed_call_graph_scoping(gateway=ctx.gateway, now_iso="2024-01-01T00:00:00Z")
    con = ctx.gateway.con

    rows_r1 = con.execute(
        "SELECT DISTINCT caller_repo FROM docs.v_call_graph_enriched WHERE caller_repo = 'r1'"
    ).fetchall()
    rows_r2 = con.execute(
        "SELECT DISTINCT caller_repo FROM docs.v_call_graph_enriched WHERE caller_repo = 'r2'"
    ).fetchall()

    if rows_r1 != [("r1",)]:
        pytest.fail(f"Unexpected caller_repo rows for r1: {rows_r1}")
    if rows_r2 != [("r2",)]:
        pytest.fail(f"Unexpected caller_repo rows for r2: {rows_r2}")
    ctx.close()


# =============================================================================
# Subsystem Docs View Schema Tests
# =============================================================================


def test_subsystem_profile_columns(docs_views_gateway: StorageGateway) -> None:
    """Subsystem profile view exposes expected columns for typed contracts."""
    bootstrap_metadata_datasets(docs_views_gateway.con)
    rel_df = (
        docs_views_gateway.con.execute("SELECT * FROM docs.v_subsystem_profile LIMIT 0").fetchdf()
    )
    cols = [c.lower() for c in rel_df.columns]
    expected = {
        "repo",
        "commit",
        "subsystem_id",
        "name",
        "description",
        "module_count",
        "modules_json",
        "entrypoints_json",
        "internal_edge_count",
        "external_edge_count",
        "fan_in",
        "fan_out",
        "function_count",
        "avg_risk_score",
        "max_risk_score",
        "high_risk_function_count",
        "risk_level",
        "import_in_degree",
        "import_out_degree",
        "import_pagerank",
        "import_betweenness",
        "import_closeness",
        "import_layer",
        "created_at",
    }
    missing = expected - set(cols)
    _require(
        condition=not missing,
        message=f"Missing columns in v_subsystem_profile: {sorted(missing)}",
    )


def test_subsystem_coverage_columns(docs_views_gateway: StorageGateway) -> None:
    """Subsystem coverage view exposes expected columns for typed contracts."""
    bootstrap_metadata_datasets(docs_views_gateway.con)
    rel_df = (
        docs_views_gateway.con.execute("SELECT * FROM docs.v_subsystem_coverage LIMIT 0").fetchdf()
    )
    cols = [c.lower() for c in rel_df.columns]
    expected = {
        "repo",
        "commit",
        "subsystem_id",
        "name",
        "description",
        "module_count",
        "function_count",
        "risk_level",
        "avg_risk_score",
        "max_risk_score",
        "test_count",
        "passed_test_count",
        "failed_test_count",
        "skipped_test_count",
        "xfail_test_count",
        "flaky_test_count",
        "total_functions_covered",
        "avg_functions_covered",
        "max_functions_covered",
        "min_functions_covered",
        "function_coverage_ratio",
        "created_at",
    }
    missing = expected - set(cols)
    _require(
        condition=not missing,
        message=f"Missing columns in v_subsystem_coverage: {sorted(missing)}",
    )


# =============================================================================
# Subsystem Cache Behavior Tests
# =============================================================================


def test_subsystem_profile_view_prefers_cache(docs_views_gateway: StorageGateway) -> None:
    """Cached subsystem profile rows should override computed values."""
    bootstrap_metadata_datasets(docs_views_gateway.con)
    seed_subsystem(docs_views_gateway.con, overrides={"module_count": 1, "function_count": 2})
    docs_views_gateway.con.execute(
        """
        INSERT INTO analytics.subsystem_profile_cache (
            repo, commit, subsystem_id, name, description, module_count,
            modules_json, entrypoints_json, internal_edge_count,
            external_edge_count, fan_in, fan_out, function_count,
            avg_risk_score, max_risk_score, high_risk_function_count,
            risk_level, import_in_degree, import_out_degree, import_pagerank,
            import_betweenness, import_closeness, import_layer, created_at
        )
        VALUES (
            'demo/repo', 'deadbeef', 'subsysdemo', 'Cached Name', 'cached',
            ?, '[]', '[]', 1, 1, 2, 3, ?, 0.5, 0.9, 7, 'medium',
            0.1, 0.2, 0.3, 0.4, 0.5, 2, CURRENT_TIMESTAMP
        )
        """,
        [EXPECTED_MODULE_COUNT_42, EXPECTED_FUNCTION_COUNT_4],
    )
    row = docs_views_gateway.con.execute(
        """
        SELECT name, module_count, function_count, risk_level
        FROM docs.v_subsystem_profile
        WHERE subsystem_id = 'subsysdemo'
        """
    ).fetchone()
    if row is None:
        pytest.fail("No subsystem profile row returned")
        return
    name, module_count, function_count, risk_level = row
    _require(condition=name == "Cached Name", message="Expected cached name to be used")
    _require(
        condition=module_count == EXPECTED_MODULE_COUNT_42,
        message="Expected cached module_count to be used",
    )
    _require(
        condition=function_count == EXPECTED_FUNCTION_COUNT_4,
        message="Expected cached function_count to be used",
    )
    _require(condition=risk_level == "medium", message="Expected cached risk_level to be used")


def test_subsystem_coverage_view_prefers_cache(docs_views_gateway: StorageGateway) -> None:
    """Cached subsystem coverage rows should override computed values."""
    bootstrap_metadata_datasets(docs_views_gateway.con)
    seed_subsystem(docs_views_gateway.con, overrides={"module_count": 1, "function_count": 2})
    docs_views_gateway.con.execute(
        """
        INSERT INTO analytics.subsystem_coverage_cache (
            repo, commit, subsystem_id, name, description, module_count,
            function_count, risk_level, avg_risk_score, max_risk_score,
            test_count, passed_test_count, failed_test_count,
            skipped_test_count, xfail_test_count, flaky_test_count,
            total_functions_covered, avg_functions_covered,
            max_functions_covered, min_functions_covered,
            function_coverage_ratio, created_at
        )
        VALUES (
            'demo/repo', 'deadbeef', 'subsysdemo', 'Cached Name', 'cached',
            3, 10, 'high', 0.7, 0.9, ?, 90, 9, 0, 0, 5,
            ?, 5.0, 10.0, 1.0, 0.5, CURRENT_TIMESTAMP
        )
        """,
        [EXPECTED_TEST_COUNT_99, EXPECTED_FUNCTIONS_COVERED_50],
    )
    row = docs_views_gateway.con.execute(
        """
        SELECT test_count, total_functions_covered, risk_level
        FROM docs.v_subsystem_coverage
        WHERE subsystem_id = 'subsysdemo'
        """
    ).fetchone()
    if row is None:
        pytest.fail("No subsystem coverage row returned")
        return
    test_count, total_functions_covered, risk_level = row
    _require(
        condition=test_count == EXPECTED_TEST_COUNT_99,
        message="Expected cached test_count to be used",
    )
    _require(
        condition=total_functions_covered == EXPECTED_FUNCTIONS_COVERED_50,
        message="Expected cached total_functions_covered to be used",
    )
    _require(condition=risk_level == "high", message="Expected cached risk_level to be used")
