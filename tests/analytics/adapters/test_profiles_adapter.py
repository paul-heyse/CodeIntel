"""Test profile adapter classes.

Test the profile-specific adapters for persisting function, file, and module
profile data using real DuckDB instances.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from codeintel.analytics.adapters.profiles import (
    FileProfileAdapter,
    FunctionProfileAdapter,
    ModuleProfileAdapter,
)
from codeintel.config.primitives import SnapshotRef
from codeintel.storage.gateway import StorageGateway
from tests._helpers.assertions import expect_equal
from tests._helpers.contracts import count_rows

# =============================================================================
# Constants
# =============================================================================

DEMO_REPO = "demo/repo"
DEMO_COMMIT = "abc123def456"
EXPECTED_COUNT_0 = 0
EXPECTED_COUNT_1 = 1
EXPECTED_COUNT_2 = 2
TEST_GOID_12345 = Decimal(12345)
TEST_GOID_67890 = Decimal(67890)
TEST_LOC_50 = 50
TEST_COMPLEXITY_5 = 5
TEST_NODE_COUNT_100 = 100
TEST_FUNCTION_COUNT_10 = 10
TEST_FILE_COUNT_5 = 5


# =============================================================================
# Test Data Factories
# =============================================================================


def _make_function_profile_row(
    goid: Decimal = TEST_GOID_12345,
    qualname: str = "module.function_name",
    rel_path: str = "src/module.py",
) -> dict[str, Any]:
    """
    Create a function profile row for testing.

    Parameters
    ----------
    goid
        Function global object ID.
    qualname
        Fully qualified function name.
    rel_path
        Relative file path.

    Returns
    -------
    dict[str, Any]
        Function profile row dict.
    """
    return {
        "function_goid_h128": goid,
        "urn": f"urn:demo:repo::{qualname}",
        "repo": DEMO_REPO,
        "commit": DEMO_COMMIT,
        "rel_path": rel_path,
        "module": rel_path.replace("/", ".").replace(".py", ""),
        "language": "python",
        "kind": "function",
        "qualname": qualname,
        "start_line": 10,
        "end_line": 30,
        "loc": TEST_LOC_50,
        "logical_loc": 40,
        "cyclomatic_complexity": TEST_COMPLEXITY_5,
        "complexity_bucket": "low",
        "param_count": 3,
        "positional_params": 2,
        "keyword_params": 1,
        "vararg": False,
        "kwarg": False,
        "max_nesting_depth": 2,
        "stmt_count": 15,
        "decorator_count": 1,
        "has_docstring": True,
        "total_params": 3,
        "annotated_params": 3,
        "return_type": "str",
        "param_types": ["int", "str", "bool"],
        "fully_typed": True,
        "partial_typed": False,
        "untyped": False,
        "typedness_bucket": "fully_typed",
        "typedness_source": "annotations",
        "file_typed_ratio": 0.95,
        "static_error_count": 0,
        "has_static_errors": False,
        "executable_lines": 40,
        "covered_lines": 35,
        "coverage_ratio": 0.875,
        "tested": True,
        "untested_reason": None,
        "tests_touching": 5,
        "failing_tests": 0,
        "slow_tests": 0,
        "flaky_tests": 0,
        "last_test_status": "passed",
        "dominant_test_status": "passed",
        "slow_test_threshold_ms": 1000.0,
        "created_in_commit": DEMO_COMMIT,
        "created_at_history": datetime.now(tz=UTC),
        "last_modified_commit": DEMO_COMMIT,
        "last_modified_at": datetime.now(tz=UTC),
        "age_days": 30,
        "commit_count": 10,
        "author_count": 3,
        "lines_added": 100,
        "lines_deleted": 20,
        "churn_score": 0.5,
        "stability_bucket": "stable",
        "call_fan_in": 5,
        "call_fan_out": 3,
        "call_edge_in_count": 5,
        "call_edge_out_count": 3,
        "call_is_leaf": False,
        "call_is_entrypoint": False,
        "call_is_public": True,
        "risk_score": 0.25,
        "risk_level": "low",
        "risk_component_coverage": 0.1,
        "risk_component_complexity": 0.05,
        "risk_component_static": 0.0,
        "risk_component_hotspot": 0.1,
        "is_pure": True,
        "uses_io": False,
        "touches_db": False,
        "uses_time": False,
        "uses_randomness": False,
        "modifies_globals": False,
        "modifies_closure": False,
        "spawns_threads_or_tasks": False,
        "has_transitive_effects": False,
        "purity_confidence": 0.95,
        "param_nullability_json": [],
        "return_nullability": "non_null",
        "has_preconditions": False,
        "has_postconditions": False,
        "has_raises": False,
        "contract_confidence": 0.9,
        "role": "helper",
        "framework": None,
        "role_confidence": 0.85,
        "role_sources_json": ["path_hint"],
        "tags": [],
        "owners": [],
        "doc_short": "Test function.",
        "doc_long": "A test function for unit tests.",
        "doc_params": {"param1": "int", "param2": "str"},
        "doc_returns": {"type": "str", "description": "A string result"},
        "created_at": datetime.now(tz=UTC),
    }


def _make_file_profile_row(
    rel_path: str = "src/services/api.py",
) -> dict[str, Any]:
    """
    Create a file profile row for testing.

    Parameters
    ----------
    rel_path
        Relative file path.

    Returns
    -------
    dict[str, Any]
        File profile row dict.
    """
    return {
        "repo": DEMO_REPO,
        "commit": DEMO_COMMIT,
        "rel_path": rel_path,
        "module": rel_path.replace("/", ".").replace(".py", ""),
        "language": "python",
        "node_count": TEST_NODE_COUNT_100,
        "function_count": TEST_FUNCTION_COUNT_10,
        "class_count": 2,
        "avg_depth": 2.5,
        "max_depth": 5,
        "ast_complexity": 15.0,
        "hotspot_score": 0.75,
        "commit_count": 50,
        "author_count": 5,
        "lines_added": 500,
        "lines_deleted": 200,
        "annotation_ratio": 0.85,
        "untyped_defs": 2,
        "overlay_needed": False,
        "type_error_count": 0,
        "static_error_count": 0,
        "has_static_errors": False,
        "total_functions": TEST_FUNCTION_COUNT_10,
        "public_functions": 8,
        "avg_loc": 25.0,
        "max_loc": 100,
        "avg_cyclomatic_complexity": 3.5,
        "max_cyclomatic_complexity": 8,
        "high_risk_function_count": 1,
        "medium_risk_function_count": 3,
        "max_risk_score": 0.65,
        "file_coverage_ratio": 0.85,
        "tested_function_count": 8,
        "untested_function_count": 2,
        "tests_touching": 15,
        "tags": [],
        "owners": [],
        "created_at": datetime.now(tz=UTC),
    }


def _make_module_profile_row(
    module: str = "services.api",
) -> dict[str, Any]:
    """
    Create a module profile row for testing.

    Parameters
    ----------
    module
        Module name.

    Returns
    -------
    dict[str, Any]
        Module profile row dict.
    """
    return {
        "repo": DEMO_REPO,
        "commit": DEMO_COMMIT,
        "module": module,
        "path": module.replace(".", "/"),
        "language": "python",
        "file_count": TEST_FILE_COUNT_5,
        "total_loc": 500,
        "total_logical_loc": 400,
        "function_count": 25,
        "class_count": 5,
        "avg_file_complexity": 10.0,
        "max_file_complexity": 25.0,
        "high_risk_function_count": 2,
        "medium_risk_function_count": 5,
        "low_risk_function_count": 18,
        "max_risk_score": 0.85,
        "avg_risk_score": 0.35,
        "module_coverage_ratio": 0.75,
        "tested_function_count": 20,
        "untested_function_count": 5,
        "import_fan_in": 10,
        "import_fan_out": 15,
        "cycle_group": None,
        "in_cycle": False,
        "role": "service",
        "role_confidence": 0.9,
        "role_sources_json": ["path_hint", "decorator"],
        "tags": [],
        "owners": [],
        "created_at": datetime.now(tz=UTC),
    }


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def snapshot() -> SnapshotRef:
    """
    Create snapshot reference.

    Returns
    -------
    SnapshotRef
        Snapshot reference for testing.
    """
    return SnapshotRef(
        repo=DEMO_REPO,
        commit=DEMO_COMMIT,
        repo_root=Path("/workspace/demo"),
    )


# =============================================================================
# FunctionProfileAdapter Tests
# =============================================================================


def test_function_profile_adapter_table_name(
    fresh_gateway: StorageGateway,
    snapshot: SnapshotRef,
) -> None:
    """Adapter exposes correct table name."""
    adapter = FunctionProfileAdapter(fresh_gateway, snapshot)
    expect_equal(adapter.table_name, "analytics.function_profile")


def test_function_profile_adapter_load_raises(
    fresh_gateway: StorageGateway,
    snapshot: SnapshotRef,
) -> None:
    """Load raises NotImplementedError (write-only adapter)."""
    adapter = FunctionProfileAdapter(fresh_gateway, snapshot)
    with pytest.raises(NotImplementedError, match="does not support loading"):
        list(adapter.load())


def test_function_profile_adapter_persist_empty(
    fresh_gateway: StorageGateway,
    snapshot: SnapshotRef,
) -> None:
    """Persist empty list returns 0."""
    adapter = FunctionProfileAdapter(fresh_gateway, snapshot)
    count = adapter.persist([])
    expect_equal(count, EXPECTED_COUNT_0)


def test_function_profile_adapter_persist_single_row(
    fresh_gateway: StorageGateway,
    snapshot: SnapshotRef,
) -> None:
    """Persist single row inserts to database."""
    adapter = FunctionProfileAdapter(fresh_gateway, snapshot)
    row = _make_function_profile_row()

    count = adapter.persist([row])
    expect_equal(count, EXPECTED_COUNT_1)

    # Verify row was inserted
    total = count_rows(
        fresh_gateway.con,
        "SELECT COUNT(*) FROM analytics.function_profile WHERE repo = ? AND commit = ?",
        [DEMO_REPO, DEMO_COMMIT],
    )
    expect_equal(total, EXPECTED_COUNT_1)


def test_function_profile_adapter_persist_multiple_rows(
    fresh_gateway: StorageGateway,
    snapshot: SnapshotRef,
) -> None:
    """Persist multiple rows inserts all to database."""
    adapter = FunctionProfileAdapter(fresh_gateway, snapshot)

    rows = [
        _make_function_profile_row(goid=TEST_GOID_12345, qualname="module.func_a"),
        _make_function_profile_row(goid=TEST_GOID_67890, qualname="module.func_b"),
    ]

    count = adapter.persist(rows)
    expect_equal(count, EXPECTED_COUNT_2)

    # Verify rows were inserted
    total = count_rows(
        fresh_gateway.con,
        "SELECT COUNT(*) FROM analytics.function_profile WHERE repo = ? AND commit = ?",
        [DEMO_REPO, DEMO_COMMIT],
    )
    expect_equal(total, EXPECTED_COUNT_2)


# =============================================================================
# FileProfileAdapter Tests
# =============================================================================


def test_file_profile_adapter_table_name(
    fresh_gateway: StorageGateway,
    snapshot: SnapshotRef,
) -> None:
    """Adapter exposes correct table name."""
    adapter = FileProfileAdapter(fresh_gateway, snapshot)
    expect_equal(adapter.table_name, "analytics.file_profile")


def test_file_profile_adapter_load_raises(
    fresh_gateway: StorageGateway,
    snapshot: SnapshotRef,
) -> None:
    """Load raises NotImplementedError (write-only adapter)."""
    adapter = FileProfileAdapter(fresh_gateway, snapshot)
    with pytest.raises(NotImplementedError, match="does not support loading"):
        list(adapter.load())


def test_file_profile_adapter_persist_empty(
    fresh_gateway: StorageGateway,
    snapshot: SnapshotRef,
) -> None:
    """Persist empty list returns 0."""
    adapter = FileProfileAdapter(fresh_gateway, snapshot)
    count = adapter.persist([])
    expect_equal(count, EXPECTED_COUNT_0)


def test_file_profile_adapter_persist_single_row(
    fresh_gateway: StorageGateway,
    snapshot: SnapshotRef,
) -> None:
    """Persist single row inserts to database."""
    adapter = FileProfileAdapter(fresh_gateway, snapshot)
    row = _make_file_profile_row()

    count = adapter.persist([row])
    expect_equal(count, EXPECTED_COUNT_1)

    # Verify row was inserted
    total = count_rows(
        fresh_gateway.con,
        "SELECT COUNT(*) FROM analytics.file_profile WHERE repo = ? AND commit = ?",
        [DEMO_REPO, DEMO_COMMIT],
    )
    expect_equal(total, EXPECTED_COUNT_1)


def test_file_profile_adapter_persist_multiple_rows(
    fresh_gateway: StorageGateway,
    snapshot: SnapshotRef,
) -> None:
    """Persist multiple rows inserts all to database."""
    adapter = FileProfileAdapter(fresh_gateway, snapshot)

    rows = [
        _make_file_profile_row(rel_path="src/api.py"),
        _make_file_profile_row(rel_path="src/db.py"),
    ]

    count = adapter.persist(rows)
    expect_equal(count, EXPECTED_COUNT_2)

    # Verify rows were inserted
    total = count_rows(
        fresh_gateway.con,
        "SELECT COUNT(*) FROM analytics.file_profile WHERE repo = ? AND commit = ?",
        [DEMO_REPO, DEMO_COMMIT],
    )
    expect_equal(total, EXPECTED_COUNT_2)


# =============================================================================
# ModuleProfileAdapter Tests
# =============================================================================


def test_module_profile_adapter_table_name(
    fresh_gateway: StorageGateway,
    snapshot: SnapshotRef,
) -> None:
    """Adapter exposes correct table name."""
    adapter = ModuleProfileAdapter(fresh_gateway, snapshot)
    expect_equal(adapter.table_name, "analytics.module_profile")


def test_module_profile_adapter_load_raises(
    fresh_gateway: StorageGateway,
    snapshot: SnapshotRef,
) -> None:
    """Load raises NotImplementedError (write-only adapter)."""
    adapter = ModuleProfileAdapter(fresh_gateway, snapshot)
    with pytest.raises(NotImplementedError, match="does not support loading"):
        list(adapter.load())


def test_module_profile_adapter_persist_empty(
    fresh_gateway: StorageGateway,
    snapshot: SnapshotRef,
) -> None:
    """Persist empty list returns 0."""
    adapter = ModuleProfileAdapter(fresh_gateway, snapshot)
    count = adapter.persist([])
    expect_equal(count, EXPECTED_COUNT_0)


def test_module_profile_adapter_persist_single_row(
    fresh_gateway: StorageGateway,
    snapshot: SnapshotRef,
) -> None:
    """Persist single row inserts to database."""
    adapter = ModuleProfileAdapter(fresh_gateway, snapshot)
    row = _make_module_profile_row()

    count = adapter.persist([row])
    expect_equal(count, EXPECTED_COUNT_1)

    # Verify row was inserted
    total = count_rows(
        fresh_gateway.con,
        "SELECT COUNT(*) FROM analytics.module_profile WHERE repo = ? AND commit = ?",
        [DEMO_REPO, DEMO_COMMIT],
    )
    expect_equal(total, EXPECTED_COUNT_1)


def test_module_profile_adapter_persist_multiple_rows(
    fresh_gateway: StorageGateway,
    snapshot: SnapshotRef,
) -> None:
    """Persist multiple rows inserts all to database."""
    adapter = ModuleProfileAdapter(fresh_gateway, snapshot)

    rows = [
        _make_module_profile_row(module="services.api"),
        _make_module_profile_row(module="services.db"),
    ]

    count = adapter.persist(rows)
    expect_equal(count, EXPECTED_COUNT_2)

    # Verify rows were inserted
    total = count_rows(
        fresh_gateway.con,
        "SELECT COUNT(*) FROM analytics.module_profile WHERE repo = ? AND commit = ?",
        [DEMO_REPO, DEMO_COMMIT],
    )
    expect_equal(total, EXPECTED_COUNT_2)
