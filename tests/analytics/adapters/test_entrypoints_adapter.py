"""Test entrypoint adapter classes.

Test the entrypoint-specific adapters for persisting entrypoint detection
and test mapping data using real DuckDB instances.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from codeintel.analytics.adapters.entrypoints import (
    EntrypointsAdapter,
    EntrypointTestsAdapter,
)
from codeintel.config.primitives import SnapshotRef
from codeintel.storage.gateway import StorageGateway
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
TEST_DURATION_MS_150 = 150.0
TEST_COVERAGE_RATIO_0_85 = 0.85


# =============================================================================
# Test Data Factories
# =============================================================================


def _make_entrypoint_row(
    entrypoint_id: str = "ep_api_users_get",
    kind: str = "http_endpoint",
    goid: Decimal = TEST_GOID_12345,
    handler_qualname: str = "api.users.get_users",
) -> dict[str, Any]:
    """
    Create an entrypoint row for testing.

    Parameters
    ----------
    entrypoint_id
        Unique entrypoint identifier.
    kind
        Entrypoint kind (http_endpoint, cli_command, etc.).
    goid
        Handler function global object ID.
    handler_qualname
        Fully qualified handler function name.

    Returns
    -------
    dict[str, Any]
        Entrypoint row dict.
    """
    return {
        "repo": DEMO_REPO,
        "commit": DEMO_COMMIT,
        "entrypoint_id": entrypoint_id,
        "kind": kind,
        "framework": "fastapi",
        "handler_goid_h128": goid,
        "handler_urn": f"urn:demo:repo::{handler_qualname}",
        "handler_rel_path": "src/api/users.py",
        "handler_module": "api.users",
        "handler_qualname": handler_qualname,
        "http_method": "GET",
        "route_path": "/api/users",
        "status_codes": [200, 401, 404],
        "auth_required": True,
        "command_name": None,
        "arguments_schema": None,
        "schedule": None,
        "trigger": None,
        "extra": {},
        "subsystem_id": "users",
        "subsystem_name": "User Management",
        "tags": ["api", "users"],
        "owners": ["team-backend"],
        "tests_touching": 5,
        "failing_tests": 0,
        "slow_tests": 1,
        "flaky_tests": 0,
        "entrypoint_coverage_ratio": TEST_COVERAGE_RATIO_0_85,
        "last_test_status": "passed",
        "created_at": datetime.now(tz=UTC),
    }


def _make_cli_entrypoint_row(
    entrypoint_id: str = "ep_cli_migrate",
    goid: Decimal = TEST_GOID_67890,
) -> dict[str, Any]:
    """
    Create a CLI entrypoint row for testing.

    Parameters
    ----------
    entrypoint_id
        Unique entrypoint identifier.
    goid
        Handler function global object ID.

    Returns
    -------
    dict[str, Any]
        CLI entrypoint row dict.
    """
    return {
        "repo": DEMO_REPO,
        "commit": DEMO_COMMIT,
        "entrypoint_id": entrypoint_id,
        "kind": "cli_command",
        "framework": "click",
        "handler_goid_h128": goid,
        "handler_urn": "urn:demo:repo::cli.migrate.run_migration",
        "handler_rel_path": "src/cli/migrate.py",
        "handler_module": "cli.migrate",
        "handler_qualname": "cli.migrate.run_migration",
        "http_method": None,
        "route_path": None,
        "status_codes": None,
        "auth_required": False,
        "command_name": "migrate",
        "arguments_schema": {"version": "str", "dry_run": "bool"},
        "schedule": None,
        "trigger": None,
        "extra": {"group": "database"},
        "subsystem_id": "db",
        "subsystem_name": "Database",
        "tags": ["cli", "database"],
        "owners": ["team-platform"],
        "tests_touching": 3,
        "failing_tests": 0,
        "slow_tests": 0,
        "flaky_tests": 0,
        "entrypoint_coverage_ratio": 0.9,
        "last_test_status": "passed",
        "created_at": datetime.now(tz=UTC),
    }


def _make_entrypoint_test_row(
    entrypoint_id: str = "ep_api_users_get",
    test_id: str = "test_get_users_success",
    test_goid: Decimal = TEST_GOID_12345,
) -> dict[str, Any]:
    """
    Create an entrypoint test mapping row for testing.

    Parameters
    ----------
    entrypoint_id
        Entrypoint identifier being tested.
    test_id
        Test identifier.
    test_goid
        Test function global object ID.

    Returns
    -------
    dict[str, Any]
        Entrypoint test row dict.
    """
    return {
        "repo": DEMO_REPO,
        "commit": DEMO_COMMIT,
        "entrypoint_id": entrypoint_id,
        "test_id": test_id,
        "test_goid_h128": test_goid,
        "coverage_ratio": TEST_COVERAGE_RATIO_0_85,
        "status": "passed",
        "duration_ms": TEST_DURATION_MS_150,
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
# EntrypointsAdapter Tests
# =============================================================================


def test_entrypoints_adapter_table_name(
    fresh_gateway: StorageGateway,
    snapshot: SnapshotRef,
) -> None:
    """Adapter exposes correct table name."""
    adapter = EntrypointsAdapter(fresh_gateway, snapshot)
    assert adapter.table_name == "analytics.entrypoints"


def test_entrypoints_adapter_load_raises(
    fresh_gateway: StorageGateway,
    snapshot: SnapshotRef,
) -> None:
    """Load raises NotImplementedError (write-only adapter)."""
    adapter = EntrypointsAdapter(fresh_gateway, snapshot)
    with pytest.raises(NotImplementedError, match="does not support loading"):
        list(adapter.load())


def test_entrypoints_adapter_persist_empty(
    fresh_gateway: StorageGateway,
    snapshot: SnapshotRef,
) -> None:
    """Persist empty list returns 0."""
    adapter = EntrypointsAdapter(fresh_gateway, snapshot)
    count = adapter.persist([])
    assert count == EXPECTED_COUNT_0


def test_entrypoints_adapter_persist_http_endpoint(
    fresh_gateway: StorageGateway,
    snapshot: SnapshotRef,
) -> None:
    """Persist HTTP endpoint entrypoint."""
    adapter = EntrypointsAdapter(fresh_gateway, snapshot)
    row = _make_entrypoint_row()

    count = adapter.persist([row])
    assert count == EXPECTED_COUNT_1

    # Verify row was inserted
    total = count_rows(
        fresh_gateway.con,
        "SELECT COUNT(*) FROM analytics.entrypoints WHERE repo = ? AND commit = ?",
        [DEMO_REPO, DEMO_COMMIT],
    )
    assert total == EXPECTED_COUNT_1


def test_entrypoints_adapter_persist_cli_command(
    fresh_gateway: StorageGateway,
    snapshot: SnapshotRef,
) -> None:
    """Persist CLI command entrypoint."""
    adapter = EntrypointsAdapter(fresh_gateway, snapshot)
    row = _make_cli_entrypoint_row()

    count = adapter.persist([row])
    assert count == EXPECTED_COUNT_1

    # Verify CLI-specific fields
    result = fresh_gateway.con.execute(
        """
        SELECT kind, command_name, http_method
        FROM analytics.entrypoints
        WHERE repo = ? AND commit = ?
        """,
        [DEMO_REPO, DEMO_COMMIT],
    ).fetchone()
    assert result is not None
    assert result[0] == "cli_command"
    assert result[1] == "migrate"
    assert result[2] is None  # http_method is null for CLI


def test_entrypoints_adapter_persist_multiple(
    fresh_gateway: StorageGateway,
    snapshot: SnapshotRef,
) -> None:
    """Persist multiple entrypoints."""
    adapter = EntrypointsAdapter(fresh_gateway, snapshot)

    rows = [
        _make_entrypoint_row(entrypoint_id="ep_1", handler_qualname="api.get"),
        _make_cli_entrypoint_row(entrypoint_id="ep_2"),
    ]

    count = adapter.persist(rows)
    assert count == EXPECTED_COUNT_2


# =============================================================================
# EntrypointTestsAdapter Tests
# =============================================================================


def test_entrypoint_tests_adapter_table_name(
    fresh_gateway: StorageGateway,
    snapshot: SnapshotRef,
) -> None:
    """Adapter exposes correct table name."""
    adapter = EntrypointTestsAdapter(fresh_gateway, snapshot)
    assert adapter.table_name == "analytics.entrypoint_tests"


def test_entrypoint_tests_adapter_load_raises(
    fresh_gateway: StorageGateway,
    snapshot: SnapshotRef,
) -> None:
    """Load raises NotImplementedError (write-only adapter)."""
    adapter = EntrypointTestsAdapter(fresh_gateway, snapshot)
    with pytest.raises(NotImplementedError, match="does not support loading"):
        list(adapter.load())


def test_entrypoint_tests_adapter_persist_empty(
    fresh_gateway: StorageGateway,
    snapshot: SnapshotRef,
) -> None:
    """Persist empty list returns 0."""
    adapter = EntrypointTestsAdapter(fresh_gateway, snapshot)
    count = adapter.persist([])
    assert count == EXPECTED_COUNT_0


def test_entrypoint_tests_adapter_persist_single(
    fresh_gateway: StorageGateway,
    snapshot: SnapshotRef,
) -> None:
    """Persist single entrypoint-test mapping."""
    adapter = EntrypointTestsAdapter(fresh_gateway, snapshot)
    row = _make_entrypoint_test_row()

    count = adapter.persist([row])
    assert count == EXPECTED_COUNT_1

    # Verify row was inserted
    total = count_rows(
        fresh_gateway.con,
        "SELECT COUNT(*) FROM analytics.entrypoint_tests WHERE repo = ? AND commit = ?",
        [DEMO_REPO, DEMO_COMMIT],
    )
    assert total == EXPECTED_COUNT_1


def test_entrypoint_tests_adapter_persist_multiple(
    fresh_gateway: StorageGateway,
    snapshot: SnapshotRef,
) -> None:
    """Persist multiple entrypoint-test mappings."""
    adapter = EntrypointTestsAdapter(fresh_gateway, snapshot)

    rows = [
        _make_entrypoint_test_row(test_id="test_1", test_goid=Decimal(1001)),
        _make_entrypoint_test_row(test_id="test_2", test_goid=Decimal(1002)),
    ]

    count = adapter.persist(rows)
    assert count == EXPECTED_COUNT_2


def test_entrypoint_tests_adapter_persist_verifies_data(
    fresh_gateway: StorageGateway,
    snapshot: SnapshotRef,
) -> None:
    """Persisted data can be retrieved and verified."""
    adapter = EntrypointTestsAdapter(fresh_gateway, snapshot)
    row = _make_entrypoint_test_row(
        entrypoint_id="ep_verify",
        test_id="test_verify",
    )
    adapter.persist([row])

    # Query and verify
    result = fresh_gateway.con.execute(
        """
        SELECT entrypoint_id, test_id, status, coverage_ratio
        FROM analytics.entrypoint_tests
        WHERE repo = ? AND commit = ?
        """,
        [DEMO_REPO, DEMO_COMMIT],
    ).fetchone()

    assert result is not None
    assert result[0] == "ep_verify"
    assert result[1] == "test_verify"
    assert result[2] == "passed"
    assert float(result[3]) == pytest.approx(TEST_COVERAGE_RATIO_0_85)
