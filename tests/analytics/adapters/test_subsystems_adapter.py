"""Test subsystem adapter classes.

Test the subsystem-specific adapters for persisting subsystem classification
and module mapping data using real DuckDB instances.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from codeintel.analytics.adapters.subsystems import (
    SubsystemModulesAdapter,
    SubsystemsAdapter,
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
EXPECTED_COUNT_3 = 3
MODULE_COUNT_5 = 5
FUNCTION_COUNT_25 = 25
HIGH_RISK_COUNT_2 = 2
FAN_IN_10 = 10
FAN_OUT_15 = 15
AVG_RISK_0_35 = 0.35
MAX_RISK_0_85 = 0.85


# =============================================================================
# Test Data Factories
# =============================================================================


def _make_subsystem_row(
    subsystem_id: str = "auth",
    name: str = "Authentication",
) -> dict[str, Any]:
    """
    Create a subsystem row for testing.

    Parameters
    ----------
    subsystem_id
        Unique subsystem identifier.
    name
        Human-readable subsystem name.

    Returns
    -------
    dict[str, Any]
        Subsystem row dict.
    """
    return {
        "repo": DEMO_REPO,
        "commit": DEMO_COMMIT,
        "subsystem_id": subsystem_id,
        "name": name,
        "description": f"The {name} subsystem handles related functionality.",
        "module_count": MODULE_COUNT_5,
        "modules_json": ["auth.core", "auth.providers", "auth.tokens"],
        "entrypoints_json": ["POST /login", "POST /logout", "GET /me"],
        "internal_edge_count": 20,
        "external_edge_count": 8,
        "fan_in": FAN_IN_10,
        "fan_out": FAN_OUT_15,
        "function_count": FUNCTION_COUNT_25,
        "avg_risk_score": AVG_RISK_0_35,
        "max_risk_score": MAX_RISK_0_85,
        "high_risk_function_count": HIGH_RISK_COUNT_2,
        "risk_level": "moderate",
        "created_at": datetime.now(tz=UTC),
    }


def _make_subsystem_module_row(
    subsystem_id: str = "auth",
    module: str = "auth.core",
    role: str = "core",
) -> dict[str, Any]:
    """
    Create a subsystem module mapping row for testing.

    Parameters
    ----------
    subsystem_id
        Subsystem identifier.
    module
        Module name.
    role
        Module's role within the subsystem.

    Returns
    -------
    dict[str, Any]
        Subsystem module row dict.
    """
    return {
        "repo": DEMO_REPO,
        "commit": DEMO_COMMIT,
        "subsystem_id": subsystem_id,
        "module": module,
        "role": role,
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
# SubsystemsAdapter Tests
# =============================================================================


def test_subsystems_adapter_table_name(
    fresh_gateway: StorageGateway,
    snapshot: SnapshotRef,
) -> None:
    """Adapter exposes correct table name."""
    adapter = SubsystemsAdapter(fresh_gateway, snapshot)
    assert adapter.table_name == "analytics.subsystems"


def test_subsystems_adapter_load_raises(
    fresh_gateway: StorageGateway,
    snapshot: SnapshotRef,
) -> None:
    """Load raises NotImplementedError (write-only adapter)."""
    adapter = SubsystemsAdapter(fresh_gateway, snapshot)
    with pytest.raises(NotImplementedError, match="does not support loading"):
        list(adapter.load())


def test_subsystems_adapter_persist_empty(
    fresh_gateway: StorageGateway,
    snapshot: SnapshotRef,
) -> None:
    """Persist empty list returns 0."""
    adapter = SubsystemsAdapter(fresh_gateway, snapshot)
    count = adapter.persist([])
    assert count == EXPECTED_COUNT_0


def test_subsystems_adapter_persist_single(
    fresh_gateway: StorageGateway,
    snapshot: SnapshotRef,
) -> None:
    """Persist single subsystem."""
    adapter = SubsystemsAdapter(fresh_gateway, snapshot)
    row = _make_subsystem_row()

    count = adapter.persist([row])
    assert count == EXPECTED_COUNT_1

    # Verify row was inserted
    total = count_rows(
        fresh_gateway.con,
        "SELECT COUNT(*) FROM analytics.subsystems WHERE repo = ? AND commit = ?",
        [DEMO_REPO, DEMO_COMMIT],
    )
    assert total == EXPECTED_COUNT_1


def test_subsystems_adapter_persist_multiple(
    fresh_gateway: StorageGateway,
    snapshot: SnapshotRef,
) -> None:
    """Persist multiple subsystems."""
    adapter = SubsystemsAdapter(fresh_gateway, snapshot)

    rows = [
        _make_subsystem_row(subsystem_id="auth", name="Authentication"),
        _make_subsystem_row(subsystem_id="users", name="User Management"),
        _make_subsystem_row(subsystem_id="payments", name="Payments"),
    ]

    count = adapter.persist(rows)
    assert count == EXPECTED_COUNT_3


def test_subsystems_adapter_persist_verifies_data(
    fresh_gateway: StorageGateway,
    snapshot: SnapshotRef,
) -> None:
    """Persisted data can be retrieved and verified."""
    adapter = SubsystemsAdapter(fresh_gateway, snapshot)
    row = _make_subsystem_row(subsystem_id="billing", name="Billing")
    adapter.persist([row])

    # Query and verify
    result = fresh_gateway.con.execute(
        """
        SELECT subsystem_id, name, risk_level, function_count
        FROM analytics.subsystems
        WHERE repo = ? AND commit = ?
        """,
        [DEMO_REPO, DEMO_COMMIT],
    ).fetchone()

    assert result is not None
    assert result[0] == "billing"
    assert result[1] == "Billing"
    assert result[2] == "moderate"
    assert result[3] == FUNCTION_COUNT_25


# =============================================================================
# SubsystemModulesAdapter Tests
# =============================================================================


def test_subsystem_modules_adapter_table_name(
    fresh_gateway: StorageGateway,
    snapshot: SnapshotRef,
) -> None:
    """Adapter exposes correct table name."""
    adapter = SubsystemModulesAdapter(fresh_gateway, snapshot)
    assert adapter.table_name == "analytics.subsystem_modules"


def test_subsystem_modules_adapter_load_raises(
    fresh_gateway: StorageGateway,
    snapshot: SnapshotRef,
) -> None:
    """Load raises NotImplementedError (write-only adapter)."""
    adapter = SubsystemModulesAdapter(fresh_gateway, snapshot)
    with pytest.raises(NotImplementedError, match="does not support loading"):
        list(adapter.load())


def test_subsystem_modules_adapter_persist_empty(
    fresh_gateway: StorageGateway,
    snapshot: SnapshotRef,
) -> None:
    """Persist empty list returns 0."""
    adapter = SubsystemModulesAdapter(fresh_gateway, snapshot)
    count = adapter.persist([])
    assert count == EXPECTED_COUNT_0


def test_subsystem_modules_adapter_persist_single(
    fresh_gateway: StorageGateway,
    snapshot: SnapshotRef,
) -> None:
    """Persist single subsystem-module mapping."""
    adapter = SubsystemModulesAdapter(fresh_gateway, snapshot)
    row = _make_subsystem_module_row()

    count = adapter.persist([row])
    assert count == EXPECTED_COUNT_1

    # Verify row was inserted
    total = count_rows(
        fresh_gateway.con,
        "SELECT COUNT(*) FROM analytics.subsystem_modules WHERE repo = ? AND commit = ?",
        [DEMO_REPO, DEMO_COMMIT],
    )
    assert total == EXPECTED_COUNT_1


def test_subsystem_modules_adapter_persist_multiple(
    fresh_gateway: StorageGateway,
    snapshot: SnapshotRef,
) -> None:
    """Persist multiple subsystem-module mappings."""
    adapter = SubsystemModulesAdapter(fresh_gateway, snapshot)

    rows = [
        _make_subsystem_module_row(module="auth.core", role="core"),
        _make_subsystem_module_row(module="auth.providers", role="adapter"),
        _make_subsystem_module_row(module="auth.tokens", role="utility"),
    ]

    count = adapter.persist(rows)
    assert count == EXPECTED_COUNT_3


def test_subsystem_modules_adapter_persist_verifies_data(
    fresh_gateway: StorageGateway,
    snapshot: SnapshotRef,
) -> None:
    """Persisted data can be retrieved and verified."""
    adapter = SubsystemModulesAdapter(fresh_gateway, snapshot)
    row = _make_subsystem_module_row(
        subsystem_id="users",
        module="users.service",
        role="service",
    )
    adapter.persist([row])

    # Query and verify
    result = fresh_gateway.con.execute(
        """
        SELECT subsystem_id, module, role
        FROM analytics.subsystem_modules
        WHERE repo = ? AND commit = ?
        """,
        [DEMO_REPO, DEMO_COMMIT],
    ).fetchone()

    assert result is not None
    assert result[0] == "users"
    assert result[1] == "users.service"
    assert result[2] == "service"
