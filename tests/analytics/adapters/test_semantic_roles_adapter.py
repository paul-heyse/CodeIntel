"""Test semantic roles adapter classes.

Test the semantic roles adapters for persisting function and module role
classification data using real DuckDB instances.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import pytest

from codeintel.analytics.adapters.semantic_roles import (
    SemanticRolesFunctionsAdapter,
    SemanticRolesModulesAdapter,
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
TEST_GOID_12345 = Decimal(12345)
TEST_GOID_67890 = Decimal(67890)
TEST_GOID_11111 = Decimal(11111)
CONFIDENCE_0_85 = 0.85
CONFIDENCE_0_90 = 0.9
CONFIDENCE_0_75 = 0.75


# =============================================================================
# Test Data Factories
# =============================================================================


def _make_function_role_row(
    goid: Decimal = TEST_GOID_12345,
    role: str = "api_handler",
    confidence: float = CONFIDENCE_0_85,
) -> tuple[object, ...]:
    """
    Create a function semantic role row for testing.

    Parameters
    ----------
    goid
        Function global object ID.
    role
        Classified semantic role.
    confidence
        Role classification confidence.

    Returns
    -------
    tuple[object, ...]
        Function role row tuple.
    """
    # Columns: repo, commit, function_goid_h128, role, framework, role_confidence,
    #          role_sources_json, created_at
    return (
        DEMO_REPO,
        DEMO_COMMIT,
        goid,
        role,
        "fastapi",  # framework
        confidence,
        ["decorator", "path_hint"],  # role_sources_json
        datetime.now(tz=UTC),
    )


def _make_module_role_row(
    module: str = "api.users",
    role: str = "service",
    confidence: float = CONFIDENCE_0_90,
) -> tuple[object, ...]:
    """
    Create a module semantic role row for testing.

    Parameters
    ----------
    module
        Module name.
    role
        Classified semantic role.
    confidence
        Role classification confidence.

    Returns
    -------
    tuple[object, ...]
        Module role row tuple.
    """
    # Columns: repo, commit, module, role, role_confidence, role_sources_json, created_at
    return (
        DEMO_REPO,
        DEMO_COMMIT,
        module,
        role,
        confidence,
        ["path_hint", "module_tags"],  # role_sources_json
        datetime.now(tz=UTC),
    )


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
# SemanticRolesFunctionsAdapter Tests
# =============================================================================


def test_functions_adapter_table_name(
    fresh_gateway: StorageGateway,
    snapshot: SnapshotRef,
) -> None:
    """Adapter exposes correct table name."""
    adapter = SemanticRolesFunctionsAdapter(fresh_gateway, snapshot)
    assert adapter.table_name == "analytics.semantic_roles_functions"


def test_functions_adapter_load_raises(
    fresh_gateway: StorageGateway,
    snapshot: SnapshotRef,
) -> None:
    """Load raises NotImplementedError (write-only adapter)."""
    adapter = SemanticRolesFunctionsAdapter(fresh_gateway, snapshot)
    with pytest.raises(NotImplementedError, match="does not support loading"):
        list(adapter.load())


def test_functions_adapter_persist_empty(
    fresh_gateway: StorageGateway,
    snapshot: SnapshotRef,
) -> None:
    """Persist empty list returns 0."""
    adapter = SemanticRolesFunctionsAdapter(fresh_gateway, snapshot)
    count = adapter.persist([])
    assert count == EXPECTED_COUNT_0


def test_functions_adapter_persist_single(
    fresh_gateway: StorageGateway,
    snapshot: SnapshotRef,
) -> None:
    """Persist single function role."""
    adapter = SemanticRolesFunctionsAdapter(fresh_gateway, snapshot)
    row = _make_function_role_row()

    count = adapter.persist([row])
    assert count == EXPECTED_COUNT_1

    # Verify row was inserted
    total = count_rows(
        fresh_gateway.con,
        "SELECT COUNT(*) FROM analytics.semantic_roles_functions WHERE repo = ? AND commit = ?",
        [DEMO_REPO, DEMO_COMMIT],
    )
    assert total == EXPECTED_COUNT_1


def test_functions_adapter_persist_multiple(
    fresh_gateway: StorageGateway,
    snapshot: SnapshotRef,
) -> None:
    """Persist multiple function roles."""
    adapter = SemanticRolesFunctionsAdapter(fresh_gateway, snapshot)

    rows = [
        _make_function_role_row(goid=TEST_GOID_12345, role="api_handler"),
        _make_function_role_row(goid=TEST_GOID_67890, role="repository"),
        _make_function_role_row(goid=TEST_GOID_11111, role="validator"),
    ]

    count = adapter.persist(rows)
    assert count == EXPECTED_COUNT_3


def test_functions_adapter_persist_verifies_data(
    fresh_gateway: StorageGateway,
    snapshot: SnapshotRef,
) -> None:
    """Persisted data can be retrieved and verified."""
    adapter = SemanticRolesFunctionsAdapter(fresh_gateway, snapshot)
    row = _make_function_role_row(
        goid=TEST_GOID_12345,
        role="test_helper",
        confidence=CONFIDENCE_0_75,
    )
    adapter.persist([row])

    # Query and verify
    result = fresh_gateway.con.execute(
        """
        SELECT role, role_confidence
        FROM analytics.semantic_roles_functions
        WHERE repo = ? AND commit = ?
        """,
        [DEMO_REPO, DEMO_COMMIT],
    ).fetchone()

    assert result is not None
    assert result[0] == "test_helper"
    assert float(result[1]) == pytest.approx(CONFIDENCE_0_75)


# =============================================================================
# SemanticRolesModulesAdapter Tests
# =============================================================================


def test_modules_adapter_table_name(
    fresh_gateway: StorageGateway,
    snapshot: SnapshotRef,
) -> None:
    """Adapter exposes correct table name."""
    adapter = SemanticRolesModulesAdapter(fresh_gateway, snapshot)
    assert adapter.table_name == "analytics.semantic_roles_modules"


def test_modules_adapter_load_raises(
    fresh_gateway: StorageGateway,
    snapshot: SnapshotRef,
) -> None:
    """Load raises NotImplementedError (write-only adapter)."""
    adapter = SemanticRolesModulesAdapter(fresh_gateway, snapshot)
    with pytest.raises(NotImplementedError, match="does not support loading"):
        list(adapter.load())


def test_modules_adapter_persist_empty(
    fresh_gateway: StorageGateway,
    snapshot: SnapshotRef,
) -> None:
    """Persist empty list returns 0."""
    adapter = SemanticRolesModulesAdapter(fresh_gateway, snapshot)
    count = adapter.persist([])
    assert count == EXPECTED_COUNT_0


def test_modules_adapter_persist_single(
    fresh_gateway: StorageGateway,
    snapshot: SnapshotRef,
) -> None:
    """Persist single module role."""
    adapter = SemanticRolesModulesAdapter(fresh_gateway, snapshot)
    row = _make_module_role_row()

    count = adapter.persist([row])
    assert count == EXPECTED_COUNT_1

    # Verify row was inserted
    total = count_rows(
        fresh_gateway.con,
        "SELECT COUNT(*) FROM analytics.semantic_roles_modules WHERE repo = ? AND commit = ?",
        [DEMO_REPO, DEMO_COMMIT],
    )
    assert total == EXPECTED_COUNT_1


def test_modules_adapter_persist_multiple(
    fresh_gateway: StorageGateway,
    snapshot: SnapshotRef,
) -> None:
    """Persist multiple module roles."""
    adapter = SemanticRolesModulesAdapter(fresh_gateway, snapshot)

    rows = [
        _make_module_role_row(module="api.users", role="api"),
        _make_module_role_row(module="db.models", role="repository"),
        _make_module_role_row(module="cli.commands", role="cli"),
    ]

    count = adapter.persist(rows)
    assert count == EXPECTED_COUNT_3


def test_modules_adapter_persist_verifies_data(
    fresh_gateway: StorageGateway,
    snapshot: SnapshotRef,
) -> None:
    """Persisted data can be retrieved and verified."""
    adapter = SemanticRolesModulesAdapter(fresh_gateway, snapshot)
    row = _make_module_role_row(
        module="utils.helpers",
        role="utility",
        confidence=CONFIDENCE_0_85,
    )
    adapter.persist([row])

    # Query and verify
    result = fresh_gateway.con.execute(
        """
        SELECT module, role, role_confidence
        FROM analytics.semantic_roles_modules
        WHERE repo = ? AND commit = ?
        """,
        [DEMO_REPO, DEMO_COMMIT],
    ).fetchone()

    assert result is not None
    assert result[0] == "utils.helpers"
    assert result[1] == "utility"
    assert float(result[2]) == pytest.approx(CONFIDENCE_0_85)
