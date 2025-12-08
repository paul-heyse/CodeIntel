"""Test semantic roles adapter classes.

Test the semantic roles adapters for persisting function and module role
classification data using real DuckDB instances.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest

from codeintel.analytics.adapters.semantic_roles import (
    SemanticRolesFunctionsAdapter,
    SemanticRolesModulesAdapter,
)
from codeintel.config.primitives import SnapshotRef
from codeintel.storage.gateway import StorageGateway
from tests._helpers.assertions import (
    expect_equal,
    require_row,
)
from tests._helpers.contracts import count_rows
from tests._helpers.rows import (
    SemanticRoleFunctionSeed,
    SemanticRoleModuleSeed,
    semantic_role_function_row,
    semantic_role_module_row,
)

# =============================================================================
# Constants
# =============================================================================

DEMO_REPO = "demo/repo"
DEMO_COMMIT = "abc123def456"
EXPECTED_COUNT_0 = 0
EXPECTED_COUNT_1 = 1
EXPECTED_COUNT_2 = 2
EXPECTED_COUNT_3 = 3
TEST_GOID_12345 = 12345
TEST_GOID_67890 = 67890
TEST_GOID_11111 = 11111
CONFIDENCE_0_85 = 0.85
CONFIDENCE_0_90 = 0.9
CONFIDENCE_0_75 = 0.75


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
    expect_equal(adapter.table_name, "analytics.semantic_roles_functions")


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
    expect_equal(count, EXPECTED_COUNT_0)


def test_functions_adapter_persist_single(
    fresh_gateway: StorageGateway,
    snapshot: SnapshotRef,
) -> None:
    """Persist single function role."""
    adapter = SemanticRolesFunctionsAdapter(fresh_gateway, snapshot)
    row = semantic_role_function_row(
        SemanticRoleFunctionSeed(
            goid=int(TEST_GOID_12345),
            role="api_handler",
            confidence=CONFIDENCE_0_85,
            repo=DEMO_REPO,
            commit=DEMO_COMMIT,
        )
    )

    count = adapter.persist([row])
    expect_equal(count, EXPECTED_COUNT_1)

    # Verify row was inserted
    total = count_rows(
        fresh_gateway.con,
        "SELECT COUNT(*) FROM analytics.semantic_roles_functions WHERE repo = ? AND commit = ?",
        [DEMO_REPO, DEMO_COMMIT],
    )
    expect_equal(total, EXPECTED_COUNT_1)


def test_functions_adapter_persist_multiple(
    fresh_gateway: StorageGateway,
    snapshot: SnapshotRef,
) -> None:
    """Persist multiple function roles."""
    adapter = SemanticRolesFunctionsAdapter(fresh_gateway, snapshot)

    rows = [
        semantic_role_function_row(
            SemanticRoleFunctionSeed(
                goid=int(TEST_GOID_12345),
                role="api_handler",
                confidence=CONFIDENCE_0_85,
                repo=DEMO_REPO,
                commit=DEMO_COMMIT,
            )
        ),
        semantic_role_function_row(
            SemanticRoleFunctionSeed(
                goid=int(TEST_GOID_67890),
                role="repository",
                confidence=CONFIDENCE_0_90,
                repo=DEMO_REPO,
                commit=DEMO_COMMIT,
            )
        ),
        semantic_role_function_row(
            SemanticRoleFunctionSeed(
                goid=int(TEST_GOID_11111),
                role="validator",
                confidence=CONFIDENCE_0_75,
                repo=DEMO_REPO,
                commit=DEMO_COMMIT,
            )
        ),
    ]

    count = adapter.persist(rows)
    expect_equal(count, EXPECTED_COUNT_3)


def test_functions_adapter_persist_verifies_data(
    fresh_gateway: StorageGateway,
    snapshot: SnapshotRef,
) -> None:
    """Persisted data can be retrieved and verified."""
    adapter = SemanticRolesFunctionsAdapter(fresh_gateway, snapshot)
    row = semantic_role_function_row(
        SemanticRoleFunctionSeed(
            goid=int(TEST_GOID_12345),
            role="test_helper",
            confidence=CONFIDENCE_0_75,
            repo=DEMO_REPO,
            commit=DEMO_COMMIT,
        )
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

    row = require_row(result, message="Expected semantic role function row")
    confidence = float(cast("float | int | str", row[1]))
    expect_equal(row[0], "test_helper")
    expect_equal(confidence, pytest.approx(CONFIDENCE_0_75))


# =============================================================================
# SemanticRolesModulesAdapter Tests
# =============================================================================


def test_modules_adapter_table_name(
    fresh_gateway: StorageGateway,
    snapshot: SnapshotRef,
) -> None:
    """Adapter exposes correct table name."""
    adapter = SemanticRolesModulesAdapter(fresh_gateway, snapshot)
    expect_equal(adapter.table_name, "analytics.semantic_roles_modules")


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
    expect_equal(count, EXPECTED_COUNT_0)


def test_modules_adapter_persist_single(
    fresh_gateway: StorageGateway,
    snapshot: SnapshotRef,
) -> None:
    """Persist single module role."""
    adapter = SemanticRolesModulesAdapter(fresh_gateway, snapshot)
    row = semantic_role_module_row(
        SemanticRoleModuleSeed(
            module="api.users",
            role="service",
            confidence=CONFIDENCE_0_90,
            repo=DEMO_REPO,
            commit=DEMO_COMMIT,
        )
    )

    count = adapter.persist([row])
    expect_equal(count, EXPECTED_COUNT_1)

    # Verify row was inserted
    total = count_rows(
        fresh_gateway.con,
        "SELECT COUNT(*) FROM analytics.semantic_roles_modules WHERE repo = ? AND commit = ?",
        [DEMO_REPO, DEMO_COMMIT],
    )
    expect_equal(total, EXPECTED_COUNT_1)


def test_modules_adapter_persist_multiple(
    fresh_gateway: StorageGateway,
    snapshot: SnapshotRef,
) -> None:
    """Persist multiple module roles."""
    adapter = SemanticRolesModulesAdapter(fresh_gateway, snapshot)

    rows = [
        semantic_role_module_row(
            SemanticRoleModuleSeed(
                module="api.users",
                role="api",
                confidence=CONFIDENCE_0_90,
                repo=DEMO_REPO,
                commit=DEMO_COMMIT,
            )
        ),
        semantic_role_module_row(
            SemanticRoleModuleSeed(
                module="db.models",
                role="repository",
                confidence=CONFIDENCE_0_85,
                repo=DEMO_REPO,
                commit=DEMO_COMMIT,
            )
        ),
        semantic_role_module_row(
            SemanticRoleModuleSeed(
                module="cli.commands",
                role="cli",
                confidence=CONFIDENCE_0_75,
                repo=DEMO_REPO,
                commit=DEMO_COMMIT,
            )
        ),
    ]

    count = adapter.persist(rows)
    expect_equal(count, EXPECTED_COUNT_3)


def test_modules_adapter_persist_verifies_data(
    fresh_gateway: StorageGateway,
    snapshot: SnapshotRef,
) -> None:
    """Persisted data can be retrieved and verified."""
    adapter = SemanticRolesModulesAdapter(fresh_gateway, snapshot)
    row = semantic_role_module_row(
        SemanticRoleModuleSeed(
            module="utils.helpers",
            role="utility",
            confidence=CONFIDENCE_0_85,
            repo=DEMO_REPO,
            commit=DEMO_COMMIT,
        )
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

    row = require_row(result, message="Expected semantic role module row")
    confidence = float(cast("float | int | str", row[2]))
    expect_equal(row[0], "utils.helpers")
    expect_equal(row[1], "utility")
    expect_equal(confidence, pytest.approx(CONFIDENCE_0_85))
