"""Test semantic roles adapter classes.

Test the semantic roles adapters for persisting function and module role
classification data using real DuckDB instances.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import cast

import pytest

from codeintel.analytics.adapters.semantic_roles import (
    SemanticRolesFunctionsAdapter,
    SemanticRolesModulesAdapter,
)
from codeintel.config.primitives import SnapshotRef
from tests._helpers.assertions import (
    expect_equal,
    expect_true,
    require_row,
)
from tests._helpers.context import TestContext, create_test_context
from tests._helpers.contracts import count_rows
from tests._helpers.env_options import EnvOptions
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
def ctx(tmp_path: Path) -> Iterator[TestContext]:
    """
    Create a test context aligned with the demo repo/commit.

    Parameters
    ----------
    tmp_path
        Temporary directory for test artifacts.

    Yields
    ------
    TestContext
        Configured context with schemas applied.
    """
    options = EnvOptions(repo=DEMO_REPO, commit=DEMO_COMMIT)
    context = create_test_context(tmp_path, options=options)
    try:
        yield context
    finally:
        context.close()


@pytest.fixture
def snapshot(ctx: TestContext) -> SnapshotRef:
    """Expose the snapshot from the shared test context.

    Returns
    -------
    SnapshotRef
        Snapshot associated with the shared test context.
    """
    return ctx.snapshot


# =============================================================================
# SemanticRolesFunctionsAdapter Tests
# =============================================================================


def test_functions_adapter_table_name(
    ctx: TestContext,
) -> None:
    """Adapter exposes correct table name."""
    adapter = SemanticRolesFunctionsAdapter(ctx.gateway, ctx.snapshot)
    expect_equal(adapter.table_name, "analytics.semantic_roles_functions")


def test_functions_adapter_load_raises(
    ctx: TestContext,
) -> None:
    """Load raises NotImplementedError (write-only adapter)."""
    adapter = SemanticRolesFunctionsAdapter(ctx.gateway, ctx.snapshot)
    with pytest.raises(NotImplementedError, match="does not support loading"):
        list(adapter.load())


def test_functions_adapter_persist_empty(
    ctx: TestContext,
) -> None:
    """Persist empty list returns 0."""
    adapter = SemanticRolesFunctionsAdapter(ctx.gateway, ctx.snapshot)
    count = adapter.persist([])
    expect_equal(count, EXPECTED_COUNT_0)


def test_functions_adapter_persist_single(
    ctx: TestContext,
) -> None:
    """Persist single function role."""
    adapter = SemanticRolesFunctionsAdapter(ctx.gateway, ctx.snapshot)
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
        ctx.gateway.con,
        "SELECT COUNT(*) FROM analytics.semantic_roles_functions WHERE repo = ? AND commit = ?",
        [DEMO_REPO, DEMO_COMMIT],
    )
    expect_equal(total, EXPECTED_COUNT_1)


def test_functions_adapter_persist_multiple(
    ctx: TestContext,
) -> None:
    """Persist multiple function roles."""
    adapter = SemanticRolesFunctionsAdapter(ctx.gateway, ctx.snapshot)

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
    ctx: TestContext,
) -> None:
    """Persisted data can be retrieved and verified."""
    adapter = SemanticRolesFunctionsAdapter(ctx.gateway, ctx.snapshot)
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
    result = ctx.gateway.con.execute(
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


def test_functions_adapter_applies_defaults(
    ctx: TestContext,
) -> None:
    """Persist fills optional fields and created_at."""
    adapter = SemanticRolesFunctionsAdapter(ctx.gateway, ctx.snapshot)
    row = (DEMO_REPO, DEMO_COMMIT, TEST_GOID_12345, "api_handler", CONFIDENCE_0_85)

    adapter.persist([row])

    result = ctx.gateway.con.execute(
        """
        SELECT framework, role_sources_json, created_at
        FROM analytics.semantic_roles_functions
        WHERE repo = ? AND commit = ?
        """,
        [DEMO_REPO, DEMO_COMMIT],
    ).fetchone()
    db_row = require_row(result, message="Expected semantic role function row with defaults")
    expect_equal(db_row[0], None)
    expect_equal(db_row[1], "[]")
    expect_true(db_row[2] is not None)


def test_functions_adapter_rejects_bad_length(
    ctx: TestContext,
) -> None:
    """Persist raises when tuple length is invalid."""
    adapter = SemanticRolesFunctionsAdapter(ctx.gateway, ctx.snapshot)
    with pytest.raises(ValueError, match="legacy 5-tuple"):
        adapter.persist([(DEMO_REPO, DEMO_COMMIT, TEST_GOID_12345, "api_handler")])


# =============================================================================
# SemanticRolesModulesAdapter Tests
# =============================================================================


def test_modules_adapter_table_name(
    ctx: TestContext,
) -> None:
    """Adapter exposes correct table name."""
    adapter = SemanticRolesModulesAdapter(ctx.gateway, ctx.snapshot)
    expect_equal(adapter.table_name, "analytics.semantic_roles_modules")


def test_modules_adapter_load_raises(
    ctx: TestContext,
) -> None:
    """Load raises NotImplementedError (write-only adapter)."""
    adapter = SemanticRolesModulesAdapter(ctx.gateway, ctx.snapshot)
    with pytest.raises(NotImplementedError, match="does not support loading"):
        list(adapter.load())


def test_modules_adapter_persist_empty(
    ctx: TestContext,
) -> None:
    """Persist empty list returns 0."""
    adapter = SemanticRolesModulesAdapter(ctx.gateway, ctx.snapshot)
    count = adapter.persist([])
    expect_equal(count, EXPECTED_COUNT_0)


def test_modules_adapter_persist_single(
    ctx: TestContext,
) -> None:
    """Persist single module role."""
    adapter = SemanticRolesModulesAdapter(ctx.gateway, ctx.snapshot)
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
        ctx.gateway.con,
        "SELECT COUNT(*) FROM analytics.semantic_roles_modules WHERE repo = ? AND commit = ?",
        [DEMO_REPO, DEMO_COMMIT],
    )
    expect_equal(total, EXPECTED_COUNT_1)


def test_modules_adapter_persist_multiple(
    ctx: TestContext,
) -> None:
    """Persist multiple module roles."""
    adapter = SemanticRolesModulesAdapter(ctx.gateway, ctx.snapshot)

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
    ctx: TestContext,
) -> None:
    """Persisted data can be retrieved and verified."""
    adapter = SemanticRolesModulesAdapter(ctx.gateway, ctx.snapshot)
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
    result = ctx.gateway.con.execute(
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


def test_modules_adapter_applies_defaults(
    ctx: TestContext,
) -> None:
    """Persist fills optional fields and created_at for module rows."""
    adapter = SemanticRolesModulesAdapter(ctx.gateway, ctx.snapshot)
    row = (DEMO_REPO, DEMO_COMMIT, "api.users", "service", CONFIDENCE_0_90)

    adapter.persist([row])

    result = ctx.gateway.con.execute(
        """
        SELECT role_sources_json, created_at
        FROM analytics.semantic_roles_modules
        WHERE repo = ? AND commit = ?
        """,
        [DEMO_REPO, DEMO_COMMIT],
    ).fetchone()
    db_row = require_row(result, message="Expected semantic role module row with defaults")
    expect_equal(db_row[0], "[]")
    expect_true(db_row[1] is not None)


def test_modules_adapter_rejects_bad_length(
    ctx: TestContext,
) -> None:
    """Persist raises when module tuple length is invalid."""
    adapter = SemanticRolesModulesAdapter(ctx.gateway, ctx.snapshot)
    with pytest.raises(ValueError, match="legacy 5-tuple"):
        adapter.persist([(DEMO_REPO, DEMO_COMMIT, "api.users", "service")])
