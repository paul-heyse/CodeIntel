"""Test entrypoint adapter classes.

Test the entrypoint-specific adapters for persisting entrypoint detection
and test mapping data using real DuckDB instances.
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from typing import Iterator

import pytest

from codeintel.analytics.adapters.entrypoints import (
    EntrypointsAdapter,
    EntrypointTestsAdapter,
)
from codeintel.config.primitives import SnapshotRef
from tests._helpers.assertions import (
    expect_equal,
    expect_is_none,
    expect_is_not_none,
)
from tests._helpers.contracts import count_rows
from tests._helpers.context import TestContext, create_test_context
from tests._helpers.env_options import EnvOptions
from tests._helpers.rows import (
    EntrypointPayloadSeed,
    EntrypointTestPayloadSeed,
    entrypoint_payload,
    entrypoint_test_payload,
)

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
# Fixtures
# =============================================================================


@pytest.fixture
def ctx(tmp_path: Path) -> Iterator[TestContext]:
    """
    Create a test context aligned to the demo repo/commit constants.

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
    """Expose the snapshot from the shared test context."""
    return ctx.snapshot


# =============================================================================
# EntrypointsAdapter Tests
# =============================================================================


def test_entrypoints_adapter_table_name(
    ctx: TestContext,
) -> None:
    """Adapter exposes correct table name."""
    adapter = EntrypointsAdapter(ctx.gateway, ctx.snapshot)
    expect_equal(adapter.table_name, "analytics.entrypoints")


def test_entrypoints_adapter_load_raises(
    ctx: TestContext,
) -> None:
    """Load raises NotImplementedError (write-only adapter)."""
    adapter = EntrypointsAdapter(ctx.gateway, ctx.snapshot)
    with pytest.raises(NotImplementedError, match="does not support loading"):
        list(adapter.load())


def test_entrypoints_adapter_persist_empty(
    ctx: TestContext,
) -> None:
    """Persist empty list returns 0."""
    adapter = EntrypointsAdapter(ctx.gateway, ctx.snapshot)
    count = adapter.persist([])
    expect_equal(count, EXPECTED_COUNT_0)


def test_entrypoints_adapter_persist_http_endpoint(
    ctx: TestContext,
) -> None:
    """Persist HTTP endpoint entrypoint."""
    adapter = EntrypointsAdapter(ctx.gateway, ctx.snapshot)
    row = entrypoint_payload(
        EntrypointPayloadSeed(
            entrypoint_id="ep_api_users_get",
            handler_qualname="api.users.get_users",
            handler_module="api.users",
            handler_rel_path="src/api/users.py",
            handler_goid_h128=TEST_GOID_12345,
            tests_touching=5,
            slow_tests=1,
            entrypoint_coverage_ratio=TEST_COVERAGE_RATIO_0_85,
            tags=["api", "users"],
            owners=["team-backend"],
            subsystem_id="users",
            subsystem_name="User Management",
            repo=DEMO_REPO,
            commit=DEMO_COMMIT,
        )
    )

    count = adapter.persist([row])
    expect_equal(count, EXPECTED_COUNT_1)

    # Verify row was inserted
    total = count_rows(
        ctx.gateway.con,
        "SELECT COUNT(*) FROM analytics.entrypoints WHERE repo = ? AND commit = ?",
        [DEMO_REPO, DEMO_COMMIT],
    )
    expect_equal(total, EXPECTED_COUNT_1)


def test_entrypoints_adapter_persist_cli_command(
    ctx: TestContext,
) -> None:
    """Persist CLI command entrypoint."""
    adapter = EntrypointsAdapter(ctx.gateway, ctx.snapshot)
    row = entrypoint_payload(
        EntrypointPayloadSeed(
            entrypoint_id="ep_cli_migrate",
            kind="cli_command",
            framework="click",
            handler_goid_h128=TEST_GOID_67890,
            handler_qualname="cli.migrate.run_migration",
            handler_module="cli.migrate",
            handler_rel_path="src/cli/migrate.py",
            command_name="migrate",
            arguments_schema={"version": "str", "dry_run": "bool"},
            route_path=None,
            http_method=None,
            status_codes=None,
            auth_required=False,
            extra={"group": "database"},
            tags=["cli", "database"],
            owners=["team-platform"],
            subsystem_id="db",
            subsystem_name="Database",
            entrypoint_coverage_ratio=0.9,
            repo=DEMO_REPO,
            commit=DEMO_COMMIT,
        )
    )

    count = adapter.persist([row])
    expect_equal(count, EXPECTED_COUNT_1)

    # Verify CLI-specific fields
    result = ctx.gateway.con.execute(
        """
        SELECT kind, command_name, http_method
        FROM analytics.entrypoints
        WHERE repo = ? AND commit = ?
        """,
        [DEMO_REPO, DEMO_COMMIT],
    ).fetchone()
    row = expect_is_not_none(result)
    expect_equal(row[0], "cli_command")
    expect_equal(row[1], "migrate")
    expect_is_none(row[2])  # http_method is null for CLI


def test_entrypoints_adapter_persist_multiple(
    ctx: TestContext,
) -> None:
    """Persist multiple entrypoints."""
    adapter = EntrypointsAdapter(ctx.gateway, ctx.snapshot)

    rows = [
        entrypoint_payload(
            EntrypointPayloadSeed(
                entrypoint_id="ep_1",
                handler_qualname="api.get",
                handler_module="api",
                handler_rel_path="src/api.py",
                handler_goid_h128=TEST_GOID_12345,
                repo=DEMO_REPO,
                commit=DEMO_COMMIT,
            )
        ),
        entrypoint_payload(
            EntrypointPayloadSeed(
                entrypoint_id="ep_2",
                kind="cli_command",
                framework="click",
                handler_goid_h128=TEST_GOID_67890,
                handler_qualname="cli.cmd",
                handler_module="cli",
                handler_rel_path="src/cli/cmd.py",
                command_name="cmd",
                route_path=None,
                http_method=None,
                status_codes=None,
                auth_required=False,
                repo=DEMO_REPO,
                commit=DEMO_COMMIT,
            )
        ),
    ]

    count = adapter.persist(rows)
    expect_equal(count, EXPECTED_COUNT_2)


# =============================================================================
# EntrypointTestsAdapter Tests
# =============================================================================


def test_entrypoint_tests_adapter_table_name(
    ctx: TestContext,
) -> None:
    """Adapter exposes correct table name."""
    adapter = EntrypointTestsAdapter(ctx.gateway, ctx.snapshot)
    expect_equal(adapter.table_name, "analytics.entrypoint_tests")


def test_entrypoint_tests_adapter_load_raises(
    ctx: TestContext,
) -> None:
    """Load raises NotImplementedError (write-only adapter)."""
    adapter = EntrypointTestsAdapter(ctx.gateway, ctx.snapshot)
    with pytest.raises(NotImplementedError, match="does not support loading"):
        list(adapter.load())


def test_entrypoint_tests_adapter_persist_empty(
    ctx: TestContext,
) -> None:
    """Persist empty list returns 0."""
    adapter = EntrypointTestsAdapter(ctx.gateway, ctx.snapshot)
    count = adapter.persist([])
    expect_equal(count, EXPECTED_COUNT_0)


def test_entrypoint_tests_adapter_persist_single(
    ctx: TestContext,
) -> None:
    """Persist single entrypoint-test mapping."""
    adapter = EntrypointTestsAdapter(ctx.gateway, ctx.snapshot)
    row = entrypoint_test_payload(
        EntrypointTestPayloadSeed(
            entrypoint_id="ep_api_users_get",
            test_id="test_get_users_success",
            test_goid_h128=TEST_GOID_12345,
            coverage_ratio=TEST_COVERAGE_RATIO_0_85,
            status="passed",
            duration_ms=TEST_DURATION_MS_150,
            repo=DEMO_REPO,
            commit=DEMO_COMMIT,
        )
    )

    count = adapter.persist([row])
    expect_equal(count, EXPECTED_COUNT_1)

    # Verify row was inserted
    total = count_rows(
        ctx.gateway.con,
        "SELECT COUNT(*) FROM analytics.entrypoint_tests WHERE repo = ? AND commit = ?",
        [DEMO_REPO, DEMO_COMMIT],
    )
    expect_equal(total, EXPECTED_COUNT_1)


def test_entrypoint_tests_adapter_persist_multiple(
    ctx: TestContext,
) -> None:
    """Persist multiple entrypoint-test mappings."""
    adapter = EntrypointTestsAdapter(ctx.gateway, ctx.snapshot)

    rows = [
        entrypoint_test_payload(
            EntrypointTestPayloadSeed(
                entrypoint_id="ep_api_users_get",
                test_id="test_1",
                test_goid_h128=Decimal(1001),
                coverage_ratio=TEST_COVERAGE_RATIO_0_85,
                status="passed",
                duration_ms=TEST_DURATION_MS_150,
                repo=DEMO_REPO,
                commit=DEMO_COMMIT,
            )
        ),
        entrypoint_test_payload(
            EntrypointTestPayloadSeed(
                entrypoint_id="ep_api_users_get",
                test_id="test_2",
                test_goid_h128=Decimal(1002),
                coverage_ratio=TEST_COVERAGE_RATIO_0_85,
                status="passed",
                duration_ms=TEST_DURATION_MS_150,
                repo=DEMO_REPO,
                commit=DEMO_COMMIT,
            )
        ),
    ]

    count = adapter.persist(rows)
    expect_equal(count, EXPECTED_COUNT_2)


def test_entrypoint_tests_adapter_persist_verifies_data(
    ctx: TestContext,
) -> None:
    """Persisted data can be retrieved and verified."""
    adapter = EntrypointTestsAdapter(ctx.gateway, ctx.snapshot)
    row = entrypoint_test_payload(
        EntrypointTestPayloadSeed(
            entrypoint_id="ep_verify",
            test_id="test_verify",
            test_goid_h128=TEST_GOID_12345,
            coverage_ratio=TEST_COVERAGE_RATIO_0_85,
            status="passed",
            duration_ms=TEST_DURATION_MS_150,
            repo=DEMO_REPO,
            commit=DEMO_COMMIT,
        )
    )
    adapter.persist([row])

    # Query and verify
    result = ctx.gateway.con.execute(
        """
        SELECT entrypoint_id, test_id, status, coverage_ratio
        FROM analytics.entrypoint_tests
        WHERE repo = ? AND commit = ?
        """,
        [DEMO_REPO, DEMO_COMMIT],
    ).fetchone()

    row = expect_is_not_none(result)
    expect_equal(row[0], "ep_verify")
    expect_equal(row[1], "test_verify")
    expect_equal(row[2], "passed")
    expect_equal(float(row[3]), pytest.approx(TEST_COVERAGE_RATIO_0_85))
