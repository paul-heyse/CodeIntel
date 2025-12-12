"""Test subsystem adapter classes.

Test the subsystem-specific adapters for persisting subsystem classification
and module mapping data using real DuckDB instances.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from codeintel.analytics.adapters.subsystems import (
    SubsystemModulesAdapter,
    SubsystemsAdapter,
)
from tests._helpers.assertions import (
    expect_equal,
    require_row,
)
from tests._helpers.context import create_test_context
from tests._helpers.contracts import count_rows
from tests._helpers.env_options import EnvOptions
from tests._helpers.rows import (
    SubsystemModulePayloadSeed,
    SubsystemPayloadSeed,
    subsystem_module_payload,
    subsystem_payload,
)

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from codeintel.config.primitives import SnapshotRef
    from tests._helpers.context import TestContext


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


@pytest.fixture
def ctx(tmp_path: Path) -> Iterator[TestContext]:
    """
    Create a test context aligned with the demo repo/commit identifiers.

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


def test_subsystems_adapter_table_name(
    ctx: TestContext,
) -> None:
    """Adapter exposes correct table name."""
    adapter = SubsystemsAdapter(ctx.gateway, ctx.snapshot)
    expect_equal(adapter.table_name, "analytics.subsystems")


def test_subsystems_adapter_load_raises(
    ctx: TestContext,
) -> None:
    """Load raises NotImplementedError (write-only adapter)."""
    adapter = SubsystemsAdapter(ctx.gateway, ctx.snapshot)
    with pytest.raises(NotImplementedError, match="does not support loading"):
        list(adapter.load())


def test_subsystems_adapter_persist_empty(
    ctx: TestContext,
) -> None:
    """Persist empty list returns 0."""
    adapter = SubsystemsAdapter(ctx.gateway, ctx.snapshot)
    count = adapter.persist([])
    expect_equal(count, EXPECTED_COUNT_0)


def test_subsystems_adapter_persist_single(
    ctx: TestContext,
) -> None:
    """Persist single subsystem."""
    adapter = SubsystemsAdapter(ctx.gateway, ctx.snapshot)
    row = subsystem_payload(
        SubsystemPayloadSeed(
            subsystem_id="auth",
            name="Authentication",
            description="The Authentication subsystem handles related functionality.",
            module_count=MODULE_COUNT_5,
            modules_json=["auth.core", "auth.providers", "auth.tokens"],
            entrypoints_json=["POST /login", "POST /logout", "GET /me"],
            internal_edge_count=20,
            external_edge_count=8,
            fan_in=FAN_IN_10,
            fan_out=FAN_OUT_15,
            function_count=FUNCTION_COUNT_25,
            avg_risk_score=AVG_RISK_0_35,
            max_risk_score=MAX_RISK_0_85,
            high_risk_function_count=HIGH_RISK_COUNT_2,
            risk_level="moderate",
            repo=DEMO_REPO,
            commit=DEMO_COMMIT,
        )
    )

    count = adapter.persist([row])
    expect_equal(count, EXPECTED_COUNT_1)

    total = count_rows(
        ctx.gateway.con,
        "SELECT COUNT(*) FROM analytics.subsystems WHERE repo = ? AND commit = ?",
        [DEMO_REPO, DEMO_COMMIT],
    )
    expect_equal(total, EXPECTED_COUNT_1)


def test_subsystems_adapter_persist_multiple(
    ctx: TestContext,
) -> None:
    """Persist multiple subsystems."""
    adapter = SubsystemsAdapter(ctx.gateway, ctx.snapshot)

    rows = [
        subsystem_payload(
            SubsystemPayloadSeed(
                subsystem_id="auth",
                name="Authentication",
                description="The Authentication subsystem handles related functionality.",
                module_count=MODULE_COUNT_5,
                modules_json=["auth.core", "auth.providers", "auth.tokens"],
                entrypoints_json=["POST /login", "POST /logout", "GET /me"],
                internal_edge_count=20,
                external_edge_count=8,
                fan_in=FAN_IN_10,
                fan_out=FAN_OUT_15,
                function_count=FUNCTION_COUNT_25,
                avg_risk_score=AVG_RISK_0_35,
                max_risk_score=MAX_RISK_0_85,
                high_risk_function_count=HIGH_RISK_COUNT_2,
                risk_level="moderate",
                repo=DEMO_REPO,
                commit=DEMO_COMMIT,
            )
        ),
        subsystem_payload(
            SubsystemPayloadSeed(
                subsystem_id="users",
                name="User Management",
                description="The User Management subsystem handles related functionality.",
                module_count=MODULE_COUNT_5,
                modules_json=["users.api"],
                entrypoints_json=["GET /users"],
                internal_edge_count=10,
                external_edge_count=5,
                fan_in=5,
                fan_out=6,
                function_count=FUNCTION_COUNT_25,
                avg_risk_score=AVG_RISK_0_35,
                max_risk_score=MAX_RISK_0_85,
                high_risk_function_count=HIGH_RISK_COUNT_2,
                risk_level="moderate",
                repo=DEMO_REPO,
                commit=DEMO_COMMIT,
            )
        ),
        subsystem_payload(
            SubsystemPayloadSeed(
                subsystem_id="payments",
                name="Payments",
                description="The Payments subsystem handles related functionality.",
                module_count=MODULE_COUNT_5,
                modules_json=["payments.core"],
                entrypoints_json=["POST /pay"],
                internal_edge_count=12,
                external_edge_count=4,
                fan_in=4,
                fan_out=5,
                function_count=FUNCTION_COUNT_25,
                avg_risk_score=AVG_RISK_0_35,
                max_risk_score=MAX_RISK_0_85,
                high_risk_function_count=HIGH_RISK_COUNT_2,
                risk_level="moderate",
                repo=DEMO_REPO,
                commit=DEMO_COMMIT,
            )
        ),
    ]

    count = adapter.persist(rows)
    expect_equal(count, EXPECTED_COUNT_3)


def test_subsystems_adapter_persist_verifies_data(
    ctx: TestContext,
) -> None:
    """Persisted data can be retrieved and verified."""
    adapter = SubsystemsAdapter(ctx.gateway, ctx.snapshot)
    row = subsystem_payload(
        SubsystemPayloadSeed(
            subsystem_id="billing",
            name="Billing",
            description="The Billing subsystem handles related functionality.",
            module_count=MODULE_COUNT_5,
            modules_json=["billing.core"],
            entrypoints_json=["POST /billing"],
            internal_edge_count=5,
            external_edge_count=2,
            fan_in=2,
            fan_out=3,
            function_count=FUNCTION_COUNT_25,
            avg_risk_score=AVG_RISK_0_35,
            max_risk_score=MAX_RISK_0_85,
            high_risk_function_count=HIGH_RISK_COUNT_2,
            risk_level="moderate",
            repo=DEMO_REPO,
            commit=DEMO_COMMIT,
        )
    )
    adapter.persist([row])

    result = ctx.gateway.con.execute(
        """
        SELECT subsystem_id, name, risk_level, function_count
        FROM analytics.subsystems
        WHERE repo = ? AND commit = ?
        """,
        [DEMO_REPO, DEMO_COMMIT],
    ).fetchone()

    row = require_row(result, message="Expected subsystem row")
    expect_equal(row[0], "billing")
    expect_equal(row[1], "Billing")
    expect_equal(row[2], "moderate")
    expect_equal(row[3], FUNCTION_COUNT_25)


def test_subsystem_modules_adapter_table_name(
    ctx: TestContext,
) -> None:
    """Adapter exposes correct table name."""
    adapter = SubsystemModulesAdapter(ctx.gateway, ctx.snapshot)
    expect_equal(adapter.table_name, "analytics.subsystem_modules")


def test_subsystem_modules_adapter_load_raises(
    ctx: TestContext,
) -> None:
    """Load raises NotImplementedError (write-only adapter)."""
    adapter = SubsystemModulesAdapter(ctx.gateway, ctx.snapshot)
    with pytest.raises(NotImplementedError, match="does not support loading"):
        list(adapter.load())


def test_subsystem_modules_adapter_persist_empty(
    ctx: TestContext,
) -> None:
    """Persist empty list returns 0."""
    adapter = SubsystemModulesAdapter(ctx.gateway, ctx.snapshot)
    count = adapter.persist([])
    expect_equal(count, EXPECTED_COUNT_0)


def test_subsystem_modules_adapter_persist_single(
    ctx: TestContext,
) -> None:
    """Persist single subsystem-module mapping."""
    adapter = SubsystemModulesAdapter(ctx.gateway, ctx.snapshot)
    row = subsystem_module_payload(
        SubsystemModulePayloadSeed(
            subsystem_id="auth",
            module="auth.core",
            role="core",
            repo=DEMO_REPO,
            commit=DEMO_COMMIT,
        )
    )

    count = adapter.persist([row])
    expect_equal(count, EXPECTED_COUNT_1)

    total = count_rows(
        ctx.gateway.con,
        "SELECT COUNT(*) FROM analytics.subsystem_modules WHERE repo = ? AND commit = ?",
        [DEMO_REPO, DEMO_COMMIT],
    )
    expect_equal(total, EXPECTED_COUNT_1)


def test_subsystem_modules_adapter_persist_multiple(
    ctx: TestContext,
) -> None:
    """Persist multiple subsystem-module mappings."""
    adapter = SubsystemModulesAdapter(ctx.gateway, ctx.snapshot)

    rows = [
        subsystem_module_payload(
            SubsystemModulePayloadSeed(
                subsystem_id="auth",
                module="auth.core",
                role="core",
                repo=DEMO_REPO,
                commit=DEMO_COMMIT,
            )
        ),
        subsystem_module_payload(
            SubsystemModulePayloadSeed(
                subsystem_id="auth",
                module="auth.providers",
                role="adapter",
                repo=DEMO_REPO,
                commit=DEMO_COMMIT,
            )
        ),
        subsystem_module_payload(
            SubsystemModulePayloadSeed(
                subsystem_id="auth",
                module="auth.tokens",
                role="utility",
                repo=DEMO_REPO,
                commit=DEMO_COMMIT,
            )
        ),
    ]

    count = adapter.persist(rows)
    expect_equal(count, EXPECTED_COUNT_3)


def test_subsystem_modules_adapter_persist_verifies_data(
    ctx: TestContext,
) -> None:
    """Persisted data can be retrieved and verified."""
    adapter = SubsystemModulesAdapter(ctx.gateway, ctx.snapshot)
    row = subsystem_module_payload(
        SubsystemModulePayloadSeed(
            subsystem_id="users",
            module="users.service",
            role="service",
            repo=DEMO_REPO,
            commit=DEMO_COMMIT,
        )
    )
    adapter.persist([row])

    result = ctx.gateway.con.execute(
        """
        SELECT subsystem_id, module, role
        FROM analytics.subsystem_modules
        WHERE repo = ? AND commit = ?
        """,
        [DEMO_REPO, DEMO_COMMIT],
    ).fetchone()

    row = require_row(result, message="Expected subsystem module row")
    expect_equal(row[0], "users")
    expect_equal(row[1], "users.service")
    expect_equal(row[2], "service")
