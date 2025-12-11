"""Test profile adapter classes.

Test the profile-specific adapters for persisting function, file, and module
profile data using real DuckDB instances.
"""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING

import pytest

from codeintel.analytics.adapters.profiles import (
    FileProfileAdapter,
    FunctionProfileAdapter,
    ModuleProfileAdapter,
)
from tests._helpers.assertions import expect_equal
from tests._helpers.context import create_test_context
from tests._helpers.contracts import count_rows
from tests._helpers.env_options import EnvOptions
from tests._helpers.rows import file_profile_row, function_profile_row, module_profile_row

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from codeintel.config.primitives import SnapshotRef
    from tests._helpers.context import TestContext

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
TEST_FILE_COUNT_5 = 5


# =============================================================================
# Test Data Factories
# =============================================================================


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def ctx(tmp_path: Path) -> Iterator[TestContext]:
    """Create a test context aligned with the demo repo/commit.

    Yields
    ------
    TestContext
        Context with schemas applied for profile adapter tests.
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
# FunctionProfileAdapter Tests
# =============================================================================


def test_function_profile_adapter_table_name(
    ctx: TestContext,
) -> None:
    """Adapter exposes correct table name."""
    adapter = FunctionProfileAdapter(ctx.gateway, ctx.snapshot)
    expect_equal(adapter.table_name, "analytics.function_profile")


def test_function_profile_adapter_load_raises(
    ctx: TestContext,
) -> None:
    """Load raises NotImplementedError (write-only adapter)."""
    adapter = FunctionProfileAdapter(ctx.gateway, ctx.snapshot)
    with pytest.raises(NotImplementedError, match="does not support loading"):
        list(adapter.load())


def test_function_profile_adapter_persist_empty(
    ctx: TestContext,
) -> None:
    """Persist empty list returns 0."""
    adapter = FunctionProfileAdapter(ctx.gateway, ctx.snapshot)
    count = adapter.persist([])
    expect_equal(count, EXPECTED_COUNT_0)


def test_function_profile_adapter_persist_single_row(
    ctx: TestContext,
) -> None:
    """Persist single row inserts to database."""
    adapter = FunctionProfileAdapter(ctx.gateway, ctx.snapshot)
    row = function_profile_row(repo=DEMO_REPO, commit=DEMO_COMMIT)

    count = adapter.persist([row])
    expect_equal(count, EXPECTED_COUNT_1)

    # Verify row was inserted
    total = count_rows(
        ctx.gateway.con,
        "SELECT COUNT(*) FROM analytics.function_profile WHERE repo = ? AND commit = ?",
        [DEMO_REPO, DEMO_COMMIT],
    )
    expect_equal(total, EXPECTED_COUNT_1)


def test_function_profile_adapter_persist_multiple_rows(
    ctx: TestContext,
) -> None:
    """Persist multiple rows inserts all to database."""
    adapter = FunctionProfileAdapter(ctx.gateway, ctx.snapshot)

    rows = [
        function_profile_row(
            goid=TEST_GOID_12345, qualname="module.func_a", repo=DEMO_REPO, commit=DEMO_COMMIT
        ),
        function_profile_row(
            goid=TEST_GOID_67890, qualname="module.func_b", repo=DEMO_REPO, commit=DEMO_COMMIT
        ),
    ]

    count = adapter.persist(rows)
    expect_equal(count, EXPECTED_COUNT_2)

    # Verify rows were inserted
    total = count_rows(
        ctx.gateway.con,
        "SELECT COUNT(*) FROM analytics.function_profile WHERE repo = ? AND commit = ?",
        [DEMO_REPO, DEMO_COMMIT],
    )
    expect_equal(total, EXPECTED_COUNT_2)


# =============================================================================
# FileProfileAdapter Tests
# =============================================================================


def test_file_profile_adapter_table_name(
    ctx: TestContext,
) -> None:
    """Adapter exposes correct table name."""
    adapter = FileProfileAdapter(ctx.gateway, ctx.snapshot)
    expect_equal(adapter.table_name, "analytics.file_profile")


def test_file_profile_adapter_load_raises(
    ctx: TestContext,
) -> None:
    """Load raises NotImplementedError (write-only adapter)."""
    adapter = FileProfileAdapter(ctx.gateway, ctx.snapshot)
    with pytest.raises(NotImplementedError, match="does not support loading"):
        list(adapter.load())


def test_file_profile_adapter_persist_empty(
    ctx: TestContext,
) -> None:
    """Persist empty list returns 0."""
    adapter = FileProfileAdapter(ctx.gateway, ctx.snapshot)
    count = adapter.persist([])
    expect_equal(count, EXPECTED_COUNT_0)


def test_file_profile_adapter_persist_single_row(
    ctx: TestContext,
) -> None:
    """Persist single row inserts to database."""
    adapter = FileProfileAdapter(ctx.gateway, ctx.snapshot)
    row = file_profile_row(repo=DEMO_REPO, commit=DEMO_COMMIT)

    count = adapter.persist([row])
    expect_equal(count, EXPECTED_COUNT_1)

    # Verify row was inserted
    total = count_rows(
        ctx.gateway.con,
        "SELECT COUNT(*) FROM analytics.file_profile WHERE repo = ? AND commit = ?",
        [DEMO_REPO, DEMO_COMMIT],
    )
    expect_equal(total, EXPECTED_COUNT_1)


def test_file_profile_adapter_persist_multiple_rows(
    ctx: TestContext,
) -> None:
    """Persist multiple rows inserts all to database."""
    adapter = FileProfileAdapter(ctx.gateway, ctx.snapshot)

    rows = [
        file_profile_row(rel_path="src/api.py", repo=DEMO_REPO, commit=DEMO_COMMIT),
        file_profile_row(rel_path="src/db.py", repo=DEMO_REPO, commit=DEMO_COMMIT),
    ]

    count = adapter.persist(rows)
    expect_equal(count, EXPECTED_COUNT_2)

    # Verify rows were inserted
    total = count_rows(
        ctx.gateway.con,
        "SELECT COUNT(*) FROM analytics.file_profile WHERE repo = ? AND commit = ?",
        [DEMO_REPO, DEMO_COMMIT],
    )
    expect_equal(total, EXPECTED_COUNT_2)


# =============================================================================
# ModuleProfileAdapter Tests
# =============================================================================


def test_module_profile_adapter_table_name(
    ctx: TestContext,
) -> None:
    """Adapter exposes correct table name."""
    adapter = ModuleProfileAdapter(ctx.gateway, ctx.snapshot)
    expect_equal(adapter.table_name, "analytics.module_profile")


def test_module_profile_adapter_load_raises(
    ctx: TestContext,
) -> None:
    """Load raises NotImplementedError (write-only adapter)."""
    adapter = ModuleProfileAdapter(ctx.gateway, ctx.snapshot)
    with pytest.raises(NotImplementedError, match="does not support loading"):
        list(adapter.load())


def test_module_profile_adapter_persist_empty(
    ctx: TestContext,
) -> None:
    """Persist empty list returns 0."""
    adapter = ModuleProfileAdapter(ctx.gateway, ctx.snapshot)
    count = adapter.persist([])
    expect_equal(count, EXPECTED_COUNT_0)


def test_module_profile_adapter_persist_single_row(
    ctx: TestContext,
) -> None:
    """Persist single row inserts to database."""
    adapter = ModuleProfileAdapter(ctx.gateway, ctx.snapshot)
    row = module_profile_row(repo=DEMO_REPO, commit=DEMO_COMMIT)

    count = adapter.persist([row])
    expect_equal(count, EXPECTED_COUNT_1)

    # Verify row was inserted
    total = count_rows(
        ctx.gateway.con,
        "SELECT COUNT(*) FROM analytics.module_profile WHERE repo = ? AND commit = ?",
        [DEMO_REPO, DEMO_COMMIT],
    )
    expect_equal(total, EXPECTED_COUNT_1)


def test_module_profile_adapter_persist_multiple_rows(
    ctx: TestContext,
) -> None:
    """Persist multiple rows inserts all to database."""
    adapter = ModuleProfileAdapter(ctx.gateway, ctx.snapshot)

    rows = [
        module_profile_row(module="services.api", repo=DEMO_REPO, commit=DEMO_COMMIT),
        module_profile_row(module="services.db", repo=DEMO_REPO, commit=DEMO_COMMIT),
    ]

    count = adapter.persist(rows)
    expect_equal(count, EXPECTED_COUNT_2)

    # Verify rows were inserted
    total = count_rows(
        ctx.gateway.con,
        "SELECT COUNT(*) FROM analytics.module_profile WHERE repo = ? AND commit = ?",
        [DEMO_REPO, DEMO_COMMIT],
    )
    expect_equal(total, EXPECTED_COUNT_2)
