"""Test profile adapter classes.

Test the profile-specific adapters for persisting function, file, and module
profile data using real DuckDB instances.
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

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
from tests._helpers.rows import file_profile_row, function_profile_row, module_profile_row

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
    row = function_profile_row(repo=DEMO_REPO, commit=DEMO_COMMIT)

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
    row = file_profile_row(repo=DEMO_REPO, commit=DEMO_COMMIT)

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
        file_profile_row(rel_path="src/api.py", repo=DEMO_REPO, commit=DEMO_COMMIT),
        file_profile_row(rel_path="src/db.py", repo=DEMO_REPO, commit=DEMO_COMMIT),
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
    row = module_profile_row(repo=DEMO_REPO, commit=DEMO_COMMIT)

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
        module_profile_row(module="services.api", repo=DEMO_REPO, commit=DEMO_COMMIT),
        module_profile_row(module="services.db", repo=DEMO_REPO, commit=DEMO_COMMIT),
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
