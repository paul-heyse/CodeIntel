"""Test base adapter classes and protocols.

Test the foundational adapter abstractions using real DuckDB instances.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from codeintel.analytics.adapters.base import (
    AnalyticsAdapter,
    BatchAdapter,
    DeleteScope,
    SimpleBatchAdapter,
)
from codeintel.config.primitives import SnapshotRef
from tests._helpers import assert_frozen
from tests._helpers.assertions import (
    expect_equal,
    expect_is_none,
    expect_is_not_none,
    expect_length,
    expect_true,
)

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from codeintel.storage.gateway import StorageGateway

# =============================================================================
# Constants
# =============================================================================

DEMO_REPO = "demo/repo"
DEMO_COMMIT = "abc123"
EXPECTED_COUNT_0 = 0
EXPECTED_COUNT_1 = 1
EXPECTED_COUNT_2 = 2
EXPECTED_COUNT_3 = 3


@dataclass(frozen=True)
class SampleRow:
    """Sample row for adapter tests."""

    repo: str
    commit: str
    value: int
    name: str


# =============================================================================
# Concrete Adapter Implementations for Testing
# =============================================================================


class ConcreteAnalyticsAdapter(AnalyticsAdapter[SampleRow]):
    """Concrete implementation of AnalyticsAdapter for testing."""

    def load(self) -> Iterator[SampleRow]:
        """
        Load sample rows from database.

        Yields
        ------
        SampleRow
            Sample rows from the database.
        """
        query = """
            SELECT repo, commit, value, name
            FROM sample_analytics
            WHERE repo = ? AND commit = ?
        """
        result = self._gateway.con.execute(
            query,
            [self._snapshot.repo, self._snapshot.commit],
        )
        for row in result.fetchall():
            yield SampleRow(
                repo=str(row[0]),
                commit=str(row[1]),
                value=int(row[2]),
                name=str(row[3]),
            )

    def persist(self, rows: Sequence[SampleRow]) -> int:
        """
        Persist sample rows to database.

        Parameters
        ----------
        rows
            Rows to persist.

        Returns
        -------
        int
            Number of rows persisted.
        """
        if not rows:
            return EXPECTED_COUNT_0

        # Delete existing rows first
        delete_query = """
            DELETE FROM sample_analytics
            WHERE repo = ? AND commit = ?
        """
        self._gateway.con.execute(
            delete_query,
            [self._snapshot.repo, self._snapshot.commit],
        )

        # Insert new rows
        for row in rows:
            insert_query = """
                INSERT INTO sample_analytics (repo, commit, value, name)
                VALUES (?, ?, ?, ?)
            """
            self._gateway.con.execute(
                insert_query,
                [row.repo, row.commit, row.value, row.name],
            )

        return len(rows)


class ConcreteBatchAdapter(BatchAdapter[SampleRow]):
    """Concrete implementation of BatchAdapter for testing."""

    @property
    def table_name(self) -> str:
        """Return the target table name."""
        return "sample_batch"

    def load(self) -> Iterator[SampleRow]:
        """
        Load sample rows from database.

        Yields
        ------
        SampleRow
            Sample rows from the database.
        """
        query = """
            SELECT repo, commit, value, name
            FROM sample_batch
            WHERE repo = ? AND commit = ?
        """
        result = self._gateway.con.execute(
            query,
            [self._snapshot.repo, self._snapshot.commit],
        )
        for row in result.fetchall():
            yield SampleRow(
                repo=str(row[0]),
                commit=str(row[1]),
                value=int(row[2]),
                name=str(row[3]),
            )

    def persist(self, rows: Sequence[SampleRow]) -> int:
        """
        Persist sample rows to database.

        Parameters
        ----------
        rows
            Rows to persist.

        Returns
        -------
        int
            Number of rows persisted.
        """
        if not rows:
            return EXPECTED_COUNT_0

        for row in rows:
            insert_query = """
                INSERT INTO sample_batch (repo, commit, value, name)
                VALUES (?, ?, ?, ?)
            """
            self._gateway.con.execute(
                insert_query,
                [row.repo, row.commit, row.value, row.name],
            )

        return len(rows)


class ConcreteSimpleBatchAdapter(SimpleBatchAdapter[SampleRow]):
    """Concrete implementation of SimpleBatchAdapter for testing."""

    @property
    def table_name(self) -> str:
        """Return the target table name."""
        return "sample_simple_batch"

    def insert_rows(
        self,
        gateway: StorageGateway,
        rows: Sequence[SampleRow],
    ) -> int:
        """
        Insert sample rows into database.

        Parameters
        ----------
        gateway
            Storage gateway for database access.
        rows
            Rows to insert.

        Returns
        -------
        int
            Number of rows inserted.
        """
        # Access self.table_name to satisfy PLR6301 (method uses self)
        _ = self.table_name
        # Record that a gateway was provided to ensure interface compliance
        expect_is_not_none(gateway)
        return len(rows)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def gateway_with_tables(fresh_gateway: StorageGateway) -> StorageGateway:
    """
    Create gateway with sample tables.

    Parameters
    ----------
    fresh_gateway
        Base gateway from main conftest.

    Returns
    -------
    StorageGateway
        Gateway with sample tables created.
    """
    # Create sample tables on top of the fresh gateway
    fresh_gateway.con.execute("""
        CREATE TABLE IF NOT EXISTS sample_analytics (
            repo VARCHAR,
            commit VARCHAR,
            value INTEGER,
            name VARCHAR
        )
    """)
    fresh_gateway.con.execute("""
        CREATE TABLE IF NOT EXISTS sample_batch (
            repo VARCHAR,
            commit VARCHAR,
            value INTEGER,
            name VARCHAR
        )
    """)
    fresh_gateway.con.execute("""
        CREATE TABLE IF NOT EXISTS sample_simple_batch (
            repo VARCHAR,
            commit VARCHAR,
            value INTEGER,
            name VARCHAR
        )
    """)
    return fresh_gateway


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
        repo_root=Path.cwd(),
    )


# =============================================================================
# DeleteScope Tests
# =============================================================================


def test_create_delete_scope() -> None:
    """Create delete scope with repo and commit."""
    scope = DeleteScope(repo=DEMO_REPO, commit=DEMO_COMMIT)
    expect_equal(scope.repo, DEMO_REPO)
    expect_equal(scope.commit, DEMO_COMMIT)


def test_delete_scope_defaults() -> None:
    """Delete scope has None defaults for optional fields."""
    scope = DeleteScope(repo=DEMO_REPO, commit=DEMO_COMMIT)
    expect_is_none(scope.columns)


def test_delete_scope_with_columns() -> None:
    """Create delete scope with custom columns."""
    scope = DeleteScope(
        repo=DEMO_REPO,
        commit=DEMO_COMMIT,
        columns=("repo", "commit", "version"),
    )
    expect_is_not_none(scope.columns)
    expect_length(scope.columns or (), EXPECTED_COUNT_3)


def test_delete_scope_is_frozen() -> None:
    """Delete scope is immutable."""
    scope = DeleteScope(repo=DEMO_REPO, commit=DEMO_COMMIT)
    assert_frozen(scope, "repo", "other")


# =============================================================================
# AnalyticsAdapter Tests
# =============================================================================


def test_adapter_properties(
    gateway_with_tables: StorageGateway,
    snapshot: SnapshotRef,
) -> None:
    """Adapter exposes gateway, snapshot, repo, commit properties."""
    adapter = ConcreteAnalyticsAdapter(gateway_with_tables, snapshot)

    expect_equal(adapter.gateway, gateway_with_tables)
    expect_equal(adapter.snapshot, snapshot)
    expect_equal(adapter.repo, DEMO_REPO)
    expect_equal(adapter.commit, DEMO_COMMIT)


def test_adapter_load_empty(
    gateway_with_tables: StorageGateway,
    snapshot: SnapshotRef,
) -> None:
    """Load from empty table returns no rows."""
    adapter = ConcreteAnalyticsAdapter(gateway_with_tables, snapshot)
    rows = list(adapter.load())
    expect_true(not rows)


def test_adapter_persist_and_load(
    gateway_with_tables: StorageGateway,
    snapshot: SnapshotRef,
) -> None:
    """Persist rows and load them back."""
    adapter = ConcreteAnalyticsAdapter(gateway_with_tables, snapshot)

    # Persist rows
    sample_rows = [
        SampleRow(repo=DEMO_REPO, commit=DEMO_COMMIT, value=1, name="first"),
        SampleRow(repo=DEMO_REPO, commit=DEMO_COMMIT, value=2, name="second"),
    ]
    count = adapter.persist(sample_rows)
    expect_equal(count, EXPECTED_COUNT_2)

    # Load back
    loaded = list(adapter.load())
    expect_length(loaded, EXPECTED_COUNT_2)


def test_adapter_persist_empty(
    gateway_with_tables: StorageGateway,
    snapshot: SnapshotRef,
) -> None:
    """Persist empty list returns 0."""
    adapter = ConcreteAnalyticsAdapter(gateway_with_tables, snapshot)
    count = adapter.persist([])
    expect_equal(count, EXPECTED_COUNT_0)


def test_adapter_persist_replaces_existing(
    gateway_with_tables: StorageGateway,
    snapshot: SnapshotRef,
) -> None:
    """Persist replaces existing rows for same repo/commit."""
    adapter = ConcreteAnalyticsAdapter(gateway_with_tables, snapshot)

    # First persist
    adapter.persist([SampleRow(DEMO_REPO, DEMO_COMMIT, 1, "original")])

    # Second persist should replace
    adapter.persist([SampleRow(DEMO_REPO, DEMO_COMMIT, 2, "replaced")])

    loaded = list(adapter.load())
    expect_length(loaded, EXPECTED_COUNT_1)
    expect_equal(loaded[0].name, "replaced")


# =============================================================================
# BatchAdapter Tests
# =============================================================================


def test_batch_table_name_property(
    gateway_with_tables: StorageGateway,
    snapshot: SnapshotRef,
) -> None:
    """Batch adapter exposes table_name property."""
    adapter = ConcreteBatchAdapter(gateway_with_tables, snapshot)
    expect_equal(adapter.table_name, "sample_batch")


def test_batch_delete_scope_default(
    gateway_with_tables: StorageGateway,
    snapshot: SnapshotRef,
) -> None:
    """Default delete scope uses repo/commit."""
    adapter = ConcreteBatchAdapter(gateway_with_tables, snapshot)
    scope = adapter.delete_scope()
    expect_equal(scope.repo, DEMO_REPO)
    expect_equal(scope.commit, DEMO_COMMIT)


def test_batch_persist_with_delete(
    gateway_with_tables: StorageGateway,
    snapshot: SnapshotRef,
) -> None:
    """Persist batch deletes existing rows first."""
    adapter = ConcreteBatchAdapter(gateway_with_tables, snapshot)

    # First persist
    sample_rows = [SampleRow(DEMO_REPO, DEMO_COMMIT, 1, "first")]
    adapter.persist_batch(sample_rows, delete_before=True)

    # Second persist with delete
    new_rows = [SampleRow(DEMO_REPO, DEMO_COMMIT, 2, "second")]
    adapter.persist_batch(new_rows, delete_before=True)

    loaded = list(adapter.load())
    expect_length(loaded, EXPECTED_COUNT_1)
    expect_equal(loaded[0].name, "second")


def test_batch_persist_without_delete(
    gateway_with_tables: StorageGateway,
    snapshot: SnapshotRef,
) -> None:
    """Persist batch without delete appends rows."""
    adapter = ConcreteBatchAdapter(gateway_with_tables, snapshot)

    # First persist
    adapter.persist_batch(
        [SampleRow(DEMO_REPO, DEMO_COMMIT, 1, "first")],
        delete_before=False,
    )

    # Second persist without delete
    adapter.persist_batch(
        [SampleRow(DEMO_REPO, DEMO_COMMIT, 2, "second")],
        delete_before=False,
    )

    loaded = list(adapter.load())
    expect_length(loaded, EXPECTED_COUNT_2)


def test_batch_persist_empty(
    gateway_with_tables: StorageGateway,
    snapshot: SnapshotRef,
) -> None:
    """Persist empty batch returns 0."""
    adapter = ConcreteBatchAdapter(gateway_with_tables, snapshot)
    count = adapter.persist_batch([])
    expect_equal(count, EXPECTED_COUNT_0)


# =============================================================================
# SimpleBatchAdapter Tests
# =============================================================================


def test_simple_batch_table_name_property() -> None:
    """Simple batch adapter exposes table_name property."""
    adapter = ConcreteSimpleBatchAdapter()
    expect_equal(adapter.table_name, "sample_simple_batch")


def test_simple_batch_insert_rows(
    gateway_with_tables: StorageGateway,
) -> None:
    """Insert rows returns count."""
    adapter = ConcreteSimpleBatchAdapter()
    rows = [
        SampleRow(DEMO_REPO, DEMO_COMMIT, 1, "a"),
        SampleRow(DEMO_REPO, DEMO_COMMIT, 2, "b"),
    ]
    # Note: Our test impl just returns len(rows)
    count = adapter.insert_rows(gateway_with_tables, rows)
    expect_equal(count, EXPECTED_COUNT_2)


def test_simple_batch_execute_delete(
    gateway_with_tables: StorageGateway,
) -> None:
    """Execute delete removes matching rows."""
    # Insert some rows
    gateway_with_tables.con.execute(
        """
        INSERT INTO sample_simple_batch (repo, commit, value, name)
        VALUES (?, ?, ?, ?)
        """,
        [DEMO_REPO, DEMO_COMMIT, 1, "test"],
    )

    adapter = ConcreteSimpleBatchAdapter()
    scope = DeleteScope(repo=DEMO_REPO, commit=DEMO_COMMIT)
    deleted = adapter.execute_delete(gateway_with_tables, scope)

    expect_equal(deleted, EXPECTED_COUNT_1)

    # Verify row is gone
    result = gateway_with_tables.con.execute(
        "SELECT COUNT(*) FROM sample_simple_batch WHERE repo = ? AND commit = ?",
        [DEMO_REPO, DEMO_COMMIT],
    ).fetchone()
    row = expect_is_not_none(result)
    expect_equal(row[0], EXPECTED_COUNT_0)


def test_simple_batch_execute_delete_no_rows(
    gateway_with_tables: StorageGateway,
) -> None:
    """Execute delete returns 0 when no matching rows."""
    adapter = ConcreteSimpleBatchAdapter()
    scope = DeleteScope(repo="nonexistent", commit="none")
    deleted = adapter.execute_delete(gateway_with_tables, scope)

    expect_equal(deleted, EXPECTED_COUNT_0)
