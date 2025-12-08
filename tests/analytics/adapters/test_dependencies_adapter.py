"""Test dependency adapter classes.

Test the dependency-specific adapters for persisting external dependency
call and aggregate data using real DuckDB instances.
"""

from __future__ import annotations

from dataclasses import replace
from decimal import Decimal
from pathlib import Path
from typing import TypedDict, Unpack

import pytest

from codeintel.analytics.adapters.dependencies import (
    DependencyAggregateAdapter,
    DependencyAggregateRow,
    DependencyCallAdapter,
    DependencyCallRow,
    compute_dep_id,
    to_decimal,
)
from codeintel.config.primitives import SnapshotRef
from codeintel.storage.gateway import StorageGateway
from tests._helpers import assert_frozen
from tests._helpers.assertions import (
    expect_equal,
    expect_in,
    expect_is_instance,
    expect_is_none,
    expect_is_not_none,
    expect_length,
    expect_not_equal,
    expect_true,
)
from tests._helpers.contracts import count_rows
from tests._helpers.rows import (
    DependencyAggregatePayloadSeed,
    DependencyCallPayloadSeed,
    dependency_aggregate_payload,
    dependency_call_payload,
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
CALLSITE_COUNT_5 = 5
CALLSITE_COUNT_10 = 10
FUNCTION_COUNT_3 = 3
RISK_SCORE_0_75 = 0.75
CRITICALITY_0_5 = 0.5
DEP_ID_LENGTH = 16


# =============================================================================
# Test Data
# =============================================================================


class DependencyCallOverrides(TypedDict, total=False):
    """Optional overrides for dependency call payload seeds."""

    library: str
    service_name: str
    qualname: str
    rel_path: str
    module: str
    callsite_count: int
    function_goid: int
    modes: tuple[str, ...] | list[str]
    repo: str
    commit: str
    evidence_json: list[dict[str, object]] | None


class DependencyAggregateOverrides(TypedDict, total=False):
    """Optional overrides for dependency aggregate payload seeds."""

    library: str
    service_name: str
    category: str | None
    severity: str | None
    criticality: float | None
    risk_score: float | None
    function_count: int
    callsite_count: int
    risk_level: str
    repo: str
    commit: str
    modules_json: list[str] | None
    usage_modes: list[str] | None
    config_keys: list[str] | None


def _call_seed(
    *,
    goid: int | None = None,
    **overrides: Unpack[DependencyCallOverrides],
) -> DependencyCallPayloadSeed:
    base = DependencyCallPayloadSeed(
        library="requests",
        service_name="HTTP Client",
        qualname="api.fn",
        rel_path="src/services/api.py",
        module="services.api",
        callsite_count=1,
        function_goid=TEST_GOID_12345,
        modes=("read",),
        repo=DEMO_REPO,
        commit=DEMO_COMMIT,
        evidence_json=None,
    )
    merged_overrides: dict[str, object] = dict(overrides)
    if goid is not None:
        merged_overrides["function_goid"] = goid
    return replace(base, **merged_overrides)


def _aggregate_seed(
    **overrides: Unpack[DependencyAggregateOverrides],
) -> DependencyAggregatePayloadSeed:
    base = DependencyAggregatePayloadSeed(
        library="requests",
        service_name="HTTP Client",
        category="database",
        severity="medium",
        criticality=CRITICALITY_0_5,
        risk_score=RISK_SCORE_0_75,
        function_count=FUNCTION_COUNT_3,
        callsite_count=CALLSITE_COUNT_10,
        risk_level="moderate",
        repo=DEMO_REPO,
        commit=DEMO_COMMIT,
        modules_json=["services.api", "services.db"],
        usage_modes=["read", "write"],
        config_keys=["DATABASE_URL"],
    )
    return replace(base, **overrides)


def _dependency_call_row(seed: DependencyCallPayloadSeed) -> DependencyCallRow:
    payload = dependency_call_payload(seed)
    repo = str(payload["repo"])
    commit = str(payload["commit"])
    library = str(payload["library"])
    dep_id = compute_dep_id(repo, commit, library)
    return DependencyCallRow(
        repo=repo,
        commit=commit,
        dep_id=dep_id,
        library=library,
        service_name=str(payload["service_name"]),
        function_goid_h128=payload["function_goid_h128"],
        function_urn=str(payload["function_urn"]),
        rel_path=str(payload["rel_path"]),
        module=str(payload["module"]),
        qualname=str(payload["qualname"]),
        callsite_count=payload["callsite_count"],
        modes=payload["modes"],
        evidence_json=payload["evidence_json"],
        created_at=payload["created_at"],
    )


def _dependency_aggregate_row(seed: DependencyAggregatePayloadSeed) -> DependencyAggregateRow:
    payload = dependency_aggregate_payload(seed)
    repo = str(payload["repo"])
    commit = str(payload["commit"])
    library = str(payload["library"])
    dep_id = compute_dep_id(repo, commit, library)
    return DependencyAggregateRow(
        repo=repo,
        commit=commit,
        dep_id=dep_id,
        library=library,
        service_name=str(payload["service_name"]),
        category=payload["category"],
        language=str(payload["language"]),
        severity=payload["severity"],
        criticality=payload["criticality"],
        risk_score=payload["risk_score"],
        function_count=payload["function_count"],
        callsite_count=payload["callsite_count"],
        modules_json=payload["modules_json"],
        usage_modes=payload["usage_modes"],
        config_keys=payload["config_keys"],
        risk_level=str(payload["risk_level"]),
        created_at=payload["created_at"],
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
# Helper Function Tests
# =============================================================================


def test_compute_dep_id_deterministic() -> None:
    """Compute dependency ID is deterministic for same inputs."""
    dep_id_1 = compute_dep_id(DEMO_REPO, DEMO_COMMIT, "requests")
    dep_id_2 = compute_dep_id(DEMO_REPO, DEMO_COMMIT, "requests")
    expect_equal(dep_id_1, dep_id_2)


def test_compute_dep_id_length() -> None:
    """Compute dependency ID returns 16-character hex string."""
    dep_id = compute_dep_id(DEMO_REPO, DEMO_COMMIT, "sqlalchemy")
    expect_length(dep_id, DEP_ID_LENGTH)
    # Verify it's a valid hex string
    int(dep_id, 16)


def test_compute_dep_id_unique_per_library() -> None:
    """Compute dependency ID is unique per library."""
    dep_id_requests = compute_dep_id(DEMO_REPO, DEMO_COMMIT, "requests")
    dep_id_sqlalchemy = compute_dep_id(DEMO_REPO, DEMO_COMMIT, "sqlalchemy")
    expect_not_equal(dep_id_requests, dep_id_sqlalchemy)


def test_compute_dep_id_unique_per_repo() -> None:
    """Compute dependency ID is unique per repository."""
    dep_id_1 = compute_dep_id("repo/a", DEMO_COMMIT, "requests")
    dep_id_2 = compute_dep_id("repo/b", DEMO_COMMIT, "requests")
    expect_not_equal(dep_id_1, dep_id_2)


def test_compute_dep_id_unique_per_commit() -> None:
    """Compute dependency ID is unique per commit."""
    dep_id_1 = compute_dep_id(DEMO_REPO, "commit1", "requests")
    dep_id_2 = compute_dep_id(DEMO_REPO, "commit2", "requests")
    expect_not_equal(dep_id_1, dep_id_2)


def test_to_decimal_converts_int() -> None:
    """Convert integer to Decimal."""
    result = to_decimal(TEST_GOID_12345)
    expect_is_instance(result, Decimal)
    expect_equal(result, Decimal(TEST_GOID_12345))


def test_to_decimal_handles_zero() -> None:
    """Convert zero to Decimal."""
    result = to_decimal(0)
    expect_equal(result, Decimal(0))


def test_to_decimal_handles_large_int() -> None:
    """Convert large integer to Decimal (for hugeint support)."""
    large_int = 2**127 - 1
    result = to_decimal(large_int)
    expect_equal(result, Decimal(large_int))


# =============================================================================
# DependencyCallRow Tests
# =============================================================================


def test_dependency_call_row_creation() -> None:
    """Create DependencyCallRow with all required fields."""
    seed = _call_seed(
        library="requests",
        service_name="HTTP Client",
        qualname="services.api.fetch_data",
    )
    row = _dependency_call_row(seed)

    expect_equal(row.repo, DEMO_REPO)
    expect_equal(row.commit, DEMO_COMMIT)
    expect_equal(row.library, "requests")
    expect_equal(row.service_name, "HTTP Client")
    expect_equal(row.qualname, "services.api.fetch_data")


def test_dependency_call_row_goid_is_decimal() -> None:
    """DependencyCallRow goid is stored as Decimal."""
    seed = _call_seed(
        library="requests",
        service_name="HTTP Client",
        qualname="services.api.fetch_data",
        goid=TEST_GOID_67890,
    )
    row = _dependency_call_row(seed)

    expect_is_instance(row.function_goid_h128, Decimal)
    expect_equal(row.function_goid_h128, Decimal(TEST_GOID_67890))


def test_dependency_call_row_modes_is_list() -> None:
    """DependencyCallRow modes is a list of strings."""
    seed = _call_seed(
        library="redis",
        service_name="Redis Cache",
        qualname="cache.client.get",
        modes=("read", "write", "delete"),
    )
    row = _dependency_call_row(seed)

    expect_is_instance(row.modes, list)
    expect_in("read", row.modes)
    expect_in("write", row.modes)
    expect_in("delete", row.modes)


def test_dependency_call_row_is_frozen() -> None:
    """DependencyCallRow is immutable."""
    seed = _call_seed(
        library="requests",
        service_name="HTTP Client",
        qualname="api.fetch",
    )
    row = _dependency_call_row(seed)

    assert_frozen(row, "library", "other")


# =============================================================================
# DependencyAggregateRow Tests
# =============================================================================


def test_dependency_aggregate_row_creation() -> None:
    """Create DependencyAggregateRow with all required fields."""
    seed = _aggregate_seed(
        library="sqlalchemy",
        service_name="SQL Database",
        category="database",
        risk_level="high",
    )
    row = _dependency_aggregate_row(seed)

    expect_equal(row.repo, DEMO_REPO)
    expect_equal(row.commit, DEMO_COMMIT)
    expect_equal(row.library, "sqlalchemy")
    expect_equal(row.service_name, "SQL Database")
    expect_equal(row.category, "database")
    expect_equal(row.risk_level, "high")


def test_dependency_aggregate_row_optional_fields() -> None:
    """DependencyAggregateRow handles optional fields."""
    seed = _aggregate_seed(
        library="custom_lib",
        service_name="Custom Service",
        category=None,
        severity=None,
        criticality=None,
        risk_score=None,
    )
    row = _dependency_aggregate_row(seed)

    expect_is_none(row.category)
    expect_is_none(row.severity)
    expect_is_none(row.criticality)
    expect_is_none(row.risk_score)


def test_dependency_aggregate_row_numeric_fields() -> None:
    """DependencyAggregateRow numeric fields have correct types."""
    seed = _aggregate_seed(
        library="redis",
        service_name="Redis Cache",
        criticality=CRITICALITY_0_5,
        risk_score=RISK_SCORE_0_75,
        function_count=FUNCTION_COUNT_3,
        callsite_count=CALLSITE_COUNT_10,
    )
    row = _dependency_aggregate_row(seed)

    expect_equal(row.criticality, CRITICALITY_0_5)
    expect_equal(row.risk_score, RISK_SCORE_0_75)
    expect_equal(row.function_count, FUNCTION_COUNT_3)
    expect_equal(row.callsite_count, CALLSITE_COUNT_10)


def test_dependency_aggregate_row_list_fields() -> None:
    """DependencyAggregateRow list fields are populated."""
    seed = _aggregate_seed(
        library="requests",
        service_name="HTTP Client",
    )
    row = _dependency_aggregate_row(seed)

    expect_is_instance(row.modules_json, list)
    expect_is_instance(row.usage_modes, list)
    expect_is_instance(row.config_keys, list)


def test_dependency_aggregate_row_is_frozen() -> None:
    """DependencyAggregateRow is immutable."""
    seed = _aggregate_seed(
        library="requests",
        service_name="HTTP Client",
    )
    row = _dependency_aggregate_row(seed)

    assert_frozen(row, "library", "other")


# =============================================================================
# DependencyCallAdapter Tests
# =============================================================================


def test_call_adapter_table_name(
    fresh_gateway: StorageGateway,
    snapshot: SnapshotRef,
) -> None:
    """Adapter exposes correct table name."""
    adapter = DependencyCallAdapter(fresh_gateway, snapshot)
    expect_equal(adapter.table_name, "analytics.external_dependency_calls")


def test_call_adapter_load_returns_empty(
    fresh_gateway: StorageGateway,
    snapshot: SnapshotRef,
) -> None:
    """Load returns empty iterator (write-only adapter)."""
    adapter = DependencyCallAdapter(fresh_gateway, snapshot)
    rows = list(adapter.load())
    expect_true(not rows)


def test_call_adapter_persist_empty(
    fresh_gateway: StorageGateway,
    snapshot: SnapshotRef,
) -> None:
    """Persist empty list returns 0."""
    adapter = DependencyCallAdapter(fresh_gateway, snapshot)
    count = adapter.persist([])
    expect_equal(count, EXPECTED_COUNT_0)


def test_call_adapter_persist_single_row(
    fresh_gateway: StorageGateway,
    snapshot: SnapshotRef,
) -> None:
    """Persist single row inserts to database."""
    adapter = DependencyCallAdapter(fresh_gateway, snapshot)
    seed = _call_seed(
        library="requests",
        service_name="HTTP Client",
        qualname="api.fetch_data",
        callsite_count=CALLSITE_COUNT_5,
    )
    row = _dependency_call_row(seed)

    count = adapter.persist([row])
    expect_equal(count, EXPECTED_COUNT_1)

    # Verify row was inserted
    total = count_rows(
        fresh_gateway.con,
        "SELECT COUNT(*) FROM analytics.external_dependency_calls WHERE repo = ? AND commit = ?",
        [DEMO_REPO, DEMO_COMMIT],
    )
    expect_equal(total, EXPECTED_COUNT_1)


def test_call_adapter_persist_multiple_rows(
    fresh_gateway: StorageGateway,
    snapshot: SnapshotRef,
) -> None:
    """Persist multiple rows inserts all to database."""
    adapter = DependencyCallAdapter(fresh_gateway, snapshot)

    rows = [
        _dependency_call_row(
            _call_seed(
                library="requests",
                service_name="HTTP Client",
                qualname="api.get_user",
                goid=1001,
            )
        ),
        _dependency_call_row(
            _call_seed(
                library="requests",
                service_name="HTTP Client",
                qualname="api.get_orders",
                goid=1002,
            )
        ),
        _dependency_call_row(
            _call_seed(
                library="sqlalchemy",
                service_name="SQL Database",
                qualname="db.query_users",
                goid=1003,
            )
        ),
    ]

    count = adapter.persist(rows)
    expect_equal(count, EXPECTED_COUNT_3)

    # Verify rows were inserted
    total = count_rows(
        fresh_gateway.con,
        "SELECT COUNT(*) FROM analytics.external_dependency_calls WHERE repo = ? AND commit = ?",
        [DEMO_REPO, DEMO_COMMIT],
    )
    expect_equal(total, EXPECTED_COUNT_3)


def test_call_adapter_persist_verifies_data(
    fresh_gateway: StorageGateway,
    snapshot: SnapshotRef,
) -> None:
    """Persisted data can be retrieved and verified."""
    adapter = DependencyCallAdapter(fresh_gateway, snapshot)
    seed = _call_seed(
        library="redis",
        service_name="Redis Cache",
        qualname="cache.client.get_value",
        callsite_count=CALLSITE_COUNT_5,
        modes=("read",),
    )
    row = _dependency_call_row(seed)
    adapter.persist([row])

    # Query and verify
    result = fresh_gateway.con.execute(
        """
        SELECT library, service_name, qualname, callsite_count
        FROM analytics.external_dependency_calls
        WHERE repo = ? AND commit = ?
        """,
        [DEMO_REPO, DEMO_COMMIT],
    ).fetchone()

    row = expect_is_not_none(result)
    expect_equal(row[0], "redis")
    expect_equal(row[1], "Redis Cache")
    expect_equal(row[2], "cache.client.get_value")
    expect_equal(row[3], CALLSITE_COUNT_5)


# =============================================================================
# DependencyAggregateAdapter Tests
# =============================================================================


def test_aggregate_adapter_table_name(
    fresh_gateway: StorageGateway,
    snapshot: SnapshotRef,
) -> None:
    """Adapter exposes correct table name."""
    adapter = DependencyAggregateAdapter(fresh_gateway, snapshot)
    expect_equal(adapter.table_name, "analytics.external_dependencies")


def test_aggregate_adapter_load_returns_empty(
    fresh_gateway: StorageGateway,
    snapshot: SnapshotRef,
) -> None:
    """Load returns empty iterator (write-only adapter)."""
    adapter = DependencyAggregateAdapter(fresh_gateway, snapshot)
    rows = list(adapter.load())
    expect_true(not rows)


def test_aggregate_adapter_persist_empty(
    fresh_gateway: StorageGateway,
    snapshot: SnapshotRef,
) -> None:
    """Persist empty list returns 0."""
    adapter = DependencyAggregateAdapter(fresh_gateway, snapshot)
    count = adapter.persist([])
    expect_equal(count, EXPECTED_COUNT_0)


def test_aggregate_adapter_persist_single_row(
    fresh_gateway: StorageGateway,
    snapshot: SnapshotRef,
) -> None:
    """Persist single row inserts to database."""
    adapter = DependencyAggregateAdapter(fresh_gateway, snapshot)
    seed = _aggregate_seed(
        library="requests",
        service_name="HTTP Client",
        category="http",
        risk_level="low",
    )
    row = _dependency_aggregate_row(seed)

    count = adapter.persist([row])
    expect_equal(count, EXPECTED_COUNT_1)

    # Verify row was inserted
    total = count_rows(
        fresh_gateway.con,
        "SELECT COUNT(*) FROM analytics.external_dependencies WHERE repo = ? AND commit = ?",
        [DEMO_REPO, DEMO_COMMIT],
    )
    expect_equal(total, EXPECTED_COUNT_1)


def test_aggregate_adapter_persist_multiple_rows(
    fresh_gateway: StorageGateway,
    snapshot: SnapshotRef,
) -> None:
    """Persist multiple rows inserts all to database."""
    adapter = DependencyAggregateAdapter(fresh_gateway, snapshot)

    rows = [
        _dependency_aggregate_row(
            _aggregate_seed(
                library="requests",
                service_name="HTTP Client",
                category="http",
            )
        ),
        _dependency_aggregate_row(
            _aggregate_seed(
                library="sqlalchemy",
                service_name="SQL Database",
                category="database",
            )
        ),
    ]

    count = adapter.persist(rows)
    expect_equal(count, EXPECTED_COUNT_2)

    # Verify rows were inserted
    total = count_rows(
        fresh_gateway.con,
        "SELECT COUNT(*) FROM analytics.external_dependencies WHERE repo = ? AND commit = ?",
        [DEMO_REPO, DEMO_COMMIT],
    )
    expect_equal(total, EXPECTED_COUNT_2)


def test_aggregate_adapter_persist_verifies_data(
    fresh_gateway: StorageGateway,
    snapshot: SnapshotRef,
) -> None:
    """Persisted data can be retrieved and verified."""
    adapter = DependencyAggregateAdapter(fresh_gateway, snapshot)
    seed = _aggregate_seed(
        library="sqlalchemy",
        service_name="SQL Database",
        category="database",
        severity="high",
        risk_level="critical",
        function_count=FUNCTION_COUNT_3,
        callsite_count=CALLSITE_COUNT_10,
    )
    row = _dependency_aggregate_row(seed)
    adapter.persist([row])

    # Query and verify
    result = fresh_gateway.con.execute(
        """
        SELECT library, category, severity, risk_level, function_count, callsite_count
        FROM analytics.external_dependencies
        WHERE repo = ? AND commit = ?
        """,
        [DEMO_REPO, DEMO_COMMIT],
    ).fetchone()

    row = expect_is_not_none(result)
    expect_equal(row[0], "sqlalchemy")
    expect_equal(row[1], "database")
    expect_equal(row[2], "high")
    expect_equal(row[3], "critical")
    expect_equal(row[4], FUNCTION_COUNT_3)
    expect_equal(row[5], CALLSITE_COUNT_10)


def test_aggregate_adapter_persist_with_null_fields(
    fresh_gateway: StorageGateway,
    snapshot: SnapshotRef,
) -> None:
    """Persist row with null optional fields."""
    adapter = DependencyAggregateAdapter(fresh_gateway, snapshot)
    seed = _aggregate_seed(
        library="custom_lib",
        service_name="Custom Service",
        category=None,
        severity=None,
        criticality=None,
        risk_score=None,
    )
    row = _dependency_aggregate_row(seed)

    count = adapter.persist([row])
    expect_equal(count, EXPECTED_COUNT_1)

    # Verify null fields
    result = fresh_gateway.con.execute(
        """
        SELECT category, severity, criticality, risk_score
        FROM analytics.external_dependencies
        WHERE repo = ? AND commit = ?
        """,
        [DEMO_REPO, DEMO_COMMIT],
    ).fetchone()

    row = expect_is_not_none(result)
    expect_is_none(row[0])  # category
    expect_is_none(row[1])  # severity
    expect_is_none(row[2])  # criticality
    expect_is_none(row[3])  # risk_score
