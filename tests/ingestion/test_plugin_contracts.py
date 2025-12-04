"""Tests for plugin contract validation.

This module provides comprehensive tests for the contract validation
system used to verify plugin outputs meet data quality requirements.
"""

from __future__ import annotations

from pathlib import Path

from codeintel.config.primitives import SnapshotRef
from codeintel.ingestion.validation import (
    CONSTRAINT_CHECKERS,
    ColumnConstraint,
    ContractValidationResult,
    ContractViolation,
    ForeignKeyConstraint,
    ForeignKeyContractOptions,
    IngestContractSpec,
    IngestContractValidator,
    foreign_key_contract,
    not_null_contract,
    row_count_contract,
)
from codeintel.storage.gateway import StorageGateway
from tests._helpers.assertions import assert_cannot_setattr

# Test constants
TEST_REPO = "test/repo"
TEST_COMMIT = "abc123"
TEST_REPO_ROOT = Path("/opt/test")
EXPECTED_COUNT_2 = 2
EXPECTED_COUNT_3 = 3
MIN_ROW_THRESHOLD = 10
MAX_ROW_THRESHOLD = 100
MIN_AGE_VALUE = 18.0


# =============================================================================
# Dataclass Tests
# =============================================================================


def test_column_constraint_not_null() -> None:
    """ColumnConstraint should support not_null type."""
    c = ColumnConstraint(column="name", constraint_type="not_null")
    assert c.column == "name"
    assert c.constraint_type == "not_null"
    assert c.value is None


def test_column_constraint_min_value() -> None:
    """ColumnConstraint should support min_value type."""
    c = ColumnConstraint(column="age", constraint_type="min_value", value=MIN_AGE_VALUE)
    assert c.value == MIN_AGE_VALUE


def test_column_constraint_is_frozen() -> None:
    """ColumnConstraint should be immutable."""
    c = ColumnConstraint(column="x", constraint_type="not_null")
    assert_cannot_setattr(c, "column", "y")


def test_foreign_key_constraint_basic() -> None:
    """ForeignKeyConstraint should store FK relationship."""
    fk = ForeignKeyConstraint(
        column="parent_id",
        reference_table="core.parent",
        reference_column="id",
    )
    assert fk.column == "parent_id"
    assert fk.allow_null is True


def test_ingest_contract_spec_minimal() -> None:
    """IngestContractSpec should require only table."""
    spec = IngestContractSpec(table="core.test")
    assert spec.table == "core.test"
    assert spec.min_rows is None
    assert spec.severity == "error"


def test_contract_violation_basic() -> None:
    """ContractViolation should store violation info."""
    contract = IngestContractSpec(table="core.test")
    v = ContractViolation(contract=contract, message="Failed", severity="error")
    assert v.message == "Failed"
    assert v.details == {}


def test_contract_validation_result_success() -> None:
    """ContractValidationResult.success should create valid result."""
    r = ContractValidationResult.success(checked=EXPECTED_COUNT_3)
    assert r.valid is True
    assert r.checked_contracts == EXPECTED_COUNT_3


def test_contract_validation_result_failure() -> None:
    """ContractValidationResult.failure should create failed result."""
    contract = IngestContractSpec(table="core.test")
    violations = [ContractViolation(contract=contract, message="E", severity="error")]
    r = ContractValidationResult.failure(violations, checked=2)
    assert r.valid is False
    assert len(r.violations) == 1


def test_constraint_checkers_registry() -> None:
    """CONSTRAINT_CHECKERS should have expected types."""
    expected = {"not_null", "min_value", "max_value", "positive", "unique", "min_fraction_not_null"}
    assert set(CONSTRAINT_CHECKERS.keys()) == expected


# =============================================================================
# Validator Tests
# =============================================================================


def test_validator_table_not_exists(fresh_gateway: StorageGateway) -> None:
    """Validator should report violation for missing table."""
    snapshot = SnapshotRef(repo=TEST_REPO, commit=TEST_COMMIT, repo_root=TEST_REPO_ROOT)
    validator = IngestContractValidator(fresh_gateway)
    contract = IngestContractSpec(table="nonexistent.table")
    result = validator.validate([contract], snapshot)
    assert result.valid is False
    assert "does not exist" in result.violations[0].message


def test_validator_min_rows_violation(fresh_gateway: StorageGateway) -> None:
    """Validator should report when row count is below minimum."""
    snapshot = SnapshotRef(repo=TEST_REPO, commit=TEST_COMMIT, repo_root=TEST_REPO_ROOT)
    validator = IngestContractValidator(fresh_gateway)
    contract = IngestContractSpec(table="core.modules", min_rows=MIN_ROW_THRESHOLD)
    result = validator.validate([contract], snapshot)
    assert result.valid is False


def test_validator_skip_if_empty(fresh_gateway: StorageGateway) -> None:
    """Validator should skip validation for empty tables if configured."""
    snapshot = SnapshotRef(repo=TEST_REPO, commit=TEST_COMMIT, repo_root=TEST_REPO_ROOT)
    validator = IngestContractValidator(fresh_gateway)
    contract = IngestContractSpec(table="core.modules", min_rows=10, skip_if_empty=True)
    result = validator.validate([contract], snapshot)
    assert result.valid is True


def test_validator_required_columns_missing(fresh_gateway: StorageGateway) -> None:
    """Validator should report missing required columns."""
    # Insert data so skip_if_empty doesn't short-circuit
    fresh_gateway.con.execute("""
        INSERT INTO core.modules (module, path, repo, commit, language, tags, owners)
        VALUES ('test', 'test.py', 'test/repo', 'abc123', 'python', '[]', '[]')
    """)
    snapshot = SnapshotRef(repo=TEST_REPO, commit=TEST_COMMIT, repo_root=TEST_REPO_ROOT)
    validator = IngestContractValidator(fresh_gateway)
    contract = IngestContractSpec(
        table="core.modules",
        required_columns=("nonexistent_col",),
    )
    result = validator.validate([contract], snapshot)
    assert result.valid is False


def test_validator_not_null_constraint_pass(fresh_gateway: StorageGateway) -> None:
    """Validator should pass not_null when no nulls exist."""
    fresh_gateway.con.execute("""
        INSERT INTO core.modules (module, path, repo, commit, language, tags, owners)
        VALUES ('test', 'test.py', 'r', 'c', 'python', '[]', '[]')
    """)
    snapshot = SnapshotRef(repo=TEST_REPO, commit=TEST_COMMIT, repo_root=TEST_REPO_ROOT)
    validator = IngestContractValidator(fresh_gateway)
    contract = IngestContractSpec(
        table="core.modules",
        column_constraints=(ColumnConstraint(column="module", constraint_type="not_null"),),
    )
    result = validator.validate([contract], snapshot)
    assert result.valid is True


def test_validator_not_null_constraint_fail(fresh_gateway: StorageGateway) -> None:
    """Validator should fail not_null when nulls exist."""
    fresh_gateway.con.execute("CREATE TABLE IF NOT EXISTS core.test_n (id INT, name VARCHAR)")
    fresh_gateway.con.execute("INSERT INTO core.test_n (id, name) VALUES (1, NULL)")
    snapshot = SnapshotRef(repo=TEST_REPO, commit=TEST_COMMIT, repo_root=TEST_REPO_ROOT)
    validator = IngestContractValidator(fresh_gateway)
    contract = IngestContractSpec(
        table="core.test_n",
        column_constraints=(ColumnConstraint(column="name", constraint_type="not_null"),),
    )
    result = validator.validate([contract], snapshot)
    assert result.valid is False


def test_validator_min_value_fail(fresh_gateway: StorageGateway) -> None:
    """Validator should fail min_value when values below minimum."""
    fresh_gateway.con.execute("CREATE TABLE IF NOT EXISTS core.test_min (id INT, val DOUBLE)")
    fresh_gateway.con.execute("INSERT INTO core.test_min VALUES (1, -5.0)")
    snapshot = SnapshotRef(repo=TEST_REPO, commit=TEST_COMMIT, repo_root=TEST_REPO_ROOT)
    validator = IngestContractValidator(fresh_gateway)
    contract = IngestContractSpec(
        table="core.test_min",
        column_constraints=(
            ColumnConstraint(column="val", constraint_type="min_value", value=0.0),
        ),
    )
    result = validator.validate([contract], snapshot)
    assert result.valid is False


def test_validator_max_value_fail(fresh_gateway: StorageGateway) -> None:
    """Validator should fail max_value when values exceed maximum."""
    fresh_gateway.con.execute("CREATE TABLE IF NOT EXISTS core.test_max (id INT, val DOUBLE)")
    fresh_gateway.con.execute("INSERT INTO core.test_max VALUES (1, 150.0)")
    snapshot = SnapshotRef(repo=TEST_REPO, commit=TEST_COMMIT, repo_root=TEST_REPO_ROOT)
    validator = IngestContractValidator(fresh_gateway)
    contract = IngestContractSpec(
        table="core.test_max",
        column_constraints=(
            ColumnConstraint(column="val", constraint_type="max_value", value=100.0),
        ),
    )
    result = validator.validate([contract], snapshot)
    assert result.valid is False


def test_validator_positive_fail(fresh_gateway: StorageGateway) -> None:
    """Validator should fail positive when non-positive values exist."""
    fresh_gateway.con.execute("CREATE TABLE IF NOT EXISTS core.test_pos (id INT, val INT)")
    fresh_gateway.con.execute("INSERT INTO core.test_pos VALUES (1, -5), (2, 0)")
    snapshot = SnapshotRef(repo=TEST_REPO, commit=TEST_COMMIT, repo_root=TEST_REPO_ROOT)
    validator = IngestContractValidator(fresh_gateway)
    contract = IngestContractSpec(
        table="core.test_pos",
        column_constraints=(ColumnConstraint(column="val", constraint_type="positive"),),
    )
    result = validator.validate([contract], snapshot)
    assert result.valid is False


def test_validator_unique_fail(fresh_gateway: StorageGateway) -> None:
    """Validator should fail unique when duplicates exist."""
    fresh_gateway.con.execute("CREATE TABLE IF NOT EXISTS core.test_uniq (id INT, name VARCHAR)")
    fresh_gateway.con.execute("INSERT INTO core.test_uniq VALUES (1, 'a'), (2, 'a')")
    snapshot = SnapshotRef(repo=TEST_REPO, commit=TEST_COMMIT, repo_root=TEST_REPO_ROOT)
    validator = IngestContractValidator(fresh_gateway)
    contract = IngestContractSpec(
        table="core.test_uniq",
        column_constraints=(ColumnConstraint(column="name", constraint_type="unique"),),
    )
    result = validator.validate([contract], snapshot)
    assert result.valid is False


def test_validator_min_fraction_not_null_fail(fresh_gateway: StorageGateway) -> None:
    """Validator should fail min_fraction_not_null when fraction not met."""
    fresh_gateway.con.execute("CREATE TABLE IF NOT EXISTS core.test_frac (id INT, val VARCHAR)")
    fresh_gateway.con.execute(
        "INSERT INTO core.test_frac VALUES (1, 'a'), (2, NULL), (3, NULL), (4, NULL)"
    )
    snapshot = SnapshotRef(repo=TEST_REPO, commit=TEST_COMMIT, repo_root=TEST_REPO_ROOT)
    validator = IngestContractValidator(fresh_gateway)
    contract = IngestContractSpec(
        table="core.test_frac",
        column_constraints=(
            ColumnConstraint(column="val", constraint_type="min_fraction_not_null", value=0.9),
        ),
    )
    result = validator.validate([contract], snapshot)
    assert result.valid is False


def test_validator_foreign_key_pass(fresh_gateway: StorageGateway) -> None:
    """Validator should pass FK constraint when all refs valid."""
    fresh_gateway.con.execute(
        "CREATE TABLE IF NOT EXISTS core.fk_p (id INT PRIMARY KEY, name VARCHAR)"
    )
    fresh_gateway.con.execute("CREATE TABLE IF NOT EXISTS core.fk_c (id INT, pid INT)")
    fresh_gateway.con.execute("INSERT INTO core.fk_p VALUES (1, 'a'), (2, 'b')")
    fresh_gateway.con.execute("INSERT INTO core.fk_c VALUES (1, 1), (2, 2)")
    snapshot = SnapshotRef(repo=TEST_REPO, commit=TEST_COMMIT, repo_root=TEST_REPO_ROOT)
    validator = IngestContractValidator(fresh_gateway)
    contract = IngestContractSpec(
        table="core.fk_c",
        foreign_keys=(
            ForeignKeyConstraint(column="pid", reference_table="core.fk_p", reference_column="id"),
        ),
    )
    result = validator.validate([contract], snapshot)
    assert result.valid is True


def test_validator_foreign_key_fail(fresh_gateway: StorageGateway) -> None:
    """Validator should fail FK constraint when orphans exist."""
    fresh_gateway.con.execute(
        "CREATE TABLE IF NOT EXISTS core.fk_p2 (id INT PRIMARY KEY, name VARCHAR)"
    )
    fresh_gateway.con.execute("CREATE TABLE IF NOT EXISTS core.fk_c2 (id INT, pid INT)")
    fresh_gateway.con.execute("INSERT INTO core.fk_p2 VALUES (1, 'a')")
    fresh_gateway.con.execute("INSERT INTO core.fk_c2 VALUES (1, 1), (2, 999)")
    snapshot = SnapshotRef(repo=TEST_REPO, commit=TEST_COMMIT, repo_root=TEST_REPO_ROOT)
    validator = IngestContractValidator(fresh_gateway)
    contract = IngestContractSpec(
        table="core.fk_c2",
        foreign_keys=(
            ForeignKeyConstraint(column="pid", reference_table="core.fk_p2", reference_column="id"),
        ),
    )
    result = validator.validate([contract], snapshot)
    assert result.valid is False


def test_validator_severity_warning_does_not_cause_failure(
    fresh_gateway: StorageGateway,
) -> None:
    """Validator should not fail when severity is warning."""
    # Create table with NULL values that would normally fail not_null
    fresh_gateway.con.execute("CREATE TABLE IF NOT EXISTS core.warn_test (id INT, name VARCHAR)")
    fresh_gateway.con.execute("INSERT INTO core.warn_test VALUES (1, NULL), (2, 'a')")

    snapshot = SnapshotRef(repo=TEST_REPO, commit=TEST_COMMIT, repo_root=TEST_REPO_ROOT)
    validator = IngestContractValidator(fresh_gateway)
    # Same constraint with severity="warning" should NOT fail
    contract = IngestContractSpec(
        table="core.warn_test",
        column_constraints=(ColumnConstraint(column="name", constraint_type="not_null"),),
        severity="warning",
    )
    result = validator.validate([contract], snapshot)
    # With warning severity, validation should pass even with violations
    assert result.valid is True


def test_validator_multiple_contracts(fresh_gateway: StorageGateway) -> None:
    """Validator should validate multiple contracts."""
    snapshot = SnapshotRef(repo=TEST_REPO, commit=TEST_COMMIT, repo_root=TEST_REPO_ROOT)
    validator = IngestContractValidator(fresh_gateway)
    contracts = [
        IngestContractSpec(table="core.modules", skip_if_empty=True),
        IngestContractSpec(table="core.goids", skip_if_empty=True),
    ]
    result = validator.validate(contracts, snapshot)
    assert result.valid is True
    assert result.checked_contracts == EXPECTED_COUNT_2


# =============================================================================
# Builder Function Tests
# =============================================================================


def test_row_count_contract_builder() -> None:
    """row_count_contract should create row count contract."""
    c = row_count_contract("core.test", min_rows=1, max_rows=100, plugin_name="p")
    assert c.table == "core.test"
    assert c.min_rows == 1
    assert c.max_rows == MAX_ROW_THRESHOLD


def test_not_null_contract_builder() -> None:
    """not_null_contract should create not-null constraints."""
    c = not_null_contract("core.test", ["col1", "col2"])
    assert len(c.column_constraints) == EXPECTED_COUNT_2
    assert all(x.constraint_type == "not_null" for x in c.column_constraints)


def test_foreign_key_contract_builder() -> None:
    """foreign_key_contract should create FK contract."""
    c = foreign_key_contract("core.child", "parent_id", "core.parent", "id")
    assert len(c.foreign_keys) == 1
    assert c.foreign_keys[0].column == "parent_id"


def test_foreign_key_contract_with_options() -> None:
    """foreign_key_contract should accept ForeignKeyContractOptions."""
    options = ForeignKeyContractOptions(
        allow_null=False, plugin_name="fk_plugin", severity="warning"
    )
    c = foreign_key_contract("core.child", "pid", "core.parent", "id", options=options)
    assert c.plugin_name == "fk_plugin"
    assert c.foreign_keys[0].allow_null is False
