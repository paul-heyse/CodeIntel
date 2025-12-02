"""Contract validation for ingestion plugin outputs.

This module provides a contract system for validating plugin outputs,
ensuring data quality and consistency across the ingestion pipeline.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from codeintel.config.primitives import SnapshotRef
    from codeintel.storage.gateway import StorageGateway

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ColumnConstraint:
    """Constraint on a single column.

    Attributes
    ----------
    column
        Column name.
    constraint_type
        Type of constraint to apply.
    value
        Constraint value (interpretation depends on type).
    """

    column: str
    constraint_type: Literal[
        "not_null",
        "min_value",
        "max_value",
        "in_set",
        "regex",
        "min_fraction_not_null",
        "unique",
        "positive",
    ]
    value: float | str | Sequence[str] | None = None


@dataclass(frozen=True)
class ForeignKeyConstraint:
    """Foreign key integrity constraint.

    Attributes
    ----------
    column
        Column in the source table.
    reference_table
        Target table to reference.
    reference_column
        Column in the target table.
    allow_null
        Whether NULL values are allowed.
    """

    column: str
    reference_table: str
    reference_column: str
    allow_null: bool = True


@dataclass(frozen=True)
class IngestContractSpec:
    """Specification for plugin output validation.

    Attributes
    ----------
    table
        Table key to validate (e.g., "core.ast_nodes").
    plugin_name
        Name of the plugin producing this output.
    min_rows
        Minimum expected row count.
    max_rows
        Maximum expected row count.
    required_columns
        Columns that must exist.
    column_constraints
        Value constraints on columns.
    foreign_keys
        Foreign key integrity constraints.
    description
        Human-readable description.
    severity
        How violations should be handled.
    skip_if_empty
        Skip validation if source data is empty.
    """

    table: str
    plugin_name: str = ""
    min_rows: int | None = None
    max_rows: int | None = None
    required_columns: tuple[str, ...] = ()
    column_constraints: tuple[ColumnConstraint, ...] = ()
    foreign_keys: tuple[ForeignKeyConstraint, ...] = ()
    description: str = ""
    severity: Literal["error", "warning"] = "error"
    skip_if_empty: bool = False


@dataclass(frozen=True)
class ContractViolation:
    """Record of a contract violation.

    Attributes
    ----------
    contract
        The contract that was violated.
    message
        Description of the violation.
    severity
        Severity of the violation.
    details
        Additional violation details.
    """

    contract: IngestContractSpec
    message: str
    severity: Literal["error", "warning"]
    details: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class ContractValidationResult:
    """Result of contract validation.

    Attributes
    ----------
    valid
        Whether all contracts passed.
    violations
        List of violations found.
    warnings
        Non-fatal warnings.
    checked_contracts
        Number of contracts checked.
    tables_checked
        Tables that were validated.
    """

    valid: bool
    violations: tuple[ContractViolation, ...] = ()
    warnings: tuple[str, ...] = ()
    checked_contracts: int = 0
    tables_checked: tuple[str, ...] = ()

    @staticmethod
    def success(
        *,
        checked: int = 0,
        tables: tuple[str, ...] = (),
    ) -> ContractValidationResult:
        """Create a successful result.

        Parameters
        ----------
        checked
            Number of contracts checked.
        tables
            Tables that were validated.

        Returns
        -------
        ContractValidationResult
            Successful validation result.
        """
        return ContractValidationResult(
            valid=True,
            checked_contracts=checked,
            tables_checked=tables,
        )

    @staticmethod
    def failure(
        violations: Sequence[ContractViolation],
        *,
        checked: int = 0,
        warnings: Sequence[str] = (),
        tables: tuple[str, ...] = (),
    ) -> ContractValidationResult:
        """Create a failed result.

        Parameters
        ----------
        violations
            List of violations.
        checked
            Number of contracts checked.
        warnings
            Non-fatal warnings.
        tables
            Tables that were validated.

        Returns
        -------
        ContractValidationResult
            Failed validation result.
        """
        return ContractValidationResult(
            valid=False,
            violations=tuple(violations),
            warnings=tuple(warnings),
            checked_contracts=checked,
            tables_checked=tables,
        )


class IngestContractValidator:
    """Validator for ingestion plugin output contracts.

    Validates plugin outputs against declared contracts, checking
    row counts, column presence, and column constraints.
    """

    def __init__(self, gateway: StorageGateway) -> None:
        """Initialize the validator.

        Parameters
        ----------
        gateway
            Storage gateway for database access.
        """
        self._gateway = gateway

    def validate(
        self,
        contracts: Sequence[IngestContractSpec],
        snapshot: SnapshotRef,
    ) -> ContractValidationResult:
        """Validate contracts against current database state.

        Parameters
        ----------
        contracts
            Contracts to validate.
        snapshot
            Snapshot for scoping queries.

        Returns
        -------
        ContractValidationResult
            Validation result.
        """
        violations: list[ContractViolation] = []
        warnings: list[str] = []
        tables: set[str] = set()

        for contract in contracts:
            tables.add(contract.table)
            contract_violations = self._validate_contract(contract, snapshot)
            for violation in contract_violations:
                if violation.severity == "warning":
                    warnings.append(violation.message)
                else:
                    violations.append(violation)

        if violations:
            return ContractValidationResult.failure(
                violations,
                checked=len(contracts),
                warnings=warnings,
                tables=tuple(sorted(tables)),
            )

        return ContractValidationResult.success(
            checked=len(contracts),
            tables=tuple(sorted(tables)),
        )

    def _validate_contract(
        self,
        contract: IngestContractSpec,
        snapshot: SnapshotRef,
    ) -> list[ContractViolation]:
        """Validate a single contract.

        Returns
        -------
        list[ContractViolation]
            Violations found for this contract.
        """
        violations: list[ContractViolation] = []

        # Check if table exists
        if not self._table_exists(contract.table):
            violations.append(
                ContractViolation(
                    contract=contract,
                    message=f"Table {contract.table} does not exist",
                    severity=contract.severity,
                )
            )
            return violations

        # Check row count
        row_count = self._get_row_count(contract.table, snapshot)

        if contract.skip_if_empty and row_count == 0:
            return violations

        if contract.min_rows is not None and row_count < contract.min_rows:
            violations.append(
                ContractViolation(
                    contract=contract,
                    message=(
                        f"Table {contract.table} has {row_count} rows, "
                        f"expected at least {contract.min_rows}"
                    ),
                    severity=contract.severity,
                    details={"actual_rows": row_count, "min_rows": contract.min_rows},
                )
            )

        if contract.max_rows is not None and row_count > contract.max_rows:
            violations.append(
                ContractViolation(
                    contract=contract,
                    message=(
                        f"Table {contract.table} has {row_count} rows, "
                        f"expected at most {contract.max_rows}"
                    ),
                    severity=contract.severity,
                    details={"actual_rows": row_count, "max_rows": contract.max_rows},
                )
            )

        # Check required columns
        existing_columns = self._get_columns(contract.table)
        missing_cols = [col for col in contract.required_columns if col not in existing_columns]
        violations.extend(
            ContractViolation(
                contract=contract,
                message=f"Table {contract.table} missing required column: {col}",
                severity=contract.severity,
            )
            for col in missing_cols
        )

        # Check column constraints
        for constraint in contract.column_constraints:
            constraint_violations = self._validate_column_constraint(
                contract,
                constraint,
                snapshot,
            )
            violations.extend(constraint_violations)

        # Check foreign key constraints
        for fk in contract.foreign_keys:
            fk_violations = self._validate_foreign_key(contract, fk, snapshot)
            violations.extend(fk_violations)

        return violations

    def _table_exists(self, table_key: str) -> bool:
        """Check if a table exists.

        Returns
        -------
        bool
            True if table exists.
        """
        schema, table = table_key.split(".", maxsplit=1)
        try:
            result = self._gateway.con.execute(
                """
                SELECT 1 FROM information_schema.tables
                WHERE table_schema = ? AND table_name = ?
                """,
                [schema, table],
            ).fetchone()
        except Exception:  # noqa: BLE001
            return False
        return result is not None

    def _get_row_count(self, table_key: str, snapshot: SnapshotRef) -> int:
        """Get row count for a table scoped to snapshot.

        Returns
        -------
        int
            Row count.
        """
        # Try with repo/commit scope first
        count = self._try_scoped_count(table_key, snapshot)
        if count is not None:
            return count

        # Fall back to unscoped count
        count = self._try_unscoped_count(table_key)
        return count if count is not None else 0

    def _try_scoped_count(self, table_key: str, snapshot: SnapshotRef) -> int | None:
        """Try counting rows scoped to snapshot.

        Returns
        -------
        int | None
            Row count or None if failed.
        """
        try:
            result = self._gateway.con.execute(
                f"SELECT COUNT(*) FROM {table_key} WHERE repo = ? AND commit = ?",  # noqa: S608
                [snapshot.repo, snapshot.commit],
            ).fetchone()
            return int(result[0]) if result else None
        except Exception:  # noqa: BLE001
            return None

    def _try_unscoped_count(self, table_key: str) -> int | None:
        """Try counting rows without scope.

        Returns
        -------
        int | None
            Row count or None if failed.
        """
        try:
            result = self._gateway.con.execute(
                f"SELECT COUNT(*) FROM {table_key}",  # noqa: S608
            ).fetchone()
            return int(result[0]) if result else None
        except Exception:  # noqa: BLE001
            return None

    def _get_columns(self, table_key: str) -> set[str]:
        """Get column names for a table.

        Returns
        -------
        set[str]
            Column names.
        """
        schema, table = table_key.split(".", maxsplit=1)
        try:
            rows = self._gateway.con.execute(
                """
                SELECT column_name FROM information_schema.columns
                WHERE table_schema = ? AND table_name = ?
                """,
                [schema, table],
            ).fetchall()
            return {str(row[0]) for row in rows}
        except Exception:  # noqa: BLE001
            return set()

    def _validate_column_constraint(
        self,
        contract: IngestContractSpec,
        constraint: ColumnConstraint,
        snapshot: SnapshotRef,  # noqa: ARG002
    ) -> list[ContractViolation]:
        """Validate a column constraint.

        Parameters
        ----------
        contract
            Contract being validated.
        constraint
            Column constraint to check.
        snapshot
            Snapshot reference (reserved for future scoping).

        Returns
        -------
        list[ContractViolation]
            Violations found.
        """
        violations: list[ContractViolation] = []
        col = constraint.column
        table = contract.table

        try:
            violation = self._check_single_constraint(contract, constraint, table, col)
            if violation is not None:
                violations.append(violation)
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "Failed to validate constraint %s on %s.%s: %s", constraint, table, col, exc
            )

        return violations

    def _check_single_constraint(  # noqa: C901, PLR0911
        self,
        contract: IngestContractSpec,
        constraint: ColumnConstraint,
        table: str,
        col: str,
    ) -> ContractViolation | None:
        """Check a single constraint and return violation if any.

        Parameters
        ----------
        contract
            Contract being validated.
        constraint
            Column constraint to check.
        table
            Table name.
        col
            Column name.

        Returns
        -------
        ContractViolation | None
            Violation if constraint fails, None otherwise.
        """
        if constraint.constraint_type == "not_null":
            null_count = self._count_nulls(table, col)
            if null_count > 0:
                return ContractViolation(
                    contract=contract,
                    message=f"Column {table}.{col} has {null_count} NULL values",
                    severity=contract.severity,
                    details={"null_count": null_count},
                )

        elif constraint.constraint_type == "min_value" and constraint.value is not None:
            min_val = self._get_min_value(table, col)
            expected = (
                float(constraint.value) if isinstance(constraint.value, (int, float, str)) else 0.0
            )
            if min_val is not None and min_val < expected:
                return ContractViolation(
                    contract=contract,
                    message=f"Column {table}.{col} has min value {min_val}, expected >= {constraint.value}",
                    severity=contract.severity,
                )

        elif constraint.constraint_type == "max_value" and constraint.value is not None:
            max_val = self._get_max_value(table, col)
            expected = (
                float(constraint.value) if isinstance(constraint.value, (int, float, str)) else 0.0
            )
            if max_val is not None and max_val > expected:
                return ContractViolation(
                    contract=contract,
                    message=f"Column {table}.{col} has max value {max_val}, expected <= {constraint.value}",
                    severity=contract.severity,
                )

        elif constraint.constraint_type == "positive":
            non_positive = self._count_non_positive(table, col)
            if non_positive > 0:
                return ContractViolation(
                    contract=contract,
                    message=f"Column {table}.{col} has {non_positive} non-positive values",
                    severity=contract.severity,
                )

        elif constraint.constraint_type == "unique":
            dup_count = self._count_duplicates(table, col)
            if dup_count > 0:
                return ContractViolation(
                    contract=contract,
                    message=f"Column {table}.{col} has {dup_count} duplicate values",
                    severity=contract.severity,
                )

        elif constraint.constraint_type == "min_fraction_not_null" and constraint.value is not None:
            fraction = self._not_null_fraction(table, col)
            expected = (
                float(constraint.value) if isinstance(constraint.value, (int, float, str)) else 0.0
            )
            if fraction < expected:
                return ContractViolation(
                    contract=contract,
                    message=f"Column {table}.{col} has {fraction:.2%} non-null, expected >= {expected:.2%}",
                    severity=contract.severity,
                )

        return None

    def _validate_foreign_key(
        self,
        contract: IngestContractSpec,
        fk: ForeignKeyConstraint,
        snapshot: SnapshotRef,  # noqa: ARG002
    ) -> list[ContractViolation]:
        """Validate a foreign key constraint.

        Parameters
        ----------
        contract
            Contract being validated.
        fk
            Foreign key constraint to check.
        snapshot
            Snapshot reference (reserved for future scoping).

        Returns
        -------
        list[ContractViolation]
            Violations found.
        """
        violations: list[ContractViolation] = []

        try:
            # Count orphaned references
            null_clause = f"AND t.{fk.column} IS NOT NULL" if not fk.allow_null else ""
            query = f"""
                SELECT COUNT(*) FROM {contract.table} t
                LEFT JOIN {fk.reference_table} r
                    ON t.{fk.column} = r.{fk.reference_column}
                WHERE r.{fk.reference_column} IS NULL {null_clause}
            """  # noqa: S608
            result = self._gateway.con.execute(query).fetchone()
            orphan_count = int(result[0]) if result else 0

            if orphan_count > 0:
                violations.append(
                    ContractViolation(
                        contract=contract,
                        message=(
                            f"Foreign key {contract.table}.{fk.column} -> "
                            f"{fk.reference_table}.{fk.reference_column} has "
                            f"{orphan_count} orphaned references"
                        ),
                        severity=contract.severity,
                        details={"orphan_count": orphan_count},
                    )
                )
        except Exception as exc:  # noqa: BLE001
            log.warning("Failed to validate FK %s: %s", fk, exc)

        return violations

    def _count_nulls(self, table: str, column: str) -> int:
        """Count NULL values in a column.

        Returns
        -------
        int
            Count of NULL values.
        """
        try:
            result = self._gateway.con.execute(
                f"SELECT COUNT(*) FROM {table} WHERE {column} IS NULL",  # noqa: S608
            ).fetchone()
            return int(result[0]) if result else 0
        except Exception:  # noqa: BLE001
            return 0

    def _get_min_value(self, table: str, column: str) -> float | None:
        """Get minimum value in a column.

        Returns
        -------
        float | None
            Minimum value or None if not available.
        """
        try:
            result = self._gateway.con.execute(
                f"SELECT MIN({column}) FROM {table}",  # noqa: S608
            ).fetchone()
            return float(result[0]) if result and result[0] is not None else None
        except Exception:  # noqa: BLE001
            return None

    def _get_max_value(self, table: str, column: str) -> float | None:
        """Get maximum value in a column.

        Returns
        -------
        float | None
            Maximum value or None if not available.
        """
        try:
            result = self._gateway.con.execute(
                f"SELECT MAX({column}) FROM {table}",  # noqa: S608
            ).fetchone()
            return float(result[0]) if result and result[0] is not None else None
        except Exception:  # noqa: BLE001
            return None

    def _count_non_positive(self, table: str, column: str) -> int:
        """Count non-positive values in a column.

        Returns
        -------
        int
            Count of non-positive values.
        """
        try:
            result = self._gateway.con.execute(
                f"SELECT COUNT(*) FROM {table} WHERE {column} <= 0",  # noqa: S608
            ).fetchone()
            return int(result[0]) if result else 0
        except Exception:  # noqa: BLE001
            return 0

    def _count_duplicates(self, table: str, column: str) -> int:
        """Count duplicate values in a column.

        Returns
        -------
        int
            Count of duplicate values.
        """
        try:
            result = self._gateway.con.execute(
                f"""
                SELECT COUNT(*) - COUNT(DISTINCT {column}) FROM {table}
                WHERE {column} IS NOT NULL
                """,  # noqa: S608
            ).fetchone()
            return int(result[0]) if result else 0
        except Exception:  # noqa: BLE001
            return 0

    def _not_null_fraction(self, table: str, column: str) -> float:
        """Get fraction of non-null values in a column.

        Returns
        -------
        float
            Fraction of non-null values (0.0 to 1.0).
        """
        try:
            result = self._gateway.con.execute(
                f"""
                SELECT
                    CAST(COUNT({column}) AS DOUBLE) / NULLIF(COUNT(*), 0)
                FROM {table}
                """,  # noqa: S608
            ).fetchone()
            return float(result[0]) if result and result[0] is not None else 0.0
        except Exception:  # noqa: BLE001
            return 0.0


# Common contract builders for convenience


def row_count_contract(
    table: str,
    *,
    min_rows: int | None = None,
    max_rows: int | None = None,
    plugin_name: str = "",
    severity: Literal["error", "warning"] = "error",
) -> IngestContractSpec:
    """Create a row count contract.

    Parameters
    ----------
    table
        Table key.
    min_rows
        Minimum expected rows.
    max_rows
        Maximum expected rows.
    plugin_name
        Plugin producing this output.
    severity
        Violation severity.

    Returns
    -------
    IngestContractSpec
        Row count contract.
    """
    return IngestContractSpec(
        table=table,
        plugin_name=plugin_name,
        min_rows=min_rows,
        max_rows=max_rows,
        severity=severity,
    )


def not_null_contract(
    table: str,
    columns: Sequence[str],
    *,
    plugin_name: str = "",
    severity: Literal["error", "warning"] = "error",
) -> IngestContractSpec:
    """Create a not-null contract for columns.

    Parameters
    ----------
    table
        Table key.
    columns
        Columns that must not have NULL values.
    plugin_name
        Plugin producing this output.
    severity
        Violation severity.

    Returns
    -------
    IngestContractSpec
        Not-null contract.
    """
    return IngestContractSpec(
        table=table,
        plugin_name=plugin_name,
        column_constraints=tuple(
            ColumnConstraint(column=col, constraint_type="not_null") for col in columns
        ),
        severity=severity,
    )


def foreign_key_contract(  # noqa: PLR0913
    table: str,
    column: str,
    reference_table: str,
    reference_column: str,
    *,
    allow_null: bool = True,
    plugin_name: str = "",
    severity: Literal["error", "warning"] = "error",
) -> IngestContractSpec:
    """Create a foreign key contract.

    Parameters
    ----------
    table
        Source table key.
    column
        Source column.
    reference_table
        Target table key.
    reference_column
        Target column.
    allow_null
        Whether NULL values are allowed.
    plugin_name
        Plugin producing this output.
    severity
        Violation severity.

    Returns
    -------
    IngestContractSpec
        Foreign key contract.
    """
    return IngestContractSpec(
        table=table,
        plugin_name=plugin_name,
        foreign_keys=(
            ForeignKeyConstraint(
                column=column,
                reference_table=reference_table,
                reference_column=reference_column,
                allow_null=allow_null,
            ),
        ),
        severity=severity,
    )


__all__ = [
    "ColumnConstraint",
    "ContractValidationResult",
    "ContractViolation",
    "ForeignKeyConstraint",
    "IngestContractSpec",
    "IngestContractValidator",
    "foreign_key_contract",
    "not_null_contract",
    "row_count_contract",
]
