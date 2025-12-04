"""Contract validation for ingestion plugin outputs.

This module provides a contract system for validating plugin outputs,
ensuring data quality and consistency across the ingestion pipeline.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from codeintel.ingestion.utilities.db_queries import (
    ForeignKeyRef,
    safe_count,
    safe_count_duplicates,
    safe_count_non_positive,
    safe_count_nulls,
    safe_count_orphan_refs,
    safe_count_with_scope,
    safe_get_columns,
    safe_max_value,
    safe_min_value,
    safe_not_null_fraction,
    safe_table_exists,
)

if TYPE_CHECKING:
    from codeintel.config.primitives import SnapshotRef
    from codeintel.storage.gateway import StorageGateway

log = logging.getLogger(__name__)


# Type alias for constraint checker functions
ConstraintCheckerFn = Callable[
    ["ConstraintCheckerContext"],
    "ContractViolation | None",
]


@dataclass(frozen=True)
class ConstraintCheckerContext:
    """Context for constraint checking operations.

    Encapsulates all data needed by constraint checker functions.

    Attributes
    ----------
    contract
        The contract being validated.
    constraint
        The column constraint to check.
    table
        Table name (schema.table format).
    column
        Column name.
    gateway
        Storage gateway for database queries.
    """

    contract: IngestContractSpec
    constraint: ColumnConstraint
    table: str
    column: str
    gateway: StorageGateway


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


# Constraint checker functions using strategy pattern


def _check_not_null(ctx: ConstraintCheckerContext) -> ContractViolation | None:
    """Check not_null constraint.

    Returns
    -------
    ContractViolation | None
        Violation if constraint fails, None otherwise.
    """
    null_count = safe_count_nulls(ctx.gateway, ctx.table, ctx.column)
    if null_count > 0:
        return ContractViolation(
            contract=ctx.contract,
            message=f"Column {ctx.table}.{ctx.column} has {null_count} NULL values",
            severity=ctx.contract.severity,
            details={"null_count": null_count},
        )
    return None


def _check_min_value(ctx: ConstraintCheckerContext) -> ContractViolation | None:
    """Check min_value constraint.

    Returns
    -------
    ContractViolation | None
        Violation if constraint fails, None otherwise.
    """
    if ctx.constraint.value is None:
        return None

    min_val = safe_min_value(ctx.gateway, ctx.table, ctx.column)
    expected = (
        float(ctx.constraint.value) if isinstance(ctx.constraint.value, (int, float, str)) else 0.0
    )
    if min_val is not None and min_val < expected:
        return ContractViolation(
            contract=ctx.contract,
            message=(
                f"Column {ctx.table}.{ctx.column} has min value {min_val}, "
                f"expected >= {ctx.constraint.value}"
            ),
            severity=ctx.contract.severity,
        )
    return None


def _check_max_value(ctx: ConstraintCheckerContext) -> ContractViolation | None:
    """Check max_value constraint.

    Returns
    -------
    ContractViolation | None
        Violation if constraint fails, None otherwise.
    """
    if ctx.constraint.value is None:
        return None

    max_val = safe_max_value(ctx.gateway, ctx.table, ctx.column)
    expected = (
        float(ctx.constraint.value) if isinstance(ctx.constraint.value, (int, float, str)) else 0.0
    )
    if max_val is not None and max_val > expected:
        return ContractViolation(
            contract=ctx.contract,
            message=(
                f"Column {ctx.table}.{ctx.column} has max value {max_val}, "
                f"expected <= {ctx.constraint.value}"
            ),
            severity=ctx.contract.severity,
        )
    return None


def _check_positive(ctx: ConstraintCheckerContext) -> ContractViolation | None:
    """Check positive constraint.

    Returns
    -------
    ContractViolation | None
        Violation if constraint fails, None otherwise.
    """
    non_positive = safe_count_non_positive(ctx.gateway, ctx.table, ctx.column)
    if non_positive > 0:
        return ContractViolation(
            contract=ctx.contract,
            message=f"Column {ctx.table}.{ctx.column} has {non_positive} non-positive values",
            severity=ctx.contract.severity,
        )
    return None


def _check_unique(ctx: ConstraintCheckerContext) -> ContractViolation | None:
    """Check unique constraint.

    Returns
    -------
    ContractViolation | None
        Violation if constraint fails, None otherwise.
    """
    dup_count = safe_count_duplicates(ctx.gateway, ctx.table, ctx.column)
    if dup_count > 0:
        return ContractViolation(
            contract=ctx.contract,
            message=f"Column {ctx.table}.{ctx.column} has {dup_count} duplicate values",
            severity=ctx.contract.severity,
        )
    return None


def _check_min_fraction_not_null(ctx: ConstraintCheckerContext) -> ContractViolation | None:
    """Check min_fraction_not_null constraint.

    Returns
    -------
    ContractViolation | None
        Violation if constraint fails, None otherwise.
    """
    if ctx.constraint.value is None:
        return None

    fraction = safe_not_null_fraction(ctx.gateway, ctx.table, ctx.column)
    expected = (
        float(ctx.constraint.value) if isinstance(ctx.constraint.value, (int, float, str)) else 0.0
    )
    if fraction < expected:
        return ContractViolation(
            contract=ctx.contract,
            message=(
                f"Column {ctx.table}.{ctx.column} has {fraction:.2%} non-null, "
                f"expected >= {expected:.2%}"
            ),
            severity=ctx.contract.severity,
        )
    return None


# Registry mapping constraint types to checker functions
CONSTRAINT_CHECKERS: dict[str, ConstraintCheckerFn] = {
    "not_null": _check_not_null,
    "min_value": _check_min_value,
    "max_value": _check_max_value,
    "positive": _check_positive,
    "unique": _check_unique,
    "min_fraction_not_null": _check_min_fraction_not_null,
}


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
            )
            violations.extend(constraint_violations)

        # Check foreign key constraints
        for fk in contract.foreign_keys:
            fk_violations = self._validate_foreign_key(contract, fk)
            violations.extend(fk_violations)

        return violations

    def _table_exists(self, table_key: str) -> bool:
        """Check if a table exists.

        Returns
        -------
        bool
            True if table exists.
        """
        return safe_table_exists(self._gateway, table_key)

    def _get_row_count(self, table_key: str, snapshot: SnapshotRef) -> int:
        """Get row count for a table scoped to snapshot.

        Returns
        -------
        int
            Row count.
        """
        # Try with repo/commit scope first
        count = safe_count_with_scope(self._gateway, table_key, snapshot)
        if count is not None:
            return count

        # Fall back to unscoped count
        count = safe_count(self._gateway, table_key)
        return count if count is not None else 0

    def _get_columns(self, table_key: str) -> set[str]:
        """Get column names for a table.

        Returns
        -------
        set[str]
            Column names.
        """
        return safe_get_columns(self._gateway, table_key)

    def _validate_column_constraint(
        self,
        contract: IngestContractSpec,
        constraint: ColumnConstraint,
    ) -> list[ContractViolation]:
        """Validate a column constraint.

        Parameters
        ----------
        contract
            Contract being validated.
        constraint
            Column constraint to check.

        Returns
        -------
        list[ContractViolation]
            Violations found.
        """
        violations: list[ContractViolation] = []
        col = constraint.column
        table = contract.table

        violation = self._check_single_constraint(contract, constraint, table, col)
        if violation is not None:
            violations.append(violation)

        return violations

    def _check_single_constraint(
        self,
        contract: IngestContractSpec,
        constraint: ColumnConstraint,
        table: str,
        col: str,
    ) -> ContractViolation | None:
        """Check a single constraint and return violation if any.

        Uses the CONSTRAINT_CHECKERS registry to dispatch to the appropriate
        checker function based on constraint type.

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
        checker = CONSTRAINT_CHECKERS.get(constraint.constraint_type)
        if checker is None:
            # Unsupported constraint type (e.g., "in_set", "regex")
            log.debug(
                "No checker registered for constraint type: %s",
                constraint.constraint_type,
            )
            return None

        ctx = ConstraintCheckerContext(
            contract=contract,
            constraint=constraint,
            table=table,
            column=col,
            gateway=self._gateway,
        )
        return checker(ctx)

    def _validate_foreign_key(
        self,
        contract: IngestContractSpec,
        fk: ForeignKeyConstraint,
    ) -> list[ContractViolation]:
        """Validate a foreign key constraint.

        Parameters
        ----------
        contract
            Contract being validated.
        fk
            Foreign key constraint to check.

        Returns
        -------
        list[ContractViolation]
            Violations found.
        """
        violations: list[ContractViolation] = []

        fk_ref = ForeignKeyRef(
            source_table=contract.table,
            source_column=fk.column,
            ref_table=fk.reference_table,
            ref_column=fk.reference_column,
            allow_null=fk.allow_null,
        )
        orphan_count = safe_count_orphan_refs(self._gateway, fk_ref)

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

        return violations


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


@dataclass(frozen=True)
class ForeignKeyContractSpec:
    """Specification for creating a foreign key contract.

    Attributes
    ----------
    allow_null
        Whether NULL values are allowed.
    plugin_name
        Plugin producing this output.
    severity
        Violation severity.
    """

    allow_null: bool = True
    plugin_name: str = ""
    severity: Literal["error", "warning"] = "error"


def foreign_key_contract(
    table: str,
    column: str,
    reference_table: str,
    reference_column: str,
    spec: ForeignKeyContractSpec | None = None,
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
    spec
        Optional specification with additional options.

    Returns
    -------
    IngestContractSpec
        Foreign key contract.
    """
    s = spec or ForeignKeyContractSpec()
    return IngestContractSpec(
        table=table,
        plugin_name=s.plugin_name,
        foreign_keys=(
            ForeignKeyConstraint(
                column=column,
                reference_table=reference_table,
                reference_column=reference_column,
                allow_null=s.allow_null,
            ),
        ),
        severity=s.severity,
    )


__all__ = [
    "CONSTRAINT_CHECKERS",
    "ColumnConstraint",
    "ConstraintCheckerContext",
    "ContractValidationResult",
    "ContractViolation",
    "ForeignKeyConstraint",
    "ForeignKeyContractSpec",
    "IngestContractSpec",
    "IngestContractValidator",
    "foreign_key_contract",
    "not_null_contract",
    "row_count_contract",
]
