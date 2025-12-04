"""Declarative output contracts for analytics plugins.

This module provides a contract system for validating plugin outputs,
ensuring data quality and consistency across the analytics pipeline.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from codeintel.config.primitives import SnapshotRef
from codeintel.storage.gateway import StorageGateway
from codeintel.storage.sql_builder import QueryBuilder, SafeColumn, SafeTable, render_sql

if TYPE_CHECKING:
    from codeintel.analytics.core.protocol import (
        AnalyticsPluginProtocol,
        PluginMetadata,
        PluginOutputSpec,
    )


@dataclass(frozen=True)
class ColumnConstraint:
    """Constraint on a single column.

    Attributes
    ----------
    column
        Column name.
    constraint_type
        Type of constraint.
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
    ]
    value: float | str | None = None


@dataclass(frozen=True)
class OutputContractSpec:
    """Specification for plugin output validation.

    Attributes
    ----------
    table
        Table to validate.
    min_rows
        Minimum expected rows.
    required_columns
        Columns that must exist.
    column_constraints
        Constraints on column values.
    description
        Human-readable description.
    severity
        How violations should be handled.
    """

    table: str
    min_rows: int | None = None
    required_columns: tuple[str, ...] = ()
    column_constraints: tuple[ColumnConstraint, ...] = ()
    description: str = ""
    severity: Literal["error", "warning"] = "error"


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

    contract: OutputContractSpec
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
    """

    valid: bool
    violations: tuple[ContractViolation, ...] = ()
    warnings: tuple[str, ...] = ()
    checked_contracts: int = 0

    @staticmethod
    def success(*, checked: int = 0) -> ContractValidationResult:
        """Create a successful result.

        Parameters
        ----------
        checked
            Number of contracts checked.

        Returns
        -------
        ContractValidationResult
            Successful validation result.
        """
        return ContractValidationResult(valid=True, checked_contracts=checked)

    @staticmethod
    def failure(
        violations: Sequence[ContractViolation],
        *,
        checked: int = 0,
    ) -> ContractValidationResult:
        """Create a failed result.

        Parameters
        ----------
        violations
            List of violations.
        checked
            Number of contracts checked.

        Returns
        -------
        ContractValidationResult
            Failed validation result.
        """
        return ContractValidationResult(
            valid=False,
            violations=tuple(violations),
            checked_contracts=checked,
        )


class ContractValidator:
    """Validator for plugin output contracts.

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
        contracts: Sequence[OutputContractSpec],
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

        for contract in contracts:
            contract_violations = self._validate_contract(contract, snapshot)
            for violation in contract_violations:
                if violation.severity == "warning":
                    warnings.append(violation.message)
                else:
                    violations.append(violation)

        return ContractValidationResult(
            valid=len(violations) == 0,
            violations=tuple(violations),
            warnings=tuple(warnings),
            checked_contracts=len(contracts),
        )

    def _validate_contract(
        self,
        contract: OutputContractSpec,
        snapshot: SnapshotRef,
    ) -> list[ContractViolation]:
        """Validate a single contract.

        Parameters
        ----------
        contract
            Contract to validate.
        snapshot
            Snapshot for scoping.

        Returns
        -------
        list[ContractViolation]
            List of violations found.
        """
        violations: list[ContractViolation] = []

        # Check table exists
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
        if contract.min_rows is not None:
            row_count = self._count_rows(contract.table, snapshot)
            if row_count < contract.min_rows:
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

        # Check required columns
        if contract.required_columns:
            existing_columns = self._get_columns(contract.table)
            missing = set(contract.required_columns) - existing_columns
            if missing:
                violations.append(
                    ContractViolation(
                        contract=contract,
                        message=f"Table {contract.table} missing columns: {sorted(missing)}",
                        severity=contract.severity,
                        details={"missing_columns": sorted(missing)},
                    )
                )

        # Check column constraints
        for constraint in contract.column_constraints:
            constraint_violations = self._check_constraint(contract, constraint, snapshot)
            violations.extend(constraint_violations)

        return violations

    def _table_exists(self, table: str) -> bool:
        """Check if a table exists.

        Parameters
        ----------
        table
            Table name to check.

        Returns
        -------
        bool
            True if table exists.
        """
        schema, name = self._split_table(table)
        query = """
            SELECT 1 FROM information_schema.tables
            WHERE table_schema = ? AND table_name = ?
            LIMIT 1
        """
        row = self._gateway.con.execute(query, [schema, name]).fetchone()
        return row is not None

    def _count_rows(self, table: str, snapshot: SnapshotRef) -> int:
        """Count rows for a snapshot.

        Parameters
        ----------
        table
            Table to count.
        snapshot
            Snapshot for scoping.

        Returns
        -------
        int
            Row count.
        """
        query, params = QueryBuilder.count(
            SafeTable(table),
            where={"repo": snapshot.repo, "commit": snapshot.commit},
        )
        row = self._gateway.con.execute(query, params).fetchone()
        return int(row[0]) if row else 0

    def _get_columns(self, table: str) -> set[str]:
        """Get column names for a table.

        Parameters
        ----------
        table
            Table to inspect.

        Returns
        -------
        set[str]
            Column names.
        """
        schema, name = self._split_table(table)
        query = """
            SELECT column_name FROM information_schema.columns
            WHERE table_schema = ? AND table_name = ?
        """
        rows = self._gateway.con.execute(query, [schema, name]).fetchall()
        return {row[0] for row in rows}

    def _check_constraint(
        self,
        contract: OutputContractSpec,
        constraint: ColumnConstraint,
        snapshot: SnapshotRef,
    ) -> list[ContractViolation]:
        """Check a column constraint.

        Parameters
        ----------
        contract
            Parent contract.
        constraint
            Constraint to check.
        snapshot
            Snapshot for scoping.

        Returns
        -------
        list[ContractViolation]
            List of violations found.
        """
        violations: list[ContractViolation] = []

        if constraint.constraint_type == "not_null":
            null_count = self._count_nulls(contract.table, constraint.column, snapshot)
            if null_count > 0:
                violations.append(
                    ContractViolation(
                        contract=contract,
                        message=(
                            f"Column {constraint.column} in {contract.table} "
                            f"has {null_count} NULL values"
                        ),
                        severity=contract.severity,
                        details={"null_count": null_count},
                    )
                )

        elif constraint.constraint_type == "min_fraction_not_null":
            fraction = self._not_null_fraction(contract.table, constraint.column, snapshot)
            min_fraction = float(constraint.value) if constraint.value is not None else 0.0
            if fraction < min_fraction:
                violations.append(
                    ContractViolation(
                        contract=contract,
                        message=(
                            f"Column {constraint.column} in {contract.table} "
                            f"has {fraction:.2%} non-null, expected {min_fraction:.2%}"
                        ),
                        severity=contract.severity,
                        details={"actual_fraction": fraction, "min_fraction": min_fraction},
                    )
                )

        return violations

    def _count_nulls(
        self,
        table: str,
        column: str,
        snapshot: SnapshotRef,
    ) -> int:
        """Count NULL values in a column.

        Parameters
        ----------
        table
            Table to check.
        column
            Column to check.
        snapshot
            Snapshot for scoping.

        Returns
        -------
        int
            Number of NULL values.
        """
        safe_table = SafeTable(table)
        safe_col = SafeColumn(column)
        where_clause = " AND ".join(
            (
                f"{SafeColumn('repo')} = ?",
                f"{SafeColumn('commit')} = ?",
                f"{safe_col} IS NULL",
            )
        )
        query = render_sql(
            [
                "SELECT COUNT(*) FROM",
                str(safe_table),
                "WHERE",
                where_clause,
            ]
        )
        row = self._gateway.con.execute(query, [snapshot.repo, snapshot.commit]).fetchone()
        return int(row[0]) if row else 0

    def _not_null_fraction(
        self,
        table: str,
        column: str,
        snapshot: SnapshotRef,
    ) -> float:
        """Calculate fraction of non-null values in a column.

        Parameters
        ----------
        table
            Table to check.
        column
            Column to check.
        snapshot
            Snapshot for scoping.

        Returns
        -------
        float
            Fraction of non-null values (0.0 to 1.0).
        """
        safe_table = SafeTable(table)
        safe_col = SafeColumn(column)
        select_expr = ", ".join(
            (
                "COUNT(*) as total",
                f"COUNT({safe_col}) as non_null",
            )
        )
        where_clause = " AND ".join((f"{SafeColumn('repo')} = ?", f"{SafeColumn('commit')} = ?"))
        query = render_sql(
            [
                "SELECT",
                select_expr,
                "FROM",
                str(safe_table),
                "WHERE",
                where_clause,
            ]
        )
        row = self._gateway.con.execute(query, [snapshot.repo, snapshot.commit]).fetchone()
        if row is None or row[0] == 0:
            return 0.0
        return float(row[1]) / float(row[0])

    @staticmethod
    def _split_table(table: str) -> tuple[str, str]:
        """Split table into schema and name.

        Parameters
        ----------
        table
            Full table name.

        Returns
        -------
        tuple[str, str]
            Schema and table name.
        """
        if "." in table:
            schema, name = table.split(".", maxsplit=1)
        else:
            schema, name = "analytics", table
        return schema, name


@dataclass(frozen=True)
class PluginOutputContract:
    """Complete output contract for a plugin.

    Attributes
    ----------
    plugin_name
        Name of the plugin.
    contracts
        Output contracts to validate.
    run_on_success_only
        Whether to only run on successful execution.
    """

    plugin_name: str
    contracts: tuple[OutputContractSpec, ...]
    run_on_success_only: bool = True


ContractCheckerFn = Callable[[StorageGateway, SnapshotRef], ContractValidationResult]


def create_contract_checker(
    contracts: Sequence[OutputContractSpec],
) -> ContractCheckerFn:
    """Create a contract checker function from contract specs.

    Parameters
    ----------
    contracts
        Contracts to check.

    Returns
    -------
    ContractCheckerFn
        Checker function.
    """

    def checker(
        gateway: StorageGateway,
        snapshot: SnapshotRef,
    ) -> ContractValidationResult:
        validator = ContractValidator(gateway)
        return validator.validate(contracts, snapshot)

    return checker


def _contracts_from_metadata(metadata: PluginMetadata) -> tuple[OutputContractSpec, ...]:
    return tuple(
        contract for output in metadata.outputs for contract in _contracts_from_output_spec(output)
    )


def _contracts_from_output_spec(output: PluginOutputSpec) -> tuple[OutputContractSpec, ...]:
    return tuple(
        OutputContractSpec(
            table=table,
            min_rows=output.min_rows,
            required_columns=output.required_columns,
        )
        for table in output.tables
    )


def _explicit_output_contracts(plugin: object) -> tuple[OutputContractSpec, ...]:
    explicit = getattr(plugin, "output_contracts", None)
    if not explicit or not isinstance(explicit, Sequence):
        return ()
    return tuple(contract for contract in explicit if isinstance(contract, OutputContractSpec))


def build_plugin_output_contracts(
    plugin: AnalyticsPluginProtocol,
) -> tuple[PluginOutputContract, ...]:
    """
    Derive output contracts for a plugin from explicit contracts or metadata.

    Explicit ``output_contracts`` on the plugin take precedence; otherwise,
    contracts are synthesized from ``PluginMetadata.outputs`` by converting
    each output table into an ``OutputContractSpec``. Contracts are deduplicated
    to avoid redundant validation passes.

    Returns
    -------
    tuple[PluginOutputContract, ...]
        Contracts grouped per plugin, or empty when no contracts apply.
    """
    explicit_contracts = _explicit_output_contracts(plugin)
    metadata_contracts = _contracts_from_metadata(plugin.metadata)

    all_contracts: list[OutputContractSpec] = []
    seen: set[tuple[str, tuple[str, ...], int | None]] = set()

    for contract in (*explicit_contracts, *metadata_contracts):
        key = (contract.table, contract.required_columns, contract.min_rows)
        if key in seen:
            continue
        seen.add(key)
        all_contracts.append(contract)

    if not all_contracts:
        return ()

    return (
        PluginOutputContract(
            plugin_name=plugin.metadata.name,
            contracts=tuple(all_contracts),
        ),
    )


def validate_plugin_outputs(
    gateway: StorageGateway,
    snapshot: SnapshotRef,
    plugin_contracts: Sequence[PluginOutputContract],
) -> dict[str, ContractValidationResult]:
    """Validate outputs for multiple plugins.

    Parameters
    ----------
    gateway
        Storage gateway.
    snapshot
        Snapshot for scoping.
    plugin_contracts
        Contracts for each plugin.

    Returns
    -------
    dict[str, ContractValidationResult]
        Validation results keyed by plugin name.
    """
    validator = ContractValidator(gateway)
    results: dict[str, ContractValidationResult] = {}

    for plugin_contract in plugin_contracts:
        result = validator.validate(plugin_contract.contracts, snapshot)
        results[plugin_contract.plugin_name] = result

    return results


__all__ = [
    "ColumnConstraint",
    "ContractCheckerFn",
    "ContractValidationResult",
    "ContractValidator",
    "ContractViolation",
    "OutputContractSpec",
    "PluginOutputContract",
    "create_contract_checker",
    "validate_plugin_outputs",
]
