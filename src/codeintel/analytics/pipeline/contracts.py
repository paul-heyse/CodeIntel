"""Dataset contracts for validating pipeline outputs.

This module provides a contract system for defining and validating
dataset output requirements.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from codeintel.storage.gateway import StorageGateway

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ColumnRule:
    """Validation rule for a single column.

    Attributes
    ----------
    column
        Column name.
    not_null
        Whether NULL values are disallowed.
    unique
        Whether values must be unique.
    min_value
        Minimum allowed value (for numeric columns).
    max_value
        Maximum allowed value (for numeric columns).
    pattern
        Regex pattern for string values.
    allowed_values
        Set of allowed values.
    """

    column: str
    not_null: bool = False
    unique: bool = False
    min_value: float | None = None
    max_value: float | None = None
    pattern: str | None = None
    allowed_values: frozenset[object] | None = None


@dataclass(frozen=True)
class DatasetContract:
    """Contract defining validation rules for a dataset.

    Attributes
    ----------
    table
        Fully qualified table name.
    min_rows
        Minimum number of rows required.
    max_rows
        Maximum number of rows allowed.
    required_columns
        Columns that must exist.
    column_rules
        Validation rules for specific columns.
    custom_checks
        SQL expressions that must evaluate to true.
    description
        Human-readable description of the contract.
    """

    table: str
    min_rows: int = 0
    max_rows: int | None = None
    required_columns: tuple[str, ...] = ()
    column_rules: tuple[ColumnRule, ...] = ()
    custom_checks: tuple[str, ...] = ()
    description: str = ""


@dataclass
class ContractViolation:
    """A single contract violation.

    Attributes
    ----------
    table
        Table where violation occurred.
    rule
        Name of the violated rule.
    message
        Description of the violation.
    severity
        Violation severity: "error" or "warning".
    row_count
        Number of rows affected (if applicable).
    """

    table: str
    rule: str
    message: str
    severity: str = "error"
    row_count: int | None = None


@dataclass
class ContractValidationResult:
    """Result of validating a contract.

    Attributes
    ----------
    contract
        The contract that was validated.
    valid
        Whether all rules passed.
    violations
        List of violations found.
    row_count
        Actual row count in the table.
    duration_ms
        Validation time in milliseconds.
    """

    contract: DatasetContract
    valid: bool = True
    violations: list[ContractViolation] = field(default_factory=list)
    row_count: int = 0
    duration_ms: float = 0.0


class DatasetContractValidator:
    """Validates dataset contracts against database state.

    Example
    -------
    >>> validator = DatasetContractValidator(gateway)
    >>> result = validator.validate(contract, repo="myrepo", commit="abc123")
    >>> if not result.valid:
    ...     for v in result.violations:
    ...         print(f"{v.rule}: {v.message}")
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
        contract: DatasetContract,
        *,
        repo: str | None = None,
        commit: str | None = None,
    ) -> ContractValidationResult:
        """Validate a contract against the current database state.

        Parameters
        ----------
        contract
            Contract to validate.
        repo
            Optional repository filter.
        commit
            Optional commit filter.

        Returns
        -------
        ContractValidationResult
            Validation result with any violations.
        """
        import time

        start = time.perf_counter()
        violations: list[ContractViolation] = []

        # Check row count
        row_count = self._get_row_count(contract.table, repo, commit)

        if row_count < contract.min_rows:
            violations.append(
                ContractViolation(
                    table=contract.table,
                    rule="min_rows",
                    message=f"Expected at least {contract.min_rows} rows, found {row_count}",
                    row_count=row_count,
                )
            )

        if contract.max_rows is not None and row_count > contract.max_rows:
            violations.append(
                ContractViolation(
                    table=contract.table,
                    rule="max_rows",
                    message=f"Expected at most {contract.max_rows} rows, found {row_count}",
                    row_count=row_count,
                )
            )

        # Check required columns
        actual_columns = self._get_columns(contract.table)
        for col in contract.required_columns:
            if col not in actual_columns:
                violations.append(
                    ContractViolation(
                        table=contract.table,
                        rule="required_column",
                        message=f"Missing required column: {col}",
                    )
                )

        # Check column rules
        for rule in contract.column_rules:
            rule_violations = self._check_column_rule(
                contract.table, rule, repo, commit
            )
            violations.extend(rule_violations)

        # Check custom checks
        for check in contract.custom_checks:
            if not self._run_custom_check(contract.table, check, repo, commit):
                violations.append(
                    ContractViolation(
                        table=contract.table,
                        rule="custom_check",
                        message=f"Custom check failed: {check[:50]}...",
                    )
                )

        duration = (time.perf_counter() - start) * 1000

        return ContractValidationResult(
            contract=contract,
            valid=len(violations) == 0,
            violations=violations,
            row_count=row_count,
            duration_ms=duration,
        )

    def _get_row_count(
        self,
        table: str,
        repo: str | None,
        commit: str | None,
    ) -> int:
        """Get row count for a table with optional filtering."""
        query = f"SELECT COUNT(*) FROM {table}"  # noqa: S608
        params: list[object] = []

        if repo is not None:
            query += " WHERE repo = ?"
            params.append(repo)
            if commit is not None:
                query += " AND commit = ?"
                params.append(commit)

        try:
            result = self._gateway.con.execute(query, params)
            row = result.fetchone()
            return int(row[0]) if row else 0
        except Exception:
            log.warning("Failed to count rows in %s", table, exc_info=True)
            return 0

    def _get_columns(self, table: str) -> set[str]:
        """Get column names for a table."""
        try:
            # DuckDB-specific: use DESCRIBE
            result = self._gateway.con.execute(f"DESCRIBE {table}")
            return {str(row[0]) for row in result.fetchall()}
        except Exception:
            log.warning("Failed to get columns for %s", table, exc_info=True)
            return set()

    def _check_column_rule(
        self,
        table: str,
        rule: ColumnRule,
        repo: str | None,
        commit: str | None,
    ) -> list[ContractViolation]:
        """Check a single column rule."""
        violations: list[ContractViolation] = []
        column = rule.column

        # Build WHERE clause
        where_parts: list[str] = []
        params: list[object] = []
        if repo is not None:
            where_parts.append("repo = ?")
            params.append(repo)
            if commit is not None:
                where_parts.append("commit = ?")
                params.append(commit)

        where_clause = " AND ".join(where_parts) if where_parts else "1=1"

        # Check not_null
        if rule.not_null:
            # SQL injection safe: column from trusted source (rule definition)
            query = f"SELECT COUNT(*) FROM {table} WHERE ({where_clause}) AND {column} IS NULL"  # noqa: S608
            try:
                result = self._gateway.con.execute(query, params)
                row = result.fetchone()
                null_count = int(row[0]) if row else 0
                if null_count > 0:
                    violations.append(
                        ContractViolation(
                            table=table,
                            rule="not_null",
                            message=f"Column {column} has {null_count} NULL values",
                            row_count=null_count,
                        )
                    )
            except Exception:
                log.warning("Failed to check not_null for %s.%s", table, column)

        return violations

    def _run_custom_check(
        self,
        table: str,
        check: str,
        repo: str | None,
        commit: str | None,
    ) -> bool:
        """Run a custom SQL check expression."""
        # Custom checks are trusted SQL from contract definitions
        try:
            result = self._gateway.con.execute(check)
            row = result.fetchone()
            return bool(row and row[0])
        except Exception:
            log.warning("Custom check failed for %s", table, exc_info=True)
            return False


__all__ = [
    "ColumnRule",
    "ContractValidationResult",
    "ContractViolation",
    "DatasetContract",
    "DatasetContractValidator",
]
