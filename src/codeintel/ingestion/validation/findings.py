"""Finding types and validation results for ingestion validation.

This module provides ingestion-specific validation types and utilities,
extending the core validation infrastructure with contract-based validation.

Note: The helper functions in this module work with typed ``ContractViolation``
objects rather than generic dicts, so they have specialized implementations
that complement the core validation helpers.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal

from codeintel.core.validation import BaseValidationOptions, ValidationSeverity

# =============================================================================
# Constants
# =============================================================================

# Maximum sample findings to return per rule
SAMPLE_LIMIT = 10

# Minimum row count threshold for validation
MIN_ROW_THRESHOLD = 1

# Default maximum findings per rule
MAX_FINDINGS_PER_RULE = 50


# =============================================================================
# Validation Options
# =============================================================================


@dataclass(frozen=True)
class IngestValidationOptions(BaseValidationOptions):
    """Options for controlling ingestion validation behavior.

    Extend ``BaseValidationOptions`` with ingestion-specific options.

    Attributes
    ----------
    severity_overrides
        Mapping of rule names to severity levels (inherited).
    hard_fail
        Whether to raise on error-level findings (inherited).
    max_findings_per_rule
        Maximum findings per rule (inherited, with different default).
    skip_empty_tables
        Whether to skip validation of empty tables.
    """

    max_findings_per_rule: int | None = MAX_FINDINGS_PER_RULE
    skip_empty_tables: bool = True


# =============================================================================
# Constraint Types
# =============================================================================


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


# =============================================================================
# Contract Specification
# =============================================================================


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


# =============================================================================
# Violation Types
# =============================================================================


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


# =============================================================================
# Validation Results
# =============================================================================


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


# =============================================================================
# Finding Helpers (typed for ContractViolation)
# =============================================================================


def apply_severity_overrides(
    violations: Sequence[ContractViolation],
    overrides: Mapping[str, ValidationSeverity] | None,
) -> list[ContractViolation]:
    """Apply severity overrides to violations.

    Note: This is a specialized version that works with typed
    ``ContractViolation`` objects rather than generic dicts.

    Parameters
    ----------
    violations
        Violations to process.
    overrides
        Mapping of table names or "*" to severity levels.

    Returns
    -------
    list[ContractViolation]
        Violations with potentially modified severities.
    """
    if not overrides:
        return list(violations)

    result: list[ContractViolation] = []
    for violation in violations:
        table = violation.contract.table
        # Check for specific override first, then wildcard
        override = overrides.get(table) or overrides.get("*")
        if override == "error":
            result.append(
                ContractViolation(
                    contract=violation.contract,
                    message=violation.message,
                    severity="error",
                    details=violation.details,
                )
            )
        elif override == "warning":
            result.append(
                ContractViolation(
                    contract=violation.contract,
                    message=violation.message,
                    severity="warning",
                    details=violation.details,
                )
            )
        else:
            # "info" or no override - keep original
            result.append(violation)
    return result


def cap_findings(
    violations: Sequence[ContractViolation],
    max_per_rule: int | None,
) -> list[ContractViolation]:
    """Cap the number of findings per rule/table.

    Note: This is a specialized version that works with typed
    ``ContractViolation`` objects rather than generic dicts.

    Parameters
    ----------
    violations
        Violations to cap.
    max_per_rule
        Maximum per rule (None for unlimited).

    Returns
    -------
    list[ContractViolation]
        Capped violations.
    """
    if max_per_rule is None:
        return list(violations)

    counts: dict[str, int] = {}
    result: list[ContractViolation] = []

    for violation in violations:
        key = violation.contract.table
        current = counts.get(key, 0)
        if current < max_per_rule:
            result.append(violation)
            counts[key] = current + 1

    return result


def has_error_findings(violations: Sequence[ContractViolation]) -> bool:
    """Check if any violations have error severity.

    Note: This is a specialized version that works with typed
    ``ContractViolation`` objects rather than generic dicts.

    Parameters
    ----------
    violations
        Violations to check.

    Returns
    -------
    bool
        True if any error-level violations exist.
    """
    return any(v.severity == "error" for v in violations)


__all__ = [
    # Constants
    "MAX_FINDINGS_PER_RULE",
    "MIN_ROW_THRESHOLD",
    "SAMPLE_LIMIT",
    # Types
    "ColumnConstraint",
    "ContractValidationResult",
    "ContractViolation",
    "ForeignKeyConstraint",
    "IngestContractSpec",
    "IngestValidationOptions",
    "ValidationSeverity",
    # Functions
    "apply_severity_overrides",
    "cap_findings",
    "has_error_findings",
]
