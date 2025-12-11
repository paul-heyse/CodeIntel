"""Main orchestration for running ingestion validations.

This module provides the high-level functions for executing the
validation suite and coordinating individual checks.
Analogous to graphs/validation/runner.py.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from codeintel.ingestion.validation.checks.database import (
    validate_contract,
    validate_contracts,
)
from codeintel.ingestion.validation.findings import (
    ColumnConstraint,
    ContractValidationResult,
    ForeignKeyConstraint,
    IngestContractSpec,
    IngestValidationOptions,
    apply_severity_overrides,
    cap_findings,
    has_error_findings,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from codeintel.config.primitives import SnapshotRef
    from codeintel.ingestion.validation.findings import (
        ContractViolation,
    )
    from codeintel.storage.gateway import StorageGateway

log = logging.getLogger(__name__)


class IngestContractValidator:
    """Validator for ingestion plugin output contracts.

    Validates plugin outputs against declared contracts, checking
    row counts, column presence, and column constraints.
    """

    def __init__(
        self,
        gateway: StorageGateway,
        *,
        options: IngestValidationOptions | None = None,
    ) -> None:
        """Initialize the validator.

        Parameters
        ----------
        gateway
            Storage gateway for database access.
        options
            Validation options.
        """
        self._gateway = gateway
        self._options = options or IngestValidationOptions()

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
        tables: set[str] = set()
        for contract in contracts:
            tables.add(contract.table)

        # Run all validations
        violations = validate_contracts(self._gateway, contracts, snapshot)

        # Apply severity overrides
        violations = apply_severity_overrides(violations, self._options.severity_overrides)

        # Cap findings if configured
        violations = cap_findings(violations, self._options.max_findings_per_rule)

        # Separate warnings from errors
        warnings: list[str] = []
        errors: list[ContractViolation] = []
        for violation in violations:
            if violation.severity == "warning":
                warnings.append(violation.message)
            else:
                errors.append(violation)

        if errors:
            return ContractValidationResult.failure(
                errors,
                checked=len(contracts),
                warnings=warnings,
                tables=tuple(sorted(tables)),
            )

        return ContractValidationResult.success(
            checked=len(contracts),
            tables=tuple(sorted(tables)),
        )

    def validate_single(
        self,
        contract: IngestContractSpec,
        snapshot: SnapshotRef,
    ) -> list[ContractViolation]:
        """Validate a single contract.

        Parameters
        ----------
        contract
            Contract to validate.
        snapshot
            Snapshot for scoping queries.

        Returns
        -------
        list[ContractViolation]
            Violations found.
        """
        return validate_contract(self._gateway, contract, snapshot)


def run_ingest_validations(
    gateway: StorageGateway,
    contracts: Sequence[IngestContractSpec],
    *,
    snapshot: SnapshotRef,
    options: IngestValidationOptions | None = None,
) -> ContractValidationResult:
    """Run ingestion validations on a set of contracts.

    This is the main entry point for running ingestion validation.

    Parameters
    ----------
    gateway
        Storage gateway for database access.
    contracts
        Contracts to validate.
    snapshot
        Snapshot for scoping queries.
    options
        Validation options.

    Returns
    -------
    ContractValidationResult
        Validation result.

    Raises
    ------
    RuntimeError
        When hard_fail is enabled and error-level findings are present.
    """
    opts = options or IngestValidationOptions()
    validator = IngestContractValidator(gateway, options=opts)
    result = validator.validate(contracts, snapshot)

    # Log summary
    if result.valid:
        log.info(
            "Validation passed: checked=%d tables=%d",
            result.checked_contracts,
            len(result.tables_checked),
        )
    else:
        log.warning(
            "Validation failed: violations=%d warnings=%d checked=%d",
            len(result.violations),
            len(result.warnings),
            result.checked_contracts,
        )
        for violation in result.violations:
            log.warning("  - %s", violation.message)

    # Hard fail if configured and errors present
    if opts.hard_fail and has_error_findings(result.violations):
        error_count = sum(1 for v in result.violations if v.severity == "error")
        message = f"Validation hard fail: {error_count} error-level violations"
        raise RuntimeError(message)

    return result


# =============================================================================
# Contract Builders (convenience functions)
# =============================================================================


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
class ForeignKeyContractOptions:
    """Options for creating a foreign key contract.

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
    **kwargs: ForeignKeyContractOptions | None,
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
    **kwargs
        Optional configuration via 'options' or 'spec' (deprecated) keyword.

    Returns
    -------
    IngestContractSpec
        Foreign key contract.
    """
    # Support both 'options' and 'spec' (deprecated) keyword args
    opts = kwargs.get("options") or kwargs.get("spec") or ForeignKeyContractOptions()
    return IngestContractSpec(
        table=table,
        plugin_name=opts.plugin_name,
        foreign_keys=(
            ForeignKeyConstraint(
                column=column,
                reference_table=reference_table,
                reference_column=reference_column,
                allow_null=opts.allow_null,
            ),
        ),
        severity=opts.severity,
    )


__all__ = [
    "ForeignKeyContractOptions",
    "IngestContractValidator",
    "foreign_key_contract",
    "not_null_contract",
    "row_count_contract",
    "run_ingest_validations",
]
