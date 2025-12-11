"""Database integrity checks for ingestion validation.

This module provides checks for database-level integrity such as
row counts, column presence, and foreign key relationships.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from codeintel.ingestion.infrastructure.db_queries import (
    ForeignKeyRef,
    safe_count,
    safe_count_orphan_refs,
    safe_count_with_scope,
    safe_get_columns,
    safe_table_exists,
)
from codeintel.ingestion.validation.checks.constraints import (
    ConstraintCheckerContext,
    get_constraint_checker,
)
from codeintel.ingestion.validation.findings import (
    ContractViolation,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from codeintel.config.primitives import SnapshotRef
    from codeintel.ingestion.validation.findings import (
        ForeignKeyConstraint,
        IngestContractSpec,
    )
    from codeintel.storage.gateway import StorageGateway

log = logging.getLogger(__name__)


def check_table_exists(
    gateway: StorageGateway,
    contract: IngestContractSpec,
) -> ContractViolation | None:
    """Check if a table exists.

    Parameters
    ----------
    gateway
        Storage gateway.
    contract
        Contract to check.

    Returns
    -------
    ContractViolation | None
        Violation if table doesn't exist, None otherwise.
    """
    if not safe_table_exists(gateway, contract.table):
        return ContractViolation(
            contract=contract,
            message=f"Table {contract.table} does not exist",
            severity=contract.severity,
        )
    return None


def check_row_count(
    gateway: StorageGateway,
    contract: IngestContractSpec,
    snapshot: SnapshotRef,
) -> list[ContractViolation]:
    """Check row count constraints.

    Parameters
    ----------
    gateway
        Storage gateway.
    contract
        Contract to check.
    snapshot
        Snapshot for scoping queries.

    Returns
    -------
    list[ContractViolation]
        Violations found.
    """
    violations: list[ContractViolation] = []

    # Get row count (scoped if possible)
    row_count = safe_count_with_scope(gateway, contract.table, snapshot)
    if row_count is None:
        count = safe_count(gateway, contract.table)
        row_count = count if count is not None else 0

    # Check minimum rows
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

    # Check maximum rows
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

    return violations


def check_required_columns(
    gateway: StorageGateway,
    contract: IngestContractSpec,
) -> list[ContractViolation]:
    """Check that required columns exist.

    Parameters
    ----------
    gateway
        Storage gateway.
    contract
        Contract to check.

    Returns
    -------
    list[ContractViolation]
        Violations for missing columns.
    """
    existing = safe_get_columns(gateway, contract.table)
    missing = [col for col in contract.required_columns if col not in existing]

    return [
        ContractViolation(
            contract=contract,
            message=f"Table {contract.table} missing required column: {col}",
            severity=contract.severity,
        )
        for col in missing
    ]


def check_column_constraints(
    gateway: StorageGateway,
    contract: IngestContractSpec,
) -> list[ContractViolation]:
    """Check column constraints.

    Parameters
    ----------
    gateway
        Storage gateway.
    contract
        Contract with column constraints.

    Returns
    -------
    list[ContractViolation]
        Violations found.
    """
    violations: list[ContractViolation] = []

    for constraint in contract.column_constraints:
        checker = get_constraint_checker(constraint.constraint_type)
        if checker is None:
            log.debug(
                "No checker registered for constraint type: %s",
                constraint.constraint_type,
            )
            continue

        ctx = ConstraintCheckerContext(
            contract=contract,
            constraint=constraint,
            table=contract.table,
            column=constraint.column,
            gateway=gateway,
        )
        violation = checker(ctx)
        if violation is not None:
            violations.append(violation)

    return violations


def check_foreign_keys(
    gateway: StorageGateway,
    contract: IngestContractSpec,
) -> list[ContractViolation]:
    """Check foreign key constraints.

    Parameters
    ----------
    gateway
        Storage gateway.
    contract
        Contract with foreign key constraints.

    Returns
    -------
    list[ContractViolation]
        Violations found.
    """
    violations: list[ContractViolation] = []

    for fk in contract.foreign_keys:
        violation = _check_single_foreign_key(gateway, contract, fk)
        if violation is not None:
            violations.append(violation)

    return violations


def _check_single_foreign_key(
    gateway: StorageGateway,
    contract: IngestContractSpec,
    fk: ForeignKeyConstraint,
) -> ContractViolation | None:
    """Check a single foreign key constraint.

    Parameters
    ----------
    gateway
        Storage gateway.
    contract
        Contract being validated.
    fk
        Foreign key constraint.

    Returns
    -------
    ContractViolation | None
        Violation if orphan references found.
    """
    fk_ref = ForeignKeyRef(
        source_table=contract.table,
        source_column=fk.column,
        ref_table=fk.reference_table,
        ref_column=fk.reference_column,
        allow_null=fk.allow_null,
    )
    orphan_count = safe_count_orphan_refs(gateway, fk_ref)

    if orphan_count > 0:
        return ContractViolation(
            contract=contract,
            message=(
                f"Foreign key {contract.table}.{fk.column} -> "
                f"{fk.reference_table}.{fk.reference_column} has "
                f"{orphan_count} orphaned references"
            ),
            severity=contract.severity,
            details={"orphan_count": orphan_count},
        )
    return None


def validate_contract(
    gateway: StorageGateway,
    contract: IngestContractSpec,
    snapshot: SnapshotRef,
) -> list[ContractViolation]:
    """Validate a single contract against database state.

    Parameters
    ----------
    gateway
        Storage gateway.
    contract
        Contract to validate.
    snapshot
        Snapshot for scoping queries.

    Returns
    -------
    list[ContractViolation]
        All violations found.
    """
    violations: list[ContractViolation] = []

    # Check table exists
    table_violation = check_table_exists(gateway, contract)
    if table_violation is not None:
        return [table_violation]

    # Get row count for skip_if_empty check
    row_count = safe_count_with_scope(gateway, contract.table, snapshot)
    if row_count is None:
        count = safe_count(gateway, contract.table)
        row_count = count if count is not None else 0

    if contract.skip_if_empty and row_count == 0:
        return violations

    # Check row count constraints
    violations.extend(check_row_count(gateway, contract, snapshot))

    # Check required columns
    violations.extend(check_required_columns(gateway, contract))

    # Check column constraints
    violations.extend(check_column_constraints(gateway, contract))

    # Check foreign keys
    violations.extend(check_foreign_keys(gateway, contract))

    return violations


def validate_contracts(
    gateway: StorageGateway,
    contracts: Sequence[IngestContractSpec],
    snapshot: SnapshotRef,
) -> list[ContractViolation]:
    """Validate multiple contracts.

    Parameters
    ----------
    gateway
        Storage gateway.
    contracts
        Contracts to validate.
    snapshot
        Snapshot for scoping queries.

    Returns
    -------
    list[ContractViolation]
        All violations found across all contracts.
    """
    all_violations: list[ContractViolation] = []
    for contract in contracts:
        all_violations.extend(validate_contract(gateway, contract, snapshot))
    return all_violations


__all__ = [
    "check_column_constraints",
    "check_foreign_keys",
    "check_required_columns",
    "check_row_count",
    "check_table_exists",
    "validate_contract",
    "validate_contracts",
]
