"""Column constraint validation checks.

This module provides constraint checker implementations for validating
column-level data quality constraints.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from codeintel.ingestion.infrastructure.db_queries import (
    safe_count_duplicates,
    safe_count_non_positive,
    safe_count_nulls,
    safe_max_value,
    safe_min_value,
    safe_not_null_fraction,
)
from codeintel.ingestion.validation.findings import (
    ContractViolation,
)

if TYPE_CHECKING:
    from codeintel.ingestion.validation.findings import (
        ColumnConstraint,
        IngestContractSpec,
    )
    from codeintel.storage.gateway import StorageGateway

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ConstraintCheckerContext:
    """Context for constraint checking operations.

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


# Type alias for constraint checker functions
ConstraintCheckerFn = Callable[[ConstraintCheckerContext], ContractViolation | None]


def check_not_null(ctx: ConstraintCheckerContext) -> ContractViolation | None:
    """Check not_null constraint.

    Parameters
    ----------
    ctx
        Checker context.

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


def check_min_value(ctx: ConstraintCheckerContext) -> ContractViolation | None:
    """Check min_value constraint.

    Parameters
    ----------
    ctx
        Checker context.

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


def check_max_value(ctx: ConstraintCheckerContext) -> ContractViolation | None:
    """Check max_value constraint.

    Parameters
    ----------
    ctx
        Checker context.

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


def check_positive(ctx: ConstraintCheckerContext) -> ContractViolation | None:
    """Check positive constraint.

    Parameters
    ----------
    ctx
        Checker context.

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


def check_unique(ctx: ConstraintCheckerContext) -> ContractViolation | None:
    """Check unique constraint.

    Parameters
    ----------
    ctx
        Checker context.

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


def check_min_fraction_not_null(ctx: ConstraintCheckerContext) -> ContractViolation | None:
    """Check min_fraction_not_null constraint.

    Parameters
    ----------
    ctx
        Checker context.

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
    "not_null": check_not_null,
    "min_value": check_min_value,
    "max_value": check_max_value,
    "positive": check_positive,
    "unique": check_unique,
    "min_fraction_not_null": check_min_fraction_not_null,
}


def get_constraint_checker(constraint_type: str) -> ConstraintCheckerFn | None:
    """Get a constraint checker function by type.

    Parameters
    ----------
    constraint_type
        Type of constraint.

    Returns
    -------
    ConstraintCheckerFn | None
        Checker function or None if not registered.
    """
    return CONSTRAINT_CHECKERS.get(constraint_type)


__all__ = [
    "CONSTRAINT_CHECKERS",
    "ConstraintCheckerContext",
    "ConstraintCheckerFn",
    "check_max_value",
    "check_min_fraction_not_null",
    "check_min_value",
    "check_not_null",
    "check_positive",
    "check_unique",
    "get_constraint_checker",
]
