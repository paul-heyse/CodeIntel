"""Validation check implementations.

This package contains all the specific validation checks for ingestion
data integrity, constraint validation, and database checks.

Submodules
----------
constraints
    Column constraint checkers (not_null, min_value, unique, etc.).
database
    Database integrity checks (row counts, columns, foreign keys).
"""

from codeintel.ingestion.validation.checks.constraints import (
    CONSTRAINT_CHECKERS,
    ConstraintCheckerContext,
    ConstraintCheckerFn,
    check_max_value,
    check_min_fraction_not_null,
    check_min_value,
    check_not_null,
    check_positive,
    check_unique,
    get_constraint_checker,
)
from codeintel.ingestion.validation.checks.database import (
    check_column_constraints,
    check_foreign_keys,
    check_required_columns,
    check_row_count,
    check_table_exists,
    validate_contract,
    validate_contracts,
)

__all__ = [
    "CONSTRAINT_CHECKERS",
    "ConstraintCheckerContext",
    "ConstraintCheckerFn",
    "check_column_constraints",
    "check_foreign_keys",
    "check_max_value",
    "check_min_fraction_not_null",
    "check_min_value",
    "check_not_null",
    "check_positive",
    "check_required_columns",
    "check_row_count",
    "check_table_exists",
    "check_unique",
    "get_constraint_checker",
    "validate_contract",
    "validate_contracts",
]
