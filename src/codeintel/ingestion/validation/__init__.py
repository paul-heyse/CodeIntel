"""Ingestion validation framework.

This package provides comprehensive validation for ingestion plugin outputs,
including integrity checks, constraint validation, and database checks.
Analogous to graphs/validation/ for structural alignment.

Key Components
--------------
- runner: Main orchestration for running validations
- checks: Individual validation check implementations
- findings: Finding types, severity handling, and results

Example
-------
```python
from codeintel.ingestion.validation import (
    IngestContractValidator,
    IngestValidationOptions,
    row_count_contract,
    run_ingest_validations,
)

# Create contracts
contracts = [
    row_count_contract("core.modules", min_rows=1),
    row_count_contract("core.ast_nodes", min_rows=10),
]

# Run validation
result = run_ingest_validations(gateway, contracts, snapshot=snapshot)
if not result.valid:
    for violation in result.violations:
        print(f"Violation: {violation.message}")
```
"""

from codeintel.ingestion.validation.checks import (
    CONSTRAINT_CHECKERS,
    ConstraintCheckerContext,
    ConstraintCheckerFn,
    check_column_constraints,
    check_foreign_keys,
    check_max_value,
    check_min_fraction_not_null,
    check_min_value,
    check_not_null,
    check_positive,
    check_required_columns,
    check_row_count,
    check_table_exists,
    check_unique,
    get_constraint_checker,
    validate_contract,
    validate_contracts,
)
from codeintel.ingestion.validation.findings import (
    MAX_FINDINGS_PER_RULE,
    MIN_ROW_THRESHOLD,
    SAMPLE_LIMIT,
    ColumnConstraint,
    ContractValidationResult,
    ContractViolation,
    ForeignKeyConstraint,
    IngestContractSpec,
    IngestValidationOptions,
    apply_severity_overrides,
    cap_findings,
    has_error_findings,
)
from codeintel.ingestion.validation.runner import (
    ForeignKeyContractOptions,
    IngestContractValidator,
    foreign_key_contract,
    not_null_contract,
    row_count_contract,
    run_ingest_validations,
)

__all__ = [
    "CONSTRAINT_CHECKERS",
    "MAX_FINDINGS_PER_RULE",
    "MIN_ROW_THRESHOLD",
    "SAMPLE_LIMIT",
    "ColumnConstraint",
    "ConstraintCheckerContext",
    "ConstraintCheckerFn",
    "ContractValidationResult",
    "ContractViolation",
    "ForeignKeyConstraint",
    "ForeignKeyContractOptions",
    "IngestContractSpec",
    "IngestContractValidator",
    "IngestValidationOptions",
    "apply_severity_overrides",
    "cap_findings",
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
    "foreign_key_contract",
    "get_constraint_checker",
    "has_error_findings",
    "not_null_contract",
    "row_count_contract",
    "run_ingest_validations",
    "validate_contract",
    "validate_contracts",
]
