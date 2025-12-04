"""Validation utilities for dataset contracts and data conformance.

This package provides utilities for validating:

- validation.contract: Dataset contract consistency checks
- validation.conformance: Row-level schema conformance validation
- validation.data_checks: Efficient data existence checking
"""

from __future__ import annotations

from codeintel.storage.validation.conformance import (
    ConformanceIssue,
    ConformanceReport,
    run_conformance,
)
from codeintel.storage.validation.contract import (
    BINDING_REQUIRED_DATASETS,
    _schema_path,
    collect_contract_issues,
    validate_contract_or_raise,
)
from codeintel.storage.validation.data_checks import table_has_rows_for_snapshot

__all__ = [
    "BINDING_REQUIRED_DATASETS",
    "ConformanceIssue",
    "ConformanceReport",
    "_schema_path",
    "collect_contract_issues",
    "run_conformance",
    "table_has_rows_for_snapshot",
    "validate_contract_or_raise",
]
