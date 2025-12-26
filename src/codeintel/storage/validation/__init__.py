"""Validation utilities for dataset contracts and data conformance.

This package provides utilities for validating data integrity and checking
data existence in DuckDB tables.

Submodules
----------
validation.contract
    Dataset contract consistency checks. Validates that registered datasets
    have consistent schemas, macros, and bindings.

validation.conformance
    Row-level schema conformance validation. Samples rows from tables and
    validates them against JSON Schemas.

queries.safe
    Efficient data existence and row count checking. Provides functions like
    `table_has_rows_for_snapshot()` and `count_rows_for_tables()` for
    checking whether prerequisite data exists before running pipelines.

Usage
-----
Check if data exists for a snapshot:

    from codeintel.storage.validation import table_has_rows_for_snapshot

    if table_has_rows_for_snapshot(con, contract, repo=repo, commit=commit):

        ...

Count rows across multiple tables:

    from codeintel.storage.validation import count_rows_for_tables

    counts = count_rows_for_tables(con, ["core.goids", "graph.call_graph_edges"],
                                   repo=repo, commit=commit)
"""

from __future__ import annotations

from codeintel.storage.queries.safe import (
    count_rows_for_snapshot,
    count_rows_for_tables,
    safe_count_rows,
    table_has_rows_for_snapshot,
)
from codeintel.storage.validation.conformance import (
    ConformanceIssue,
    ConformanceReport,
    run_conformance,
)
from codeintel.storage.validation.contract import (
    clear_contract_validation_cache,
    collect_contract_issues,
    collect_contract_issues_lenient,
    get_binding_required_datasets,
    validate_contract_or_raise,
)
from codeintel.storage.validation.mode import ContractValidationMode

__all__ = [
    "ConformanceIssue",
    "ConformanceReport",
    "ContractValidationMode",
    "clear_contract_validation_cache",
    "collect_contract_issues",
    "collect_contract_issues_lenient",
    "count_rows_for_snapshot",
    "count_rows_for_tables",
    "get_binding_required_datasets",
    "run_conformance",
    "safe_count_rows",
    "table_has_rows_for_snapshot",
    "validate_contract_or_raise",
]
