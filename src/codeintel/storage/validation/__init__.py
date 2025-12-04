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

validation.data_checks
    Efficient data existence and row count checking. Provides functions like
    `table_has_rows_for_snapshot()` and `count_rows_for_tables()` for
    checking whether prerequisite data exists before running pipelines.

Usage
-----
Check if data exists for a snapshot:

    from codeintel.storage.validation import table_has_rows_for_snapshot

    if table_has_rows_for_snapshot(con, contract, repo=repo, commit=commit):
        # Data exists, proceed with processing
        ...

Count rows across multiple tables:

    from codeintel.storage.validation import count_rows_for_tables

    counts = count_rows_for_tables(con, ["core.goids", "graph.call_graph_edges"],
                                   repo=repo, commit=commit)
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
from codeintel.storage.validation.data_checks import (
    count_rows_for_snapshot,
    count_rows_for_tables,
    safe_count_rows,
    table_has_rows_for_snapshot,
)

__all__ = [
    "BINDING_REQUIRED_DATASETS",
    "ConformanceIssue",
    "ConformanceReport",
    "_schema_path",
    "collect_contract_issues",
    "count_rows_for_snapshot",
    "count_rows_for_tables",
    "run_conformance",
    "safe_count_rows",
    "table_has_rows_for_snapshot",
    "validate_contract_or_raise",
]
