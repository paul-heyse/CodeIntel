"""Row assembly - re-exports from testing/.

This module provides backward-compatible imports. New code should import
directly from ``codeintel.analytics.testing.profiles.rows``.

Note: This module provides wrapper functions for write_test_profile_rows
and write_behavioral_coverage_rows that use this module's namespace for
ensure_schema and prepared_statements_dynamic. This supports legacy test
patterns that override these functions at the module level.
"""

from __future__ import annotations

import sys
from collections.abc import Iterable
from typing import cast

from codeintel.analytics.profiles.writer_guard import (
    SerializeRow,
    WriterContext,
    write_rows_with_registry_guard,
)
from codeintel.analytics.testing.coverage.inputs import (
    FunctionCoverageEntry,
    SubsystemCoverageEntry,
    TestGraphMetrics,
)
from codeintel.analytics.testing.profiles.rows import (
    BehavioralCoverageRowModel,
    build_behavioral_coverage_rows,
    build_test_profile_context,
    build_test_profile_rows,
)
from codeintel.analytics.testing.profiles.types import (
    FunctionCoverageEntryProtocol,
    ImportanceInputs,
    SubsystemCoverageEntryProtocol,
    TestAstInfo,
    TestGraphMetricsProtocol,
    TestProfileContext,
    TestRecord,
)
from codeintel.config import BehavioralCoverageStepConfig, TestProfileStepConfig
from codeintel.config.datasets import (
    BEHAVIORAL_COVERAGE_COLUMNS,
    TEST_PROFILE_COLUMNS,
    ProfileRowModel,
    behavioral_coverage_row_to_tuple,
    serialize_test_profile_row,
)
from codeintel.storage.gateway import StorageGateway
from codeintel.storage.sql_helpers import (
    ensure_schema as _ensure_schema,
)
from codeintel.storage.sql_helpers import (
    prepared_statements_dynamic as _prepared_statements_dynamic,
)

# Module-level references that can be overridden by tests
ensure_schema = _ensure_schema
prepared_statements_dynamic = _prepared_statements_dynamic


def write_test_profile_rows(
    gateway: StorageGateway,
    cfg: TestProfileStepConfig,
    rows: Iterable[ProfileRowModel],
) -> int:
    """Insert rows into analytics.test_profile with registry alignment checks.

    This wrapper uses module-level ensure_schema and prepared_statements_dynamic
    to support legacy test patterns that override these at the module level.

    Returns
    -------
    int
        Number of inserted rows.
    """
    # Access this module's namespace to get potentially overridden functions
    this_module = sys.modules[__name__]
    ensure_fn = this_module.ensure_schema
    prepared_fn = this_module.prepared_statements_dynamic

    rows_list = list(rows)
    return write_rows_with_registry_guard(
        gateway.con,
        rows=rows_list,
        context=WriterContext(
            table_key="analytics.test_profile",
            columns=TEST_PROFILE_COLUMNS,
            serialize_row=cast("SerializeRow", serialize_test_profile_row),
            repo=cfg.repo,
            commit=cfg.commit,
            delete_sql="DELETE FROM analytics.test_profile WHERE repo = ? AND commit = ?",
            ensure_schema_fn=ensure_fn,
            prepared_statements_fn=prepared_fn,
        ),
    )


def write_behavioral_coverage_rows(
    gateway: StorageGateway,
    cfg: BehavioralCoverageStepConfig,
    rows: Iterable[BehavioralCoverageRowModel],
) -> int:
    """Insert rows into analytics.behavioral_coverage with registry alignment checks.

    This wrapper uses module-level ensure_schema and prepared_statements_dynamic
    to support legacy test patterns that override these at the module level.

    Returns
    -------
    int
        Number of inserted rows.
    """
    # Access this module's namespace to get potentially overridden functions
    this_module = sys.modules[__name__]
    ensure_fn = this_module.ensure_schema
    prepared_fn = this_module.prepared_statements_dynamic

    rows_list = list(rows)
    return write_rows_with_registry_guard(
        gateway.con,
        rows=rows_list,
        context=WriterContext(
            table_key="analytics.behavioral_coverage",
            columns=BEHAVIORAL_COVERAGE_COLUMNS,
            serialize_row=cast("SerializeRow", behavioral_coverage_row_to_tuple),
            repo=cfg.repo,
            commit=cfg.commit,
            delete_sql="DELETE FROM analytics.behavioral_coverage WHERE repo = ? AND commit = ?",
            ensure_schema_fn=ensure_fn,
            prepared_statements_fn=prepared_fn,
        ),
    )


__all__ = [
    "BehavioralCoverageRowModel",
    "FunctionCoverageEntry",
    "FunctionCoverageEntryProtocol",
    "ImportanceInputs",
    "SubsystemCoverageEntry",
    "SubsystemCoverageEntryProtocol",
    "TestAstInfo",
    "TestGraphMetrics",
    "TestGraphMetricsProtocol",
    "TestProfileContext",
    "TestRecord",
    "build_behavioral_coverage_rows",
    "build_test_profile_context",
    "build_test_profile_rows",
    "ensure_schema",
    "prepared_statements_dynamic",
    "write_behavioral_coverage_rows",
    "write_test_profile_rows",
]
