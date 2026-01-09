"""Finalize gate helper re-exports for build pipelines."""

from __future__ import annotations

from codeintel.core.columnar.finalize_ops import (
    AlignmentReport,
    FinalizeDedupe,
    FinalizeInvariant,
    FinalizeListPolicy,
    FinalizeMode,
    FinalizeResult,
    FinalizeSpec,
    JoinPrecheckReport,
    NullListPolicy,
    drain_join_precheck_reports,
    finalize_join_keys,
    finalize_reader,
    finalize_spec_for_table,
    finalize_table,
    record_join_precheck_errors,
)

__all__ = [
    "AlignmentReport",
    "FinalizeDedupe",
    "FinalizeInvariant",
    "FinalizeListPolicy",
    "FinalizeMode",
    "FinalizeResult",
    "FinalizeSpec",
    "JoinPrecheckReport",
    "NullListPolicy",
    "drain_join_precheck_reports",
    "finalize_join_keys",
    "finalize_reader",
    "finalize_spec_for_table",
    "finalize_table",
    "record_join_precheck_errors",
]
