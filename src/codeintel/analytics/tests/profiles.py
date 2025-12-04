"""Test profile building - re-exports from testing/.

This module provides backward-compatible imports. New code should import
directly from ``codeintel.analytics.testing.profiles``.
"""

from __future__ import annotations

from codeintel.analytics.testing.behavioral.importance import (
    compute_flakiness_score,
    compute_importance_score,
)
from codeintel.analytics.testing.coverage.inputs import (
    FunctionCoverageEntry,
    SubsystemCoverageEntry,
    TestGraphMetrics,
)
from codeintel.analytics.testing.profiles.builder import (
    EMPTY_FUNCTION_COVERAGE_ENTRY,
    EMPTY_SUBSYSTEM_ENTRY,
    EMPTY_TEST_METRICS,
    BehavioralProfile,
    SpanConfig,
    SpanState,
    build_behavioral_coverage,
    build_test_ast_index,
    build_test_ast_index_for_tests,
    build_test_profile,
    infer_behavior_tags,
    load_functions_covered,
    load_subsystems_covered,
    load_test_graph_metrics_public,
    load_test_profile_context,
    load_test_records_public,
)
from codeintel.analytics.testing.profiles.rows import (
    build_behavioral_coverage_rows,
    build_test_profile_context,
    build_test_profile_rows,
    write_behavioral_coverage_rows,
    write_test_profile_rows,
)
from codeintel.analytics.testing.profiles.types import (
    BehavioralContext,
    BehavioralLLMRequest,
    BehavioralLLMRunner,
    ImportanceInputs,
    IoFlags,
    TestAstInfo,
    TestProfileContext,
    TestRecord,
)

__all__ = [
    "EMPTY_FUNCTION_COVERAGE_ENTRY",
    "EMPTY_SUBSYSTEM_ENTRY",
    "EMPTY_TEST_METRICS",
    "BehavioralContext",
    "BehavioralLLMRequest",
    "BehavioralLLMRunner",
    "BehavioralProfile",
    "FunctionCoverageEntry",
    "ImportanceInputs",
    "IoFlags",
    "SpanConfig",
    "SpanState",
    "SubsystemCoverageEntry",
    "TestAstInfo",
    "TestGraphMetrics",
    "TestProfileContext",
    "TestRecord",
    "build_behavioral_coverage",
    "build_behavioral_coverage_rows",
    "build_test_ast_index",
    "build_test_ast_index_for_tests",
    "build_test_profile",
    "build_test_profile_context",
    "build_test_profile_rows",
    "compute_flakiness_score",
    "compute_importance_score",
    "infer_behavior_tags",
    "load_functions_covered",
    "load_subsystems_covered",
    "load_test_graph_metrics_public",
    "load_test_profile_context",
    "load_test_records_public",
    "write_behavioral_coverage_rows",
    "write_test_profile_rows",
]
