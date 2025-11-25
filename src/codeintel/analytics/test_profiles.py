"""Compatibility wrapper for test profile analytics."""

from __future__ import annotations

from codeintel.analytics.tests.profiles import (
    BehavioralLLMRequest,
    BehavioralLLMResult,
    BehavioralProfile,
    FunctionCoverageEntry,
    ImportanceInputs,
    IoFlags,
    SubsystemCoverageEntry,
    TestAstInfo,
    TestGraphMetrics,
    TestProfileContext,
    TestRecord,
    build_behavioral_coverage,
    build_test_profile,
    compute_flakiness_score,
    compute_importance_score,
    infer_behavior_tags,
)

__all__ = [
    "BehavioralLLMRequest",
    "BehavioralLLMResult",
    "BehavioralProfile",
    "FunctionCoverageEntry",
    "ImportanceInputs",
    "IoFlags",
    "SubsystemCoverageEntry",
    "TestAstInfo",
    "TestGraphMetrics",
    "TestProfileContext",
    "TestRecord",
    "build_behavioral_coverage",
    "build_test_profile",
    "compute_flakiness_score",
    "compute_importance_score",
    "infer_behavior_tags",
]
