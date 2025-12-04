"""Shared type definitions - re-exports from testing/.

This module provides backward-compatible imports. New code should import
directly from ``codeintel.analytics.testing.profiles.types``.
"""

from __future__ import annotations

from codeintel.analytics.testing.profiles.types import (
    BehavioralContext,
    BehavioralLLMRequest,
    BehavioralLLMResult,
    BehavioralLLMRunner,
    FunctionCoverageEntryProtocol,
    ImportanceInputs,
    IoFlags,
    SubsystemCoverageEntryProtocol,
    TestAstInfo,
    TestGraphMetricsProtocol,
    TestProfileContext,
    TestRecord,
)

__all__ = [
    "BehavioralContext",
    "BehavioralLLMRequest",
    "BehavioralLLMResult",
    "BehavioralLLMRunner",
    "FunctionCoverageEntryProtocol",
    "ImportanceInputs",
    "IoFlags",
    "SubsystemCoverageEntryProtocol",
    "TestAstInfo",
    "TestGraphMetricsProtocol",
    "TestProfileContext",
    "TestRecord",
]
