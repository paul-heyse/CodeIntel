"""Test profiles types and helpers - re-exports from testing/.

This module provides backward-compatible imports. New code should import
directly from ``codeintel.analytics.testing.profiles``.
"""

from __future__ import annotations

from codeintel.analytics.testing.profiles.types import (
    BehavioralContext,
    BehavioralLLMRequest,
    BehavioralLLMResult,
    BehavioralLLMRunner,
    ImportanceInputs,
    IoFlags,
    TestAstInfo,
    TestProfileContext,
    TestRecord,
)

__all__ = [
    "BehavioralContext",
    "BehavioralLLMRequest",
    "BehavioralLLMResult",
    "BehavioralLLMRunner",
    "ImportanceInputs",
    "IoFlags",
    "TestAstInfo",
    "TestProfileContext",
    "TestRecord",
]
