"""Test profile building and shared types.

This subpackage provides:

- ``types``: Shared type definitions (TestRecord, IoFlags, etc.)
- ``builder``: Profile assembly and behavioral coverage building
- ``rows``: Row assembly helpers

Note: To avoid circular imports, ``builder`` and ``rows`` modules are not
re-exported here. Import them directly:

    from codeintel.build.analytics.testing.profiles.builder import build_test_profile_result
    from codeintel.build.analytics.testing.profiles.rows import build_test_profile_rows
"""

from __future__ import annotations

from codeintel.build.analytics.testing.profiles.types import (
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
