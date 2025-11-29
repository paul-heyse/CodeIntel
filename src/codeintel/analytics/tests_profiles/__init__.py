"""Composable helpers for test analytics profiles."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

from codeintel.analytics.tests_profiles.types import (
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

if TYPE_CHECKING:
    from codeintel.analytics.tests_profiles import (
        behavioral_tags,
        coverage_inputs,
        importance,
        rows,
    )

_MODULE_EXPORTS = {"behavioral_tags", "coverage_inputs", "importance", "rows"}

__all__ = (
    "BehavioralContext",
    "BehavioralLLMRequest",
    "BehavioralLLMResult",
    "BehavioralLLMRunner",
    "ImportanceInputs",
    "IoFlags",
    "TestAstInfo",
    "TestProfileContext",
    "TestRecord",
    "behavioral_tags",
    "coverage_inputs",
    "importance",
    "rows",
)


def __getattr__(name: str) -> object:
    if name in _MODULE_EXPORTS:
        return import_module(f"{__name__}.{name}")
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
