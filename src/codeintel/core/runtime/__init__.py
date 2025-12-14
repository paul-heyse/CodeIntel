"""Unified runtime types for CodeIntel.

This module provides the canonical runtime protocols and types used
across build, analytics, and CLI modules.

Examples
--------
>>> from codeintel.core.runtime import ExecutorProtocol, StepResult
"""

from __future__ import annotations

from codeintel.core.runtime.protocol import ExecutorProtocol, RuntimeProtocol
from codeintel.core.runtime.tracking import (
    ExecutionTracker,
    StepResult,
    StepStatus,
    TimingContext,
    timed,
)

__all__ = [
    "ExecutionTracker",
    "ExecutorProtocol",
    "RuntimeProtocol",
    "StepResult",
    "StepStatus",
    "TimingContext",
    "timed",
]
