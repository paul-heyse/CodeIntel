"""Unified result types for CodeIntel.

This module provides the canonical result protocol and base types
used across all modules for consistent success/failure/skip semantics.

Examples
--------
>>> from codeintel.core.results import BaseResult, ResultStatus
>>> result = BaseResult.ok(duration_s=1.5)
>>> result.success
True
>>> result.status
<ResultStatus.SUCCEEDED: 'succeeded'>
"""

from __future__ import annotations

from codeintel.core.results.base import BaseResult, ResultStatus
from codeintel.core.results.execution import ExecutionResult
from codeintel.core.results.protocol import ResultProtocol

__all__ = [
    "BaseResult",
    "ExecutionResult",
    "ResultProtocol",
    "ResultStatus",
]
