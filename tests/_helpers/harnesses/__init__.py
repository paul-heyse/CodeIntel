"""Test harness infrastructure for plugin testing.

This package provides test harnesses for plugins with shared base
classes to reduce code duplication.
"""

from __future__ import annotations

from tests._helpers.harnesses.base import (
    BaseResultAssertions,
    BaseTestHarness,
    ResultLike,
)

__all__ = [
    "BaseResultAssertions",
    "BaseTestHarness",
    "ResultLike",
]
