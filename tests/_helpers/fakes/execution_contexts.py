"""Deprecated shim - re-export unified execution context builders."""

from __future__ import annotations

from tests._helpers.fakes.contexts import (
    ExecutionContextBuilder as TestExecutionContextBuilder,
)
from tests._helpers.fakes.contexts import (
    build_plugin_execution_context as create_test_execution_context,
)

__all__ = [
    "TestExecutionContextBuilder",
    "create_test_execution_context",
]
