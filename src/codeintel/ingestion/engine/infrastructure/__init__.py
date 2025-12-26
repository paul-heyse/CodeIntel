"""Internal tool execution infrastructure.

This subpackage provides the low-level execution infrastructure for external
tools, including subprocess management, caching, and environment configuration.

Components
----------
ToolRunner
    Async-capable runner for external tool invocations.
ToolRunResult
    Structured output from a tool invocation.
ToolName
    Enum of supported external tools.
ToolNotFoundError, ToolExecutionError
    Exceptions for tool discovery and execution failures.
"""

from __future__ import annotations

from codeintel.ingestion.engine.infrastructure.runner import (
    ToolExecutionError,
    ToolName,
    ToolNotFoundError,
    ToolRunner,
    ToolRunOptions,
    ToolRunResult,
    ToolSpec,
    ToolSpecError,
)

__all__ = [
    "ToolExecutionError",
    "ToolName",
    "ToolNotFoundError",
    "ToolRunOptions",
    "ToolRunResult",
    "ToolRunner",
    "ToolSpec",
    "ToolSpecError",
]
