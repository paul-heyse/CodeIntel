"""Tool execution engine for ingestion pipelines.

This package provides the tool execution engine for external CLI tools,
analogous to graphs/engine/ which provides graph construction abstractions.

Key Components
--------------
- ToolService: High-level façade for tool execution
- ToolRunner: Low-level command execution
- ToolPlugin: Plugin protocol for tool integrations
- ToolPluginRegistry: Registry for tool plugins

Supported Tools
---------------
- pyright: Type checking and diagnostics
- pyrefly: Additional type checking
- ruff: Linting and formatting
- scip-python: Symbol indexing
- pytest: Test execution

Subpackages
-----------
infrastructure
    Low-level tool execution infrastructure (ToolRunner, ToolRunResult, etc.).
"""

from __future__ import annotations

from codeintel.ingestion.engine.infrastructure import (
    ToolExecutionError,
    ToolName,
    ToolNotFoundError,
    ToolRunResult,
)
from codeintel.ingestion.engine.plugins import (
    ToolPlugin,
    ToolPluginMetadata,
    ToolPluginRegistry,
    ToolPluginResult,
    build_default_registry,
)
from codeintel.ingestion.engine.results import (
    DiagnosticReport,
    FileDiagnosticCount,
    ParsedToolResult,
    ScipDocument,
    ScipIndexResult,
    ScipOccurrence,
    TestCaseResult,
    TestReport,
)
from codeintel.ingestion.engine.service import ToolService
from codeintel.ingestion.engine.status import ToolStatus

__all__ = [
    "DiagnosticReport",
    "FileDiagnosticCount",
    "ParsedToolResult",
    "ScipDocument",
    "ScipIndexResult",
    "ScipOccurrence",
    "TestCaseResult",
    "TestReport",
    "ToolExecutionError",
    "ToolName",
    "ToolNotFoundError",
    "ToolPlugin",
    "ToolPluginMetadata",
    "ToolPluginRegistry",
    "ToolPluginResult",
    "ToolRunResult",
    "ToolService",
    "ToolStatus",
    "build_default_registry",
]
