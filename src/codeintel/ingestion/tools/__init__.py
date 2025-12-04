"""Plugins and registry for ingestion tool executions.

This package provides the tool plugin system for external tool execution,
including plugins for pyright, pyrefly, ruff, coverage, scip, and pytest.

Subpackages
-----------
infrastructure
    Low-level tool execution infrastructure (ToolRunner, ToolRunResult, etc.).
"""

from __future__ import annotations

from codeintel.ingestion.tools.infrastructure import (
    ToolExecutionError,
    ToolName,
    ToolNotFoundError,
    ToolResult,
    ToolRunner,
    ToolRunResult,
)
from codeintel.ingestion.tools.plugins import (
    ToolPlugin,
    ToolPluginMetadata,
    ToolPluginRegistry,
    ToolPluginResult,
    ToolStatus,
    build_default_registry,
)
from codeintel.ingestion.tools.results import (
    CoverageFileSummary,
    CoverageReport,
    DiagnosticReport,
    FileDiagnosticCount,
    ParsedToolResult,
    ScipDocument,
    ScipIndexResult,
    ScipOccurrence,
    TestCaseResult,
    TestReport,
)
from codeintel.ingestion.tools.service import ToolService

__all__ = [
    "CoverageFileSummary",
    "CoverageReport",
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
    "ToolResult",
    "ToolRunResult",
    "ToolRunner",
    "ToolService",
    "ToolStatus",
    "build_default_registry",
]
