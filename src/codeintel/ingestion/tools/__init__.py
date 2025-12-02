"""Plugins and registry for ingestion tool executions."""

from __future__ import annotations

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
    "ToolPlugin",
    "ToolPluginMetadata",
    "ToolPluginRegistry",
    "ToolPluginResult",
    "ToolStatus",
    "build_default_registry",
]
