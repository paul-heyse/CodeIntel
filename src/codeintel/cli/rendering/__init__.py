"""Unified rendering package for CLI output.

This package provides the single source of truth for CLI output rendering:

- ``UnifiedRenderer``: Single renderer for all output
- ``RenderContext``: Context with format, color, and stream settings
- ``OutputFormat``: Output format enum (TEXT, JSON, JSONL)
- ``TableSpec``, ``ColumnSpec``: Table specification types
- Pre-built table specs for common outputs

Examples
--------
>>> from codeintel.cli.rendering import UnifiedRenderer, RenderContext
>>> ctx = RenderContext.auto_detect()
>>> renderer = UnifiedRenderer(ctx)
>>> renderer.render_message("Done!", level="success")
"""

from __future__ import annotations

from codeintel.cli.rendering.service import (
    CODEINTEL_THEME,
    RenderingService,
    UnifiedRenderer,
    get_renderer,
    render_cli_result,
)
from codeintel.cli.rendering.specs import (
    BUILD_TARGETS_TABLE,
    DATASETS_TABLE,
    HEALTH_TABLE,
    JOBS_TABLE,
    OPERATIONS_TABLE,
    SUBSYSTEMS_TABLE,
)
from codeintel.cli.rendering.table import ColumnSpec, TableSpec
from codeintel.cli.rendering.types import JustifyMethod, OutputFormat, RenderContext

__all__ = [
    "BUILD_TARGETS_TABLE",
    "CODEINTEL_THEME",
    "DATASETS_TABLE",
    "HEALTH_TABLE",
    "JOBS_TABLE",
    "OPERATIONS_TABLE",
    "SUBSYSTEMS_TABLE",
    "ColumnSpec",
    "JustifyMethod",
    "OutputFormat",
    "RenderContext",
    "RenderingService",
    "TableSpec",
    "UnifiedRenderer",
    "get_renderer",
    "render_cli_result",
]
