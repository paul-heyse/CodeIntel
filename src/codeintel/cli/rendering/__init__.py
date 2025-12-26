"""Unified rendering package for CLI output.

This package provides the single source of truth for CLI output rendering:

- ``UnifiedRenderer``: Single renderer for all output
- ``RenderContext``: Context with format, color, and stream settings
- ``OutputFormat``: Output format enum (TEXT, JSON, JSONL)

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
from codeintel.cli.rendering.types import JustifyMethod, OutputFormat, RenderContext

__all__ = [
    "CODEINTEL_THEME",
    "JustifyMethod",
    "OutputFormat",
    "RenderContext",
    "RenderingService",
    "UnifiedRenderer",
    "get_renderer",
    "render_cli_result",
]
