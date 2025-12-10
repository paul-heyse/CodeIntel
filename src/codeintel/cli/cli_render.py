"""Compatibility shim for cli_render module.

.. deprecated::
    This module is deprecated. Import from ``codeintel.cli.rendering`` instead.
    This shim will be removed in a future version.

Example migration::

    # Old (deprecated):
    from codeintel.cli.cli_render import get_renderer, render_cli_result

    # New (preferred):
    from codeintel.cli.rendering import get_renderer, render_cli_result
"""

from __future__ import annotations

import warnings

warnings.warn(
    "Importing from 'codeintel.cli.cli_render' is deprecated. "
    "Use 'codeintel.cli.rendering' instead. "
    "This compatibility shim will be removed in a future version.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export everything from the canonical location
from codeintel.cli.rendering.renderers import (
    BUILD_TARGET_TABLE_SPEC,
    CODEINTEL_THEME,
    DATASET_TABLE_SPEC,
    OPERATION_TABLE_SPEC,
    OutputRenderer,
    PlainRenderer,
    RenderMode,
    RichRenderer,
    get_renderer,
    render_cli_result,
)
from codeintel.cli.rendering.table import ColumnSpec, TableSpec

__all__ = [
    "BUILD_TARGET_TABLE_SPEC",
    "CODEINTEL_THEME",
    "ColumnSpec",
    "DATASET_TABLE_SPEC",
    "OPERATION_TABLE_SPEC",
    "OutputRenderer",
    "PlainRenderer",
    "RenderMode",
    "RichRenderer",
    "TableSpec",
    "get_renderer",
    "render_cli_result",
]
