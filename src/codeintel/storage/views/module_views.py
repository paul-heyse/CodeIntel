"""Module- and file-level docs views.

View definitions are now managed via Ibis in ibis_views.py.
This module only exports the view name constants for compatibility.
"""

from __future__ import annotations

MODULE_VIEW_NAMES: tuple[str, ...] = (
    "docs.v_module_history_timeseries",
    "docs.v_module_architecture",
    "docs.v_file_summary",
    "docs.v_entrypoints",
    "docs.v_external_dependencies",
    "docs.v_external_dependency_calls",
)

__all__ = ["MODULE_VIEW_NAMES"]
