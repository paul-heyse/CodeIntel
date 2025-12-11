"""Function-focused docs views.

View definitions are now managed via Ibis in ibis_views.py.
This module only exports the view name constants for compatibility.
"""

from __future__ import annotations

FUNCTION_VIEW_NAMES: tuple[str, ...] = (
    "docs.v_function_summary",
    "docs.v_function_architecture",
    "docs.v_function_history",
    "docs.v_function_history_timeseries",
    "docs.v_cfg_block_architecture",
    "docs.v_dfg_block_architecture",
)

__all__ = ["FUNCTION_VIEW_NAMES"]
