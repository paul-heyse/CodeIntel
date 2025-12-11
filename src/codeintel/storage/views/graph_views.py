"""Graph-oriented docs views.

View definitions are now managed via Ibis in ibis_views.py.
This module only exports the view name constants for compatibility.
"""

from __future__ import annotations

GRAPH_VIEW_NAMES: tuple[str, ...] = (
    "docs.v_call_graph_enriched",
    "docs.v_symbol_module_graph",
    "docs.v_validation_summary",
)

__all__ = ["GRAPH_VIEW_NAMES"]
