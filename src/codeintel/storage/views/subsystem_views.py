"""Subsystem docs views.

View definitions are now managed via Ibis in ibis_views.py.
This module only exports the view name constants for compatibility.
"""

from __future__ import annotations

SUBSYSTEM_VIEW_NAMES: tuple[str, ...] = (
    "docs.v_subsystem_summary",
    "docs.v_module_with_subsystem",
    "docs.v_subsystem_agreement",
    "docs.v_subsystem_profile",
    "docs.v_subsystem_coverage",
)

__all__ = ["SUBSYSTEM_VIEW_NAMES"]
