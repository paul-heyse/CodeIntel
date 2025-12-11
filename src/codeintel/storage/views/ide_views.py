"""IDE-facing docs views.

View definitions are now managed via Ibis in ibis_views.py.
This module only exports the view name constants for compatibility.
"""

from __future__ import annotations

IDE_VIEW_NAMES: tuple[str, ...] = ("docs.v_ide_hints",)

__all__ = ["IDE_VIEW_NAMES"]
