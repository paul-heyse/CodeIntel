"""Docs views for test analytics.

View definitions are now managed via Ibis in ibis_views.py.
This module only exports the view name constants for compatibility.
"""

from __future__ import annotations

TEST_VIEW_NAMES: tuple[str, ...] = (
    "docs.v_test_to_function",
    "docs.v_test_architecture",
    "docs.v_behavioral_classification_input",
)

__all__ = ["TEST_VIEW_NAMES"]
