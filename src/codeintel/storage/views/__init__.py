"""Docs view registry and creation helpers.

This module provides Ibis-based view creation using the VIEW_BUILDERS registry.
All views are now defined as Ibis expressions in ibis_views.py.

The legacy SQL-based view creation functions have been removed in favor of
the unified Ibis approach.

Note
----
View name constants (ALIAS_DOCS_VIEWS, DERIVED_DOCS_VIEWS, DOCS_VIEWS) are
in codeintel.storage.view_names to avoid circular imports. Import from
there instead of this module.

For create_all_views, import from codeintel.storage.views.creation to avoid
circular imports when config.datasets imports from storage.view_names.
"""

from __future__ import annotations

import codeintel.storage.views.ibis_views as _ibis_views
from codeintel.storage.views.ibis_registry import (
    VIEW_BUILDERS,
    ViewBuilder,
    get_registered_views,
)

__all__ = [
    "VIEW_BUILDERS",
    "ViewBuilder",
    "_ibis_views",
    "get_registered_views",
]
