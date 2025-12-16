"""Compatibility re-export for the search HTTP router.

New code should import the versioned router from
``codeintel.serving.http.routes.v1.search``.
"""

from __future__ import annotations

from codeintel.serving.http.routes.v1.search import router

__all__ = ["router"]
