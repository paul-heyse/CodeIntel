"""Compatibility re-export for the semantic HTTP router.

New code should import the versioned router from
``codeintel.serving.http.routes.v1.semantic``.
"""

from __future__ import annotations

from codeintel.serving.http.routes.v1.semantic import router

__all__ = ["router"]
