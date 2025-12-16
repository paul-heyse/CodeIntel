"""Route aggregation with API versioning.

The default version (v1) is also mounted at the root for backwards
compatibility.
"""

from __future__ import annotations

from fastapi import APIRouter

from codeintel.serving.http.routes.v1 import router as v1_router

router = APIRouter()

# Versioned routes
router.include_router(v1_router, prefix="/v1", tags=["v1"])

# Root alias for backwards compatibility (same behavior as v1)
router.include_router(v1_router, include_in_schema=False)

__all__ = ["router"]
