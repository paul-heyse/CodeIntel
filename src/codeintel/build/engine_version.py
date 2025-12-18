"""Build-engine version utilities for cache invalidation.

The build system uses content-addressed hashes for incremental execution. Those hashes must change
when the build engine changes (even if the analyzed repo/commit does not), otherwise manifests can
be incorrectly reused across incompatible versions of the pipeline.
"""

from __future__ import annotations

import os
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version


@lru_cache(maxsize=1)
def get_build_engine_version() -> str:
    """Return a stable build-engine version string.

    Resolution order:
    1) `CODEINTEL_BUILD_ENGINE_VERSION` environment override (recommended for dev workflows)
    2) Installed package version for the `codeintel` distribution
    3) "unknown" fallback

    Returns
    -------
    str
        Build-engine version string used to salt cache keys.
    """
    override = os.environ.get("CODEINTEL_BUILD_ENGINE_VERSION", "").strip()
    if override:
        return override
    try:
        return version("codeintel")
    except PackageNotFoundError:
        return "unknown"


__all__ = ["get_build_engine_version"]
