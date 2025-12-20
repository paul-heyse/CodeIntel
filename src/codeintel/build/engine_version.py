"""Build-engine version utilities for cache invalidation.

The build system uses content-addressed hashes for incremental execution. Those hashes must change
when the build engine changes (even if the analyzed repo/commit does not), otherwise manifests can
be incorrectly reused across incompatible versions of the pipeline.
"""

from __future__ import annotations

from codeintel.build.settings import BuildSettings, get_build_settings


def get_build_engine_version(settings: BuildSettings | None = None) -> str:
    """Return the build-engine version string from settings."""
    resolved = get_build_settings() if settings is None else settings
    return resolved.engine_version


__all__ = ["get_build_engine_version"]
