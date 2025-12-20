"""Build-engine version utilities for cache invalidation.

The build system uses content-addressed hashes for incremental execution. Those hashes must change
when the build engine changes (even if the analyzed repo/commit does not), otherwise manifests can
be incorrectly reused across incompatible versions of the pipeline.
"""

from __future__ import annotations

from codeintel.core.config.settings import BuildSettings


def get_build_engine_version(settings: BuildSettings) -> str:
    """Return the build-engine version string from settings.

    Parameters
    ----------
    settings
        Build settings.

    Returns
    -------
    str
        Build engine version identifier.
    """
    return settings.engine_version


__all__ = ["get_build_engine_version"]
