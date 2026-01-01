"""Entrypoint analytics plugins package.

For Hamilton native execution, use the targets under
``codeintel.build.hamilton.native.analytics.entrypoints``.
"""

from codeintel.build.analytics.entrypoints.compute import EntrypointsResult
from codeintel.build.analytics.entrypoints.core import EntrypointBuildInputs

__all__ = [
    "EntrypointBuildInputs",
    "EntrypointsResult",
]
