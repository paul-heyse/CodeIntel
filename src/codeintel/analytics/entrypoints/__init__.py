"""Entrypoint analytics plugins package.

For Hamilton native execution, use the targets under
``codeintel.build.hamilton.native.analytics.dependency_targets``.
"""

from codeintel.analytics.entrypoints.compute import EntrypointsResult
from codeintel.analytics.entrypoints.core import EntrypointBuildInputs

__all__ = [
    "EntrypointBuildInputs",
    "EntrypointsResult",
]
