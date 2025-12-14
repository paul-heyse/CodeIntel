"""Entrypoint analytics plugins package.

For Hamilton native execution, use the pure compute function:
- ``compute_entrypoints_pure`` returns ``EntrypointsResult`` without writing

The Hamilton native module is at:
``codeintel.build.hamilton.native.analytics.entrypoints``
"""

from codeintel.analytics.entrypoints.compute import (
    EntrypointsResult,
    compute_entrypoints_pure,
)
from codeintel.analytics.entrypoints.core import EntrypointBuildInputs

__all__ = [
    "EntrypointBuildInputs",
    "EntrypointsResult",
    "compute_entrypoints_pure",
]
