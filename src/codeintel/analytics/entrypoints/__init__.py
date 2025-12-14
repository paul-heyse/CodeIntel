"""Entrypoint analytics plugins package.

For Hamilton native execution, use the pure compute function:
- `compute_entrypoints_pure` returns `EntrypointsResult` without writing
"""

from codeintel.analytics.entrypoints.compute import (
    EntrypointsResult,
    compute_entrypoints_pure,
)
from codeintel.analytics.entrypoints.core import EntrypointBuildInputs, build_entrypoints

__all__ = [
    "EntrypointBuildInputs",
    "EntrypointsResult",
    "build_entrypoints",
    "compute_entrypoints_pure",
]
