"""Re-export structural helpers from build.graphs.compute.metrics."""

from __future__ import annotations

from codeintel.build.graphs.compute.metrics.structural import (
    bounded_simple_path_count,
    structural_metrics,
)

__all__ = [
    "bounded_simple_path_count",
    "structural_metrics",
]
