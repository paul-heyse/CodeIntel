"""Runtime Hamilton graph validation helpers."""

from __future__ import annotations

from codeintel.build.hamilton.driver_factory import build_driver
from codeintel.build.hamilton.validate import (
    GraphValidationResult,
    validate_nodes,
    validation_result_to_json,
)


def validate_graph() -> GraphValidationResult:
    """Validate the Hamilton graph for build invariants.

    Returns
    -------
    GraphValidationResult
        Validation result for the constructed graph.
    """
    runtime = build_driver()
    return validate_nodes(
        runtime.dr.graph.nodes,
        enforce_compute_io_purity=True,
    )


__all__ = [
    "validate_graph",
    "validation_result_to_json",
]
