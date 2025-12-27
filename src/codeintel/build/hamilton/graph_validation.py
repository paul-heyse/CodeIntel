"""Runtime Hamilton graph validation helpers."""

from __future__ import annotations

from codeintel.build.hamilton.validate import (
    GraphValidationResult,
    validate_nodes,
    validation_result_to_json,
)
from codeintel.runtime.runtime_bundle import RuntimeBundle


def validate_graph(*, runtime: RuntimeBundle) -> GraphValidationResult:
    """Validate the Hamilton graph for build invariants.

    Returns
    -------
    GraphValidationResult
        Validation result for the constructed graph.
    """
    return validate_nodes(
        runtime.dr.graph.nodes,
        enforce_compute_io_purity=True,
    )


__all__ = [
    "validate_graph",
    "validation_result_to_json",
]
