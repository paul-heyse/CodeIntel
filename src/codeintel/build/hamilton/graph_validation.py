"""Runtime Hamilton graph validation helpers."""

from __future__ import annotations

from codeintel.build.hamilton.validate import (
    GraphValidationResult,
    validate_nodes,
    validation_result_to_json,
)
from codeintel.runtime.runtime_bundle import RuntimeBundle


def validate_graph(
    *,
    runtime: RuntimeBundle,
    validate_schema: bool = True,
) -> GraphValidationResult:
    """Validate the Hamilton graph for build invariants.

    Parameters
    ----------
    runtime
        Runtime bundle containing the Hamilton driver and catalog.
    validate_schema
        Whether to validate output table schemas using the schema provider.

    Returns
    -------
    GraphValidationResult
        Validation result for the constructed graph.
    """
    return validate_nodes(
        runtime.dr.graph.nodes,
        validate_schema=validate_schema,
        enforce_compute_io_purity=True,
        module_provenance=runtime.module_provenance,
    )


__all__ = [
    "validate_graph",
    "validation_result_to_json",
]
