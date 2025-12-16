"""Validation reporters shared across analytics domains.

Column definitions and row collection for validation findings.

This module re-exports validation reporters from ``codeintel.core.validation``
for backward compatibility. New code should import directly from core.

For persistence, use the ``to_rows()`` method with Hamilton materializers.

Pure compute helpers are available in ``codeintel.analytics.parsing.compute``:
- ``materialize_function_validation`` for function validation rows
- ``materialize_graph_validation`` for graph validation rows
"""

from __future__ import annotations

# Re-export from core for backward compatibility
from codeintel.core.validation.reporters import (
    FUNCTION_VALIDATION_COLS,
    GRAPH_VALIDATION_COLS,
    BaseValidationReporter,
    FunctionValidationReporter,
    GraphValidationReporter,
    gateway_timestamp,
)

__all__ = [
    "FUNCTION_VALIDATION_COLS",
    "GRAPH_VALIDATION_COLS",
    "BaseValidationReporter",
    "FunctionValidationReporter",
    "GraphValidationReporter",
    "gateway_timestamp",
]
