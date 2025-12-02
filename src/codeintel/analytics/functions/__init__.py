"""Function-level analytics public API.

This module centralizes the main entrypoints for per-function analytics so
callers do not need to import individual implementation modules. Imports are
resolved lazily to avoid circular dependencies during package initialization.

Typedness utilities have been moved to the compute layer at
`codeintel.analytics.compute.functions.typedness`. This module re-exports
them for backward compatibility.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

from codeintel.config import FunctionAnalyticsStepConfig

__all__ = [
    "FunctionAnalyticsOptions",
    "FunctionAnalyticsStepConfig",
    "ParamStats",
    "TypednessFlags",
    "compute_function_contracts",
    "compute_function_effects",
    "compute_function_history",
    "compute_function_metrics_and_types",
    "compute_param_stats",
    "compute_typedness_flags",
]

_LAZY_ATTRS = {
    "FunctionAnalyticsOptions": "codeintel.analytics.functions.config",
    # Typedness utilities now come from compute layer
    "ParamStats": "codeintel.analytics.compute.functions.typedness",
    "TypednessFlags": "codeintel.analytics.compute.functions.typedness",
    "compute_param_stats": "codeintel.analytics.compute.functions.typedness",
    "compute_typedness_flags": "codeintel.analytics.compute.functions.typedness",
    # Domain functions
    "compute_function_contracts": "codeintel.analytics.functions.function_contracts",
    "compute_function_effects": "codeintel.analytics.functions.function_effects",
    "compute_function_history": "codeintel.analytics.functions.function_history",
    "compute_function_metrics_and_types": "codeintel.analytics.functions.metrics",
}

if TYPE_CHECKING:
    from codeintel.analytics.compute.functions.typedness import (
        ParamStats,
        TypednessFlags,
        compute_param_stats,
        compute_typedness_flags,
    )
    from codeintel.analytics.functions.config import FunctionAnalyticsOptions
    from codeintel.analytics.functions.function_contracts import compute_function_contracts
    from codeintel.analytics.functions.function_effects import compute_function_effects
    from codeintel.analytics.functions.function_history import compute_function_history
    from codeintel.analytics.functions.metrics import compute_function_metrics_and_types


def __getattr__(name: str) -> object:
    """Lazily load attributes to avoid circular imports."""
    module_path = _LAZY_ATTRS.get(name)
    if module_path is None:
        message = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(message)
    module = importlib.import_module(module_path)
    value = getattr(module, name)
    globals()[name] = value
    return value
