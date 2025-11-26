"""Function-level analytics public API.

This module centralizes the main entrypoints for per-function analytics so
callers do not need to import individual implementation modules. Imports are
resolved lazily to avoid circular dependencies during package initialization.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

from codeintel.config import FunctionAnalyticsStepConfig

__all__ = [
    "FunctionAnalyticsStepConfig",
    "FunctionAnalyticsOptions",
    "TypednessFlags",
    "compute_function_contracts",
    "compute_function_effects",
    "compute_function_history",
    "compute_function_metrics_and_types",
]

_LAZY_ATTRS = {
    "FunctionAnalyticsOptions": "codeintel.analytics.functions.config",
    "TypednessFlags": "codeintel.analytics.functions.typedness",
    "compute_function_contracts": "codeintel.analytics.functions.function_contracts",
    "compute_function_effects": "codeintel.analytics.functions.function_effects",
    "compute_function_history": "codeintel.analytics.functions.function_history",
    "compute_function_metrics_and_types": "codeintel.analytics.functions.metrics",
}

if TYPE_CHECKING:
    from codeintel.analytics.functions.config import FunctionAnalyticsOptions
    from codeintel.analytics.functions.function_contracts import compute_function_contracts
    from codeintel.analytics.functions.function_effects import compute_function_effects
    from codeintel.analytics.functions.function_history import compute_function_history
    from codeintel.analytics.functions.metrics import compute_function_metrics_and_types
    from codeintel.analytics.functions.typedness import TypednessFlags


def __getattr__(name: str) -> object:
    module_path = _LAZY_ATTRS.get(name)
    if module_path is None:
        message = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(message)
    module = importlib.import_module(module_path)
    value = getattr(module, name)
    globals()[name] = value
    return value
