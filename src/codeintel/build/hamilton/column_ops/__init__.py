"""Column operation modules for with_columns subDAGs."""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from types import ModuleType

from codeintel.build.hamilton.column_ops import function_features, module_features, risk_features

_FEATURE_MODULES: Mapping[str, ModuleType] = {
    "functions": function_features,
    "modules": module_features,
    "risk": risk_features,
}


def allowed_ops_by_table() -> dict[str, set[str]]:
    """Return allowed column operation names for each feature module."""
    allowed: dict[str, set[str]] = {}
    for table_key, module in _FEATURE_MODULES.items():
        ops = {
            name
            for name, value in inspect.getmembers(module)
            if inspect.isfunction(value) and not name.startswith("_")
        }
        allowed[table_key] = ops
    return allowed


__all__ = [
    "allowed_ops_by_table",
    "function_features",
    "module_features",
    "risk_features",
]
