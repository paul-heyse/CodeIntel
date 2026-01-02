"""Column operation modules for with_columns subDAGs."""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from types import ModuleType

_FEATURE_MODULES: Mapping[str, ModuleType] = {}


def allowed_ops_by_table() -> dict[str, set[str]]:
    """Return allowed column operation names for each feature module.

    Returns
    -------
    dict[str, set[str]]
        Mapping of table keys to allowed operation names.
    """
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
]
