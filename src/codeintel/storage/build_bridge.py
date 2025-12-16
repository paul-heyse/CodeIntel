"""Build integration shim for the storage layer.

This module centralizes storage's runtime dependencies on ``codeintel.build.*``.
Storage code should import build-owned contract/schema helpers exclusively from
this module. Doing so reduces churn during build refactors while keeping
runtime behavior identical.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from codeintel.build.hamilton.contracts.schemas.validation import validate_df
    from codeintel.build.schemas import get_schema_provider, is_view, iter_contracts

__all__ = [
    "get_schema_provider",
    "is_view",
    "iter_contracts",
    "validate_df",
]

_LAZY_IMPORTS: dict[str, str] = {
    "validate_df": "codeintel.build.hamilton.contracts.schemas.validation",
    "get_schema_provider": "codeintel.build.schemas",
    "is_view": "codeintel.build.schemas",
    "iter_contracts": "codeintel.build.schemas",
}


def __getattr__(name: str) -> object:
    module_path = _LAZY_IMPORTS.get(name)
    if module_path is None:
        msg = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(msg)
    module = importlib.import_module(module_path)
    return getattr(module, name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_IMPORTS))
