"""Legacy SQLGlot view schema inference (retired)."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

from codeintel.core.schemas.primitives import TableSchema
from codeintel.core.schemas.provider import SchemaProvider

if TYPE_CHECKING:
    from types import ModuleType


def derive_view_schemas(
    *,
    provider: SchemaProvider,
    view_keys: Iterable[str] | None = None,
    modules: tuple[ModuleType, ...] | None = None,
    allow_legacy_sqlglot: bool = False,
) -> dict[str, TableSchema]:
    """Reject SQLGlot view schema inference.

    Parameters
    ----------
    provider
        Schema provider used to supply base table schemas for type inference.
    view_keys
        Optional iterable of view keys to derive.
    modules
        Optional modules to scan for view builders.
    allow_legacy_sqlglot
        Legacy flag retained for compatibility; ignored by this implementation.

    Raises
    ------
    RuntimeError
        Always raised because SQLGlot view inference has been retired.
    """
    _ = (provider, view_keys, modules, allow_legacy_sqlglot)
    msg = (
        "SQLGlot view schema inference has been retired. "
        "Use observed schemas or Hamilton-native view outputs instead."
    )
    raise RuntimeError(msg)


__all__ = ["derive_view_schemas"]
