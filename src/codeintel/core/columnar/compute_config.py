"""Shared Arrow compute options for consistent behavior."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pyarrow as pa
import pyarrow.compute as pc

from codeintel.core.columnar.profiles import resolve_runtime_profile

if TYPE_CHECKING:
    from codeintel.core.columnar.profiles import RuntimeProfile
    from codeintel.core.config.settings import ColumnarRuntimeSettings

DEFAULT_CAST_SAFE = pc.CastOptions.safe(target_type=pa.string())
DEFAULT_SCALAR_AGG = pc.ScalarAggregateOptions(skip_nulls=True)
DEFAULT_SCALAR_AGG_ALLOW_NULL = pc.ScalarAggregateOptions(skip_nulls=False)
DEFAULT_TAKE = pc.TakeOptions(boundscheck=True)


def resolve_runtime_profile_from_settings(
    settings: ColumnarRuntimeSettings | None,
) -> RuntimeProfile | None:
    """Resolve a runtime profile from columnar settings.

    Returns
    -------
    RuntimeProfile | None
        Resolved runtime profile when configured.
    """
    if settings is None:
        return None
    return resolve_runtime_profile(settings.profile)


__all__ = [
    "DEFAULT_CAST_SAFE",
    "DEFAULT_SCALAR_AGG",
    "DEFAULT_SCALAR_AGG_ALLOW_NULL",
    "DEFAULT_TAKE",
    "resolve_runtime_profile_from_settings",
]
