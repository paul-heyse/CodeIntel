"""Composite schema definitions for profile table composition metadata."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from codeintel.config.datasets.declared_schemas import COMPOSITE_SCHEMAS as _COMPOSITE_SCHEMAS

if TYPE_CHECKING:
    from codeintel.config.datasets.primitives import CompositeSchema

COMPOSITE_SCHEMAS: Final[dict[str, CompositeSchema]] = _COMPOSITE_SCHEMAS


def get_composite_schemas() -> dict[str, CompositeSchema]:
    """Return the COMPOSITE_SCHEMAS dictionary.

    Returns
    -------
    dict[str, CompositeSchema]
        Mapping of profile table keys to composition metadata.
    """
    return COMPOSITE_SCHEMAS


__all__ = [
    "COMPOSITE_SCHEMAS",
    "get_composite_schemas",
]
