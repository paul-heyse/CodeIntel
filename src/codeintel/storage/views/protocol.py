"""Protocols for SQLGlot-based view builders.

These protocols define the minimal interface required by view builder functions.
They intentionally avoid coupling to the concrete gateway implementation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from sqlglot import exp


class ViewBuilder(Protocol):
    """Protocol for view builder functions."""

    def __call__(self) -> exp.Expression:
        """Build and return a SQLGlot expression for a view."""
        ...


__all__ = ["ViewBuilder"]
