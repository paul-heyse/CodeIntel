"""Protocols for view builder callables.

These protocols define the minimal interface required by view builder functions.
They intentionally avoid coupling to the concrete gateway implementation.
"""

from __future__ import annotations

from typing import Protocol


class ViewBuilder(Protocol):
    """Protocol for view builder functions."""

    def __call__(self) -> str:
        """Return a SQL string for a view."""
        ...


__all__ = ["ViewBuilder"]
