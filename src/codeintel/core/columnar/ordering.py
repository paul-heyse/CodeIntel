"""Ordering metadata for Arrow DSL plans."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

SortDirection = Literal["ascending", "descending"]
SortKey = tuple[str, SortDirection]
OrderingLevel = Literal["unordered", "implicit", "explicit"]


@dataclass(frozen=True, slots=True)
class OrderingSpec:
    """Describe the ordering state of a plan output."""

    level: OrderingLevel
    keys: tuple[SortKey, ...] = ()
    pipeline_breaker: bool = False
    reason: str | None = None

    @classmethod
    def unordered(
        cls,
        *,
        reason: str | None = None,
        pipeline_breaker: bool = False,
    ) -> OrderingSpec:
        """Return an unordered ordering spec.

        Returns
        -------
        OrderingSpec
            Ordering spec marked as unordered.
        """
        return cls("unordered", (), pipeline_breaker, reason)

    @classmethod
    def implicit(
        cls,
        *,
        keys: Sequence[SortKey] = (),
        reason: str | None = None,
        pipeline_breaker: bool = False,
    ) -> OrderingSpec:
        """Return an implicit ordering spec.

        Returns
        -------
        OrderingSpec
            Ordering spec marked as implicit.
        """
        return cls("implicit", tuple(keys), pipeline_breaker, reason)

    @classmethod
    def explicit(
        cls,
        *,
        keys: Sequence[SortKey],
        reason: str | None = None,
        pipeline_breaker: bool = False,
    ) -> OrderingSpec:
        """Return an explicit ordering spec.

        Returns
        -------
        OrderingSpec
            Ordering spec marked as explicit.

        Raises
        ------
        ValueError
            Raised when no sort keys are provided.
        """
        if not keys:
            msg = "Explicit ordering requires at least one sort key."
            raise ValueError(msg)
        return cls("explicit", tuple(keys), pipeline_breaker, reason)

    def is_ordered(self) -> bool:
        """Return True when the ordering is explicit.

        Returns
        -------
        bool
            True when the ordering is explicit.
        """
        return self.level == "explicit"


__all__ = [
    "OrderingLevel",
    "OrderingSpec",
    "SortDirection",
    "SortKey",
]
