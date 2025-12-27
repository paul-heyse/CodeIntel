"""Cached Hamilton tag filter queries."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hamilton.driver import Driver


def _freeze_value(value: object) -> object:
    if isinstance(value, list):
        return tuple(_freeze_value(item) for item in value)
    if isinstance(value, dict):
        return tuple(sorted((str(key), _freeze_value(val)) for key, val in value.items()))
    return value


def _freeze_filter(tag_filter: Mapping[str, object]) -> tuple[tuple[str, object], ...]:
    return tuple(
        sorted((str(key), _freeze_value(value)) for key, value in tag_filter.items())
    )


class TagQuery:
    """Cached tag-filter queries against a Hamilton Driver."""

    def __init__(self, dr: Driver) -> None:
        self._dr = dr
        self._cache: dict[tuple[tuple[str, object], ...], tuple[object, ...]] = {}

    def query(self, tag_filter: Mapping[str, object]) -> tuple[object, ...]:
        """Return available variables matching tag_filter (cached).

        Returns
        -------
        tuple[object, ...]
            Cached Hamilton variables matching the tag filter.
        """
        key = _freeze_filter(tag_filter)
        if key not in self._cache:
            self._cache[key] = tuple(
                self._dr.list_available_variables(tag_filter=dict(tag_filter))
            )
        return self._cache[key]


__all__ = ["TagQuery"]
