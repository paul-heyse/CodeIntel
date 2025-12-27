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
    return tuple(sorted((str(key), _freeze_value(value)) for key, value in tag_filter.items()))


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
            self._cache[key] = tuple(self._dr.list_available_variables(tag_filter=dict(tag_filter)))
        return self._cache[key]

    def one(self, tag_filter: Mapping[str, object]) -> object | None:
        """Return the first matching variable or None.

        Parameters
        ----------
        tag_filter
            Tag filter to query via the underlying Driver.

        Returns
        -------
        object | None
            First matching variable or None when no matches exist.
        """
        results = self.query(tag_filter)
        return results[0] if results else None

    def names(self, tag_filter: Mapping[str, object]) -> tuple[str, ...]:
        """Return matching variable names for the tag filter.

        Parameters
        ----------
        tag_filter
            Tag filter to query via the underlying Driver.

        Returns
        -------
        tuple[str, ...]
            Tuple of variable names matching the tag filter.
        """
        return tuple(_variable_name(var) for var in self.query(tag_filter))


def _variable_name(variable: object) -> str:
    name = getattr(variable, "name", None)
    return str(name) if name is not None else str(variable)


__all__ = ["TagQuery"]
