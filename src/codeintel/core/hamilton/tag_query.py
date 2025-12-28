"""Cached Hamilton tag filter queries."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
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


def _normalize_bool_like(value: object) -> object:
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "false"}:
            return lowered == "true"
    return value


def _normalize_value(value: object) -> object:
    if isinstance(value, list):
        return [_normalize_bool_like(item) for item in value]
    return _normalize_bool_like(value)


def _hamilton_filter_value(value: object) -> bool:
    if isinstance(value, str):
        return True
    if isinstance(value, list):
        return all(isinstance(item, str) for item in value)
    return False


def _split_tag_filter(
    tag_filter: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    hamilton_filter: dict[str, object] = {}
    post_filter: dict[str, object] = {}
    for key, value in tag_filter.items():
        if _hamilton_filter_value(value):
            hamilton_filter[str(key)] = value
        else:
            post_filter[str(key)] = value
    return hamilton_filter, post_filter


def _value_matches(actual: object, expected: object) -> bool:
    normalized_actual = _normalize_value(actual)
    normalized_expected = _normalize_value(expected)
    if isinstance(normalized_expected, list):
        if isinstance(normalized_actual, list):
            return any(item in normalized_expected for item in normalized_actual)
        return normalized_actual in normalized_expected
    if isinstance(normalized_actual, list):
        return normalized_expected in normalized_actual
    return normalized_actual == normalized_expected


def _matches_post_filter(variable: object, post_filter: Mapping[str, object]) -> bool:
    if not post_filter:
        return True
    tags = getattr(variable, "tags", None)
    if not isinstance(tags, dict):
        return False
    for key, expected in post_filter.items():
        if expected is None:
            if key not in tags:
                return False
            continue
        actual = tags.get(key)
        if not _value_matches(actual, expected):
            return False
    return True


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
            hamilton_filter, post_filter = _split_tag_filter(tag_filter)
            if hamilton_filter:
                variables: Iterable[object] = self._dr.list_available_variables(
                    tag_filter=dict(hamilton_filter)
                )
            else:
                variables = self._dr.list_available_variables()
            if post_filter:
                variables = (
                    variable
                    for variable in variables
                    if _matches_post_filter(variable, post_filter)
                )
            self._cache[key] = tuple(variables)
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
