"""Filter builder utilities.

This module provides utilities for building and composing filters
for repository queries.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Self

if TYPE_CHECKING:
    from collections.abc import Mapping


class FilterOperator(Enum):
    """Supported filter operators.

    Attributes
    ----------
    EQ
        Equals.
    NE
        Not equals.
    GT
        Greater than.
    GTE
        Greater than or equal.
    LT
        Less than.
    LTE
        Less than or equal.
    IN
        In list.
    NOT_IN
        Not in list.
    LIKE
        SQL LIKE pattern match.
    ILIKE
        Case-insensitive LIKE.
    IS_NULL
        Is null check.
    IS_NOT_NULL
        Is not null check.
    """

    EQ = "eq"
    NE = "ne"
    GT = "gt"
    GTE = "gte"
    LT = "lt"
    LTE = "lte"
    IN = "in"
    NOT_IN = "not_in"
    LIKE = "like"
    ILIKE = "ilike"
    IS_NULL = "is_null"
    IS_NOT_NULL = "is_not_null"


@dataclass(frozen=True)
class FilterCondition:
    """A single filter condition.

    Attributes
    ----------
    field
        Field name to filter on.
    operator
        Filter operator.
    value
        Value to compare against.

    Examples
    --------
    >>> condition = FilterCondition("age", FilterOperator.GTE, 18)
    >>> condition.field
    'age'
    """

    field: str
    operator: FilterOperator
    value: object

    def matches(self, obj: Mapping[str, Any]) -> bool:
        """Check if an object matches this condition.

        Parameters
        ----------
        obj
            Object to check.

        Returns
        -------
        bool
            True if the object matches.
        """
        field_value = obj.get(self.field)
        return _evaluate_condition(self.operator, field_value, self.value)


def _evaluate_condition(
    operator: FilterOperator, field_value: object, compare_value: object
) -> bool:
    """Evaluate a filter condition.

    Parameters
    ----------
    operator
        The filter operator.
    field_value
        Value from the object.
    compare_value
        Value to compare against.

    Returns
    -------
    bool
        True if condition matches.
    """
    evaluators = {
        FilterOperator.EQ: lambda f, v: f == v,
        FilterOperator.NE: lambda f, v: f != v,
        FilterOperator.GT: lambda f, v: f is not None and f > v,
        FilterOperator.GTE: lambda f, v: f is not None and f >= v,
        FilterOperator.LT: lambda f, v: f is not None and f < v,
        FilterOperator.LTE: lambda f, v: f is not None and f <= v,
        FilterOperator.IN: lambda f, v: f in v if v else False,
        FilterOperator.NOT_IN: lambda f, v: f not in v if v else True,
        FilterOperator.IS_NULL: lambda f, _: f is None,
        FilterOperator.IS_NOT_NULL: lambda f, _: f is not None,
        FilterOperator.LIKE: lambda f, v: _match_like(f, v, case_sensitive=True),
        FilterOperator.ILIKE: lambda f, v: _match_like(f, v, case_sensitive=False),
    }
    evaluator = evaluators.get(operator)
    if evaluator is None:
        return False
    return evaluator(field_value, compare_value)


def _match_like(field_value: object, pattern: object, *, case_sensitive: bool) -> bool:
    """Match a LIKE pattern.

    Parameters
    ----------
    field_value
        Value to match.
    pattern
        LIKE pattern with % and _ wildcards.
    case_sensitive
        Whether to match case-sensitively.

    Returns
    -------
    bool
        True if pattern matches.
    """
    if field_value is None or not isinstance(pattern, str):
        return False
    regex_pattern = pattern.replace("%", ".*").replace("_", ".")
    flags = 0 if case_sensitive else re.IGNORECASE
    return bool(re.match(regex_pattern, str(field_value), flags))


@dataclass
class FilterBuilder:
    """Builder for composing filter conditions.

    Examples
    --------
    >>> filters = (
    ...     FilterBuilder()
    ...     .eq("status", "active")
    ...     .gte("age", 18)
    ...     .in_("role", ["admin", "user"])
    ...     .build()
    ... )
    """

    _conditions: list[FilterCondition] = field(default_factory=list)

    def eq(self, field: str, value: object) -> Self:
        """Add an equality condition.

        Parameters
        ----------
        field
            Field name.
        value
            Value to compare.

        Returns
        -------
        Self
            Self for chaining.
        """
        self._conditions.append(FilterCondition(field, FilterOperator.EQ, value))
        return self

    def ne(self, field: str, value: object) -> Self:
        """Add a not-equal condition.

        Parameters
        ----------
        field
            Field name.
        value
            Value to compare.

        Returns
        -------
        Self
            Self for chaining.
        """
        self._conditions.append(FilterCondition(field, FilterOperator.NE, value))
        return self

    def gt(self, field: str, value: object) -> Self:
        """Add a greater-than condition.

        Parameters
        ----------
        field
            Field name.
        value
            Value to compare.

        Returns
        -------
        Self
            Self for chaining.
        """
        self._conditions.append(FilterCondition(field, FilterOperator.GT, value))
        return self

    def gte(self, field: str, value: object) -> Self:
        """Add a greater-than-or-equal condition.

        Parameters
        ----------
        field
            Field name.
        value
            Value to compare.

        Returns
        -------
        Self
            Self for chaining.
        """
        self._conditions.append(FilterCondition(field, FilterOperator.GTE, value))
        return self

    def lt(self, field: str, value: object) -> Self:
        """Add a less-than condition.

        Parameters
        ----------
        field
            Field name.
        value
            Value to compare.

        Returns
        -------
        Self
            Self for chaining.
        """
        self._conditions.append(FilterCondition(field, FilterOperator.LT, value))
        return self

    def lte(self, field: str, value: object) -> Self:
        """Add a less-than-or-equal condition.

        Parameters
        ----------
        field
            Field name.
        value
            Value to compare.

        Returns
        -------
        Self
            Self for chaining.
        """
        self._conditions.append(FilterCondition(field, FilterOperator.LTE, value))
        return self

    def in_(self, field: str, values: list[object]) -> Self:
        """Add an in-list condition.

        Parameters
        ----------
        field
            Field name.
        values
            List of values.

        Returns
        -------
        Self
            Self for chaining.
        """
        self._conditions.append(FilterCondition(field, FilterOperator.IN, values))
        return self

    def not_in(self, field: str, values: list[object]) -> Self:
        """Add a not-in-list condition.

        Parameters
        ----------
        field
            Field name.
        values
            List of values.

        Returns
        -------
        Self
            Self for chaining.
        """
        self._conditions.append(FilterCondition(field, FilterOperator.NOT_IN, values))
        return self

    def like(self, field: str, pattern: str) -> Self:
        """Add a LIKE pattern condition.

        Parameters
        ----------
        field
            Field name.
        pattern
            SQL LIKE pattern.

        Returns
        -------
        Self
            Self for chaining.
        """
        self._conditions.append(FilterCondition(field, FilterOperator.LIKE, pattern))
        return self

    def ilike(self, field: str, pattern: str) -> Self:
        """Add a case-insensitive LIKE condition.

        Parameters
        ----------
        field
            Field name.
        pattern
            SQL LIKE pattern.

        Returns
        -------
        Self
            Self for chaining.
        """
        self._conditions.append(FilterCondition(field, FilterOperator.ILIKE, pattern))
        return self

    def is_null(self, field: str) -> Self:
        """Add an is-null condition.

        Parameters
        ----------
        field
            Field name.

        Returns
        -------
        Self
            Self for chaining.
        """
        self._conditions.append(FilterCondition(field, FilterOperator.IS_NULL, None))
        return self

    def is_not_null(self, field: str) -> Self:
        """Add an is-not-null condition.

        Parameters
        ----------
        field
            Field name.

        Returns
        -------
        Self
            Self for chaining.
        """
        self._conditions.append(FilterCondition(field, FilterOperator.IS_NOT_NULL, None))
        return self

    def build(self) -> tuple[FilterCondition, ...]:
        """Build the filter conditions.

        Returns
        -------
        tuple[FilterCondition, ...]
            All filter conditions.
        """
        return tuple(self._conditions)

    def to_dict(self) -> dict[str, object]:
        """Convert to a simple dict format.

        Returns
        -------
        dict[str, object]
            Dictionary of field -> value for EQ conditions.
        """
        result: dict[str, object] = {}
        for condition in self._conditions:
            if condition.operator == FilterOperator.EQ:
                result[condition.field] = condition.value
        return result

    def matches(self, obj: Mapping[str, Any]) -> bool:
        """Check if an object matches all conditions.

        Parameters
        ----------
        obj
            Object to check.

        Returns
        -------
        bool
            True if all conditions match.
        """
        return all(condition.matches(obj) for condition in self._conditions)

    def __len__(self) -> int:
        """Return number of conditions.

        Returns
        -------
        int
            Number of conditions.
        """
        return len(self._conditions)


def parse_filters(filters: Mapping[str, object] | None) -> FilterBuilder:
    """Parse a simple filter dict into a FilterBuilder.

    Parameters
    ----------
    filters
        Simple dict of field -> value for equality conditions.

    Returns
    -------
    FilterBuilder
        Builder with parsed conditions.

    Examples
    --------
    >>> builder = parse_filters({"status": "active", "age": 18})
    >>> len(builder)
    2
    """
    builder = FilterBuilder()
    if filters:
        for field_name, value in filters.items():
            builder.eq(field_name, value)
    return builder


__all__ = [
    "FilterBuilder",
    "FilterCondition",
    "FilterOperator",
    "parse_filters",
]
