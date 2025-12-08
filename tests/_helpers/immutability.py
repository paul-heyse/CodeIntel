"""Immutability testing utilities for frozen dataclasses.

This module provides helpers for testing that frozen dataclass fields
cannot be modified, without resorting to type suppressions.

The utilities use `setattr` to test runtime immutability of frozen dataclasses.
"""

from __future__ import annotations

import pytest


def assert_frozen(obj: object, attr: str, new_value: object) -> None:
    """Assert that a frozen dataclass field cannot be modified.

    Use this helper to test frozen dataclass immutability without type suppressions.

    Parameters
    ----------
    obj
        Frozen dataclass instance to test.
    attr
        Attribute name to attempt to modify.
    new_value
        Value to attempt to assign.

    Examples
    --------
    >>> from dataclasses import dataclass
    >>> @dataclass(frozen=True)
    ... class Point:
    ...     x: int
    ...     y: int
    >>> p = Point(1, 2)
    >>> assert_frozen(p, "x", 10)  # Passes - field is frozen
    """
    with pytest.raises(AttributeError):
        setattr(obj, attr, new_value)


def assert_all_frozen(obj: object, **attrs: object) -> None:
    """Assert that multiple frozen dataclass fields cannot be modified.

    Convenience wrapper for testing multiple fields at once.

    Parameters
    ----------
    obj
        Frozen dataclass instance to test.
    **attrs
        Mapping of attribute names to values to attempt to assign.

    Examples
    --------
    >>> from dataclasses import dataclass
    >>> @dataclass(frozen=True)
    ... class Point:
    ...     x: int
    ...     y: int
    >>> p = Point(1, 2)
    >>> assert_all_frozen(p, x=10, y=20)  # Passes - all fields frozen
    """
    for attr, value in attrs.items():
        assert_frozen(obj, attr, value)


__all__ = ["assert_all_frozen", "assert_frozen"]
