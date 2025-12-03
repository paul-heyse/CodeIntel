"""Helpers for testing frozen dataclass immutability.

This module provides utilities for testing that frozen dataclasses
properly raise FrozenInstanceError when mutation is attempted.

The helper function accepts `object` type to avoid pyright's static
analysis of frozen attribute assignment, while still testing runtime behavior.
"""

from __future__ import annotations


def try_setattr(obj: object, attr: str, value: object) -> None:
    """Attempt to set an attribute on an object.

    Use this helper to test frozen dataclass immutability without
    triggering pyright errors. The function signature uses `object`
    types to bypass static analysis while still exercising runtime behavior.
    Exceptions raised by the underlying object propagate to the caller.

    Parameters
    ----------
    obj
        The object to modify (typically a frozen dataclass instance).
    attr
        The attribute name to set.
    value
        The value to assign.

    Examples
    --------
    >>> from dataclasses import dataclass, FrozenInstanceError
    >>> import pytest
    >>>
    >>> @dataclass(frozen=True)
    ... class Point:
    ...     x: int
    ...     y: int
    >>>
    >>> p = Point(1, 2)
    >>> with pytest.raises(FrozenInstanceError):
    ...     try_setattr(p, "x", 99)
    """
    setattr(obj, attr, value)
