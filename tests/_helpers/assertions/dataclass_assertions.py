"""Dataclass-related assertion helpers for test validation."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, is_dataclass

import pytest


def assert_cannot_setattr(instance: object, field_name: str, value: object) -> None:
    """Assert that setting an attribute on a frozen/immutable instance fails.

    This helper keeps immutability assertions type-safe by avoiding
    direct assignments that static analysis treats as errors.

    Parameters
    ----------
    instance
        The object instance to test for immutability.
    field_name
        The name of the attribute to attempt to set.
    value
        The value to attempt to assign.

    Raises
    ------
    AssertionError
        If the target instance is not frozen or attribute setting succeeds.
    """
    expected_errors = (AttributeError, FrozenInstanceError)

    params = getattr(instance, "__dataclass_params__", None)
    if is_dataclass(instance) and params is not None and not params.frozen:
        message = f"{type(instance).__name__} is not frozen; cannot assert immutability."
        raise AssertionError(message)

    with pytest.raises(expected_errors):
        setattr(instance, field_name, value)


__all__ = [
    "assert_cannot_setattr",
]
