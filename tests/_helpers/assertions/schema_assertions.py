"""Schema-focused assertion helpers for mapping payloads."""

from __future__ import annotations

from collections.abc import Mapping


def assert_mapping_value[ValueT](
    mapping: Mapping[str, object] | None,
    key: str,
    expected_type: type[ValueT],
) -> ValueT:
    """Assert a mapping contains a value of the expected type.

    Parameters
    ----------
    mapping
        Mapping to inspect.
    key
        Key whose value should be returned.
    expected_type
        Concrete type expected for the value.

    Returns
    -------
    ValueT
        Value stored at ``key``.

    Raises
    ------
    TypeError
        If the mapping is ``None``, missing the key, or the value has an unexpected type.
    """
    if mapping is None:
        message = f"Expected mapping for key '{key}', got None"
        raise TypeError(message)
    if key not in mapping:
        message = f"Expected key '{key}' in mapping"
        raise TypeError(message)
    value = mapping[key]
    if not isinstance(value, expected_type):
        message = f"Expected key '{key}' to be {expected_type.__name__}, got {type(value).__name__}"
        raise TypeError(message)
    return value


def assert_mapping_list(
    mapping: Mapping[str, object] | None,
    key: str,
) -> list[Mapping[str, object]]:
    """Assert a mapping contains a list of mapping objects.

    Parameters
    ----------
    mapping
        Mapping to inspect.
    key
        Key whose value should be a list of mappings.

    Returns
    -------
    list[Mapping[str, object]]
        List of mapping objects stored at ``key``.

    Raises
    ------
    TypeError
        If the mapping is ``None``, missing the key, or contains non-mapping items.
    """
    raw_list = assert_mapping_value(mapping, key, list)
    typed_items: list[Mapping[str, object]] = []
    for index, item in enumerate(raw_list):
        if not isinstance(item, Mapping):
            message = f"Expected mapping at {key}[{index}], got {type(item).__name__}"
            raise TypeError(message)
        typed_items.append(item)
    return typed_items


__all__ = [
    "assert_mapping_list",
    "assert_mapping_value",
]
