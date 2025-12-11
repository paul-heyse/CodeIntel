"""Schema-focused assertion helpers for mapping payloads."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from tests._helpers.assertions.expectation_assertions import expect_true


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


def assert_metric_series(
    result: Sequence[Mapping[str, object]] | Mapping[str, object] | None,
    expect_keys: set[str],
) -> list[Mapping[str, object]]:
    """Validate a metric series payload contains expected keys.

    Parameters
    ----------
    result
        Sequence of metric points or single mapping.
    expect_keys
        Required keys present in every series element.

    Returns
    -------
    list[Mapping[str, object]]
        Normalized list of metric point mappings.

    Raises
    ------
    TypeError
        If result is not a mapping or sequence of mappings.
    AssertionError
        If result is missing, wrong type, or missing required keys.
    """
    if result is None:
        message = "Expected metric series payload, got None"
        raise AssertionError(message)
    series: Sequence[Mapping[str, object]]
    if isinstance(result, Mapping):
        series = [result]
    elif isinstance(result, Sequence):
        series = result
    else:
        message = f"Expected mapping or sequence for metric series, got {type(result)}"
        raise TypeError(message)

    normalized: list[Mapping[str, object]] = []
    expect_true(bool(expect_keys), message="expect_keys must not be empty")
    for index, point in enumerate(series):
        if not isinstance(point, Mapping):
            message = f"Expected mapping at series[{index}], got {type(point)}"
            raise TypeError(message)
        missing = expect_keys - set(point.keys())
        if missing:
            message = f"Missing expected keys {sorted(missing)} at series[{index}] in {point}"
            raise AssertionError(message)
        normalized.append(point)
    return normalized


def assert_profile_payload(
    result: Mapping[str, object] | None,
    expect_fields: set[str],
) -> Mapping[str, object]:
    """Validate a profile payload contains required fields.

    Parameters
    ----------
    result
        Profile payload mapping.
    expect_fields
        Required fields expected in the payload.

    Returns
    -------
    Mapping[str, object]
        Validated payload mapping.

    Raises
    ------
    TypeError
        If payload is not a mapping.
    AssertionError
        If payload is missing or lacks expected fields.
    """
    if result is None:
        message = "Expected profile payload, got None"
        raise AssertionError(message)
    if not isinstance(result, Mapping):
        message = f"Expected mapping for profile payload, got {type(result)}"
        raise TypeError(message)
    expect_true(bool(expect_fields), message="expect_fields must not be empty")
    missing = expect_fields - set(result.keys())
    if missing:
        message = f"Missing expected profile fields: {sorted(missing)}"
        raise AssertionError(message)
    return result


__all__ = [
    "assert_mapping_list",
    "assert_mapping_value",
    "assert_metric_series",
    "assert_profile_payload",
]
