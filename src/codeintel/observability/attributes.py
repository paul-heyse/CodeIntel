"""Shared attribute shaping utilities for observability."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

SpanAttributeValue = (
    str | bool | int | float | Sequence[str] | Sequence[bool] | Sequence[int] | Sequence[float]
)


def shape_attributes(
    attributes: Mapping[str, object],
    *,
    allowed_keys: frozenset[str] | None = None,
    allowed_prefixes: Sequence[str] | None = None,
    max_list_len: int | None = None,
    max_str_len: int | None = None,
) -> dict[str, SpanAttributeValue]:
    """Filter and coerce attributes with basic cardinality controls.

    Returns
    -------
    dict[str, SpanAttributeValue]
        Filtered and coerced attributes.
    """
    if not attributes:
        return {}
    allowed = allowed_keys or frozenset()
    prefixes = tuple(allowed_prefixes or ())
    shaped: dict[str, SpanAttributeValue] = {}
    for key, value in attributes.items():
        if allowed or prefixes:
            allowlist_match = bool(allowed) and key in allowed
            prefix_match = bool(prefixes) and any(key.startswith(prefix) for prefix in prefixes)
            if not (allowlist_match or prefix_match):
                continue
        attr_value = coerce_attribute_value(
            value,
            max_list_len=max_list_len,
            max_str_len=max_str_len,
        )
        if attr_value is not None:
            shaped[key] = attr_value
    return shaped


def coerce_attribute_value(
    value: object,
    *,
    max_list_len: int | None = None,
    max_str_len: int | None = None,
) -> SpanAttributeValue | None:
    """Coerce a value into an OpenTelemetry-safe attribute type.

    Returns
    -------
    SpanAttributeValue | None
        Coerced attribute value or None when the value is invalid.
    """
    if value is None:
        return None
    if isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return _truncate_str(value, max_str_len)
    if isinstance(value, (list, tuple)):
        return _coerce_sequence(value, max_list_len=max_list_len, max_str_len=max_str_len)
    return _truncate_str(str(value), max_str_len)


def redact_command_value(value: str | None, *, keep_segments: int) -> str | None:
    """Redact a command value by keeping the trailing segments.

    Returns
    -------
    str | None
        Redacted command value or None.
    """
    return _redact_path_like(value, keep_segments=keep_segments)


def redact_path_value(value: str | None, *, keep_segments: int) -> str | None:
    """Redact a path value by keeping the trailing segments.

    Returns
    -------
    str | None
        Redacted path value or None.
    """
    return _redact_path_like(value, keep_segments=keep_segments)


def _coerce_sequence(
    values: Sequence[object],
    *,
    max_list_len: int | None,
    max_str_len: int | None,
) -> Sequence[str] | Sequence[bool] | Sequence[int] | Sequence[float]:
    truncated = _truncate_sequence(values, max_list_len)
    if all(isinstance(item, str) for item in truncated):
        return [_truncate_str(str(item), max_str_len) for item in truncated]
    if all(type(item) is bool for item in truncated):
        return [item for item in truncated if isinstance(item, bool)]
    if all(type(item) is int for item in truncated):
        return [item for item in truncated if isinstance(item, int)]
    if all(type(item) is float for item in truncated):
        return [item for item in truncated if isinstance(item, float)]
    if all(isinstance(item, (str, bool, int, float)) for item in truncated):
        return [_truncate_str(str(item), max_str_len) for item in truncated]
    return [_truncate_str(str(item), max_str_len) for item in truncated]


def _truncate_sequence(values: Sequence[object], max_len: int | None) -> Sequence[object]:
    if max_len is None or max_len < 0:
        return values
    return values[:max_len]


def _truncate_str(value: str, max_len: int | None) -> str:
    if max_len is None or max_len < 0:
        return value
    if len(value) <= max_len:
        return value
    if max_len <= 1:
        return value[:max_len]
    return value[: max_len - 1] + "."


def _redact_path_like(value: str | None, *, keep_segments: int) -> str | None:
    if value is None:
        return None
    if keep_segments <= 0:
        return ""
    path = Path(value)
    parts = path.parts
    if not parts:
        return value
    if len(parts) <= keep_segments:
        return str(path)
    keep = parts[-keep_segments:]
    return str(Path(*keep))


__all__ = [
    "SpanAttributeValue",
    "coerce_attribute_value",
    "redact_command_value",
    "redact_path_value",
    "shape_attributes",
]
