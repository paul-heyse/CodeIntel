"""Attribute shaping, truncation, and redaction helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from codeintel.observability.policy import AttributeBudget

SpanAttributeValue = (
    str | bool | int | float | Sequence[str] | Sequence[bool] | Sequence[int] | Sequence[float]
)


def shape_attributes(
    attributes: Mapping[str, object],
    *,
    allowed_keys: frozenset[str] | None = None,
    allowed_prefixes: Sequence[str] | None = None,
    budget: AttributeBudget | None = None,
    max_list_len: int | None = None,
    max_str_len: int | None = None,
) -> dict[str, SpanAttributeValue]:
    """Filter and coerce attributes with allowlists and budgets.

    Returns
    -------
    dict[str, SpanAttributeValue]
        Filtered attributes.
    """
    if not attributes:
        return {}

    allowed = allowed_keys or frozenset()
    prefixes = tuple(allowed_prefixes or ())
    list_limit = _resolve_budget_value(budget, "max_list_len", max_list_len)
    str_limit = _resolve_budget_value(budget, "max_str_len", max_str_len)

    shaped: dict[str, SpanAttributeValue] = {}
    for key, value in attributes.items():
        if allowed or prefixes:
            allowlist_match = bool(allowed) and key in allowed
            prefix_match = bool(prefixes) and any(key.startswith(prefix) for prefix in prefixes)
            if not (allowlist_match or prefix_match):
                continue
        attr_value = coerce_attribute_value(
            value,
            max_list_len=list_limit,
            max_str_len=str_limit,
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
        Coerced attribute value or None when invalid.
    """
    if value is None:
        return None
    if isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return truncate_str(value, max_str_len)
    if isinstance(value, (list, tuple)):
        return _coerce_sequence(value, max_list_len=max_list_len, max_str_len=max_str_len)
    return truncate_str(str(value), max_str_len)


def truncate_str(value: str, max_len: int | None) -> str:
    """Truncate a string to a maximum length when configured.

    Returns
    -------
    str
        Possibly truncated string value.
    """
    if max_len is None or max_len < 0:
        return value
    if len(value) <= max_len:
        return value
    if max_len <= 1:
        return value[:max_len]
    return value[: max_len - 1] + "."


def truncate_sequence(values: Sequence[object], max_len: int | None) -> Sequence[object]:
    """Truncate a sequence to a maximum length when configured.

    Returns
    -------
    Sequence[object]
        Possibly truncated sequence.
    """
    if max_len is None or max_len < 0:
        return values
    return values[:max_len]


def redact_command_value(value: str | None, *, keep_segments: int) -> str | None:
    """Redact a command value by keeping trailing segments.

    Returns
    -------
    str | None
        Redacted command value or None.
    """
    return _redact_path_like(value, keep_segments=keep_segments)


def redact_path_value(value: str | None, *, keep_segments: int) -> str | None:
    """Redact a path value by keeping trailing segments.

    Returns
    -------
    str | None
        Redacted path value or None.
    """
    return _redact_path_like(value, keep_segments=keep_segments)


def prune_none(values: Mapping[str, SpanAttributeValue | None]) -> dict[str, SpanAttributeValue]:
    """Drop None values from a mapping.

    Returns
    -------
    dict[str, SpanAttributeValue]
        Mapping without None values.
    """
    return {key: value for key, value in values.items() if value is not None}


def limit_cli_arg_names(arg_names: Sequence[str], *, max_len: int) -> tuple[str, ...]:
    """Limit CLI arg names to a configured cardinality budget.

    Returns
    -------
    tuple[str, ...]
        Possibly truncated argument name list.
    """
    if len(arg_names) <= max_len:
        return tuple(arg_names)
    return tuple(arg_names[:max_len])


def _coerce_sequence(
    values: Sequence[object],
    *,
    max_list_len: int | None,
    max_str_len: int | None,
) -> Sequence[str] | Sequence[bool] | Sequence[int] | Sequence[float]:
    truncated = truncate_sequence(values, max_list_len)
    if all(isinstance(item, str) for item in truncated):
        return [truncate_str(str(item), max_str_len) for item in truncated]
    if all(type(item) is bool for item in truncated):
        return [item for item in truncated if isinstance(item, bool)]
    if all(type(item) is int for item in truncated):
        return [item for item in truncated if isinstance(item, int)]
    if all(type(item) is float for item in truncated):
        return [item for item in truncated if isinstance(item, float)]
    if all(isinstance(item, (str, bool, int, float)) for item in truncated):
        return [truncate_str(str(item), max_str_len) for item in truncated]
    return [truncate_str(str(item), max_str_len) for item in truncated]


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


def _resolve_budget_value(
    budget: AttributeBudget | None,
    attr_name: str,
    override: int | None,
) -> int | None:
    if override is not None:
        return override
    if budget is None:
        return None
    return getattr(budget, attr_name, None)


__all__ = [
    "SpanAttributeValue",
    "coerce_attribute_value",
    "limit_cli_arg_names",
    "prune_none",
    "redact_command_value",
    "redact_path_value",
    "shape_attributes",
    "truncate_sequence",
    "truncate_str",
]
