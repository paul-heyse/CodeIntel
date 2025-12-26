"""Attribute taxonomy and cardinality guardrails for observability."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from codeintel.observability.attributes import shape_attributes
from codeintel.observability.policy import (
    DB_ATTRIBUTE_PREFIXES,
    OPERATION_ATTRIBUTE_ALLOWLIST,
    ObservabilityPolicy,
)

SpanAttributeValue = (
    str | bool | int | float | Sequence[str] | Sequence[bool] | Sequence[int] | Sequence[float]
)

CLI_ARG_NAMES_MAX: int = ObservabilityPolicy().cli_arg_names_max


def filter_operation_attributes(
    attributes: Mapping[str, object],
    *,
    policy: ObservabilityPolicy | None = None,
) -> dict[str, SpanAttributeValue]:
    """Filter operation span attributes to the low-cardinality allowlist.

    Returns
    -------
    dict[str, SpanAttributeValue]
        Filtered attributes.
    """
    resolved = policy or ObservabilityPolicy()
    return shape_attributes(
        attributes,
        allowed_keys=resolved.operation_attribute_allowlist,
    )


def filter_db_attributes(
    attributes: Mapping[str, object],
    *,
    policy: ObservabilityPolicy | None = None,
) -> dict[str, SpanAttributeValue]:
    """Filter DB span attributes to allowed prefixes.

    Returns
    -------
    dict[str, SpanAttributeValue]
        Filtered attributes.
    """
    resolved = policy or ObservabilityPolicy()
    return shape_attributes(
        attributes,
        allowed_prefixes=resolved.db_attribute_prefixes,
    )


def limit_cli_arg_names(
    arg_names: Sequence[str],
    *,
    max_len: int | None = None,
) -> tuple[str, ...]:
    """Limit CLI arg names to the configured cardinality budget.

    Returns
    -------
    tuple[str, ...]
        Tuple of bounded CLI argument names.
    """
    limit = CLI_ARG_NAMES_MAX if max_len is None else max_len
    if len(arg_names) <= limit:
        return tuple(arg_names)
    return tuple(arg_names[:limit])


__all__ = [
    "CLI_ARG_NAMES_MAX",
    "DB_ATTRIBUTE_PREFIXES",
    "OPERATION_ATTRIBUTE_ALLOWLIST",
    "filter_db_attributes",
    "filter_operation_attributes",
    "limit_cli_arg_names",
]
