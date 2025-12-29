"""Helpers for parsing and applying Hamilton tag filters in the CLI."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

from codeintel.cli.errors import ValidationError
from codeintel.core.hamilton.tag_query import TagQuery
from codeintel.runtime.runtime_bundle import RuntimeBundle


def parse_tag_filters(values: Sequence[str] | None) -> dict[str, object] | None:
    """Parse CLI tag filters into a Hamilton tag filter mapping.

    Parameters
    ----------
    values
        Raw tag filter values (repeatable). Supports key=value or key.

    Returns
    -------
    dict[str, object] | None
        Parsed tag filter mapping, or None when no filters are supplied.

    Raises
    ------
    ValidationError
        If any tag filter is empty or malformed.
    """
    if not values:
        return None
    parsed: dict[str, object] = {}
    for raw in values:
        entry = raw.strip()
        if not entry:
            msg = "Tag filters must be non-empty."
            raise ValidationError(msg)
        key, sep, value = entry.partition("=")
        key = key.strip()
        invalid_msg = f"Invalid tag filter: {raw}"
        if not key:
            raise ValidationError(invalid_msg)
        if not sep:
            _merge_tag_value(parsed, key, None)
            continue
        value = value.strip()
        if not value:
            raise ValidationError(invalid_msg)
        values_list = [item.strip() for item in value.split(",") if item.strip()]
        if not values_list:
            raise ValidationError(invalid_msg)
        merged: object = values_list if len(values_list) > 1 else values_list[0]
        _merge_tag_value(parsed, key, merged)
    return parsed


def filter_targets_by_tags(
    runtime_bundle: RuntimeBundle,
    *,
    targets: Iterable[str],
    tag_filter: Mapping[str, object] | None,
) -> list[str]:
    """Filter targets by a Hamilton tag filter mapping.

    Parameters
    ----------
    runtime_bundle
        Runtime bundle containing the Hamilton driver and catalog.
    targets
        Candidate target names to filter.
    tag_filter
        Tag filter mapping or None to disable filtering.

    Returns
    -------
    list[str]
        Target names matching the provided tag filters.
    """
    target_list = list(targets)
    if not tag_filter:
        return target_list
    tag_query = TagQuery(runtime_bundle.driver)
    matching_nodes = set(tag_query.names(tag_filter))
    filtered: list[str] = []
    for target in target_list:
        descriptor = runtime_bundle.catalog.targets.get(target)
        if descriptor is None:
            continue
        if descriptor.anchor_node in matching_nodes:
            filtered.append(target)
    return filtered


def _merge_tag_value(parsed: dict[str, object], key: str, value: object) -> None:
    if key not in parsed:
        parsed[key] = value
        return
    existing = parsed[key]
    if existing is None or value is None:
        if existing is None and value is None:
            return
        msg = f"Conflicting tag filters for {key}"
        raise ValidationError(msg)
    existing_values = existing if isinstance(existing, list) else [existing]
    new_values = value if isinstance(value, list) else [value]
    merged = [str(item) for item in [*existing_values, *new_values]]
    parsed[key] = merged


__all__ = [
    "filter_targets_by_tags",
    "parse_tag_filters",
]
