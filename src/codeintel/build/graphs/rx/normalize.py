"""Normalization utilities for rustworkx graph outputs."""

from __future__ import annotations

import math
from collections.abc import Hashable, Mapping
from typing import Literal

NanPolicy = Literal["keep", "zero", "raise"]


def stable_key(value: object) -> tuple[str, str]:
    """Return a stable sort key for domain IDs across types.

    Returns
    -------
    tuple[str, str]
        Stable key composed of a type label and normalized value string.
    """
    key_type = type(value).__name__
    key_value = repr(value)
    if value is None:
        key_type = "none"
        key_value = ""
    elif isinstance(value, str):
        key_type = "str"
        key_value = value
    elif isinstance(value, bool):
        key_type = "bool"
        key_value = "1" if value else "0"
    elif isinstance(value, int):
        key_type = "int"
        key_value = str(value)
    elif isinstance(value, float):
        key_type = "float"
        if math.isnan(value):
            key_value = "nan"
        elif math.isinf(value):
            key_value = "inf" if value > 0 else "-inf"
        else:
            key_value = repr(value)
    elif isinstance(value, tuple):
        parts = [stable_key(item) for item in value]
        key_type = "tuple"
        key_value = "|".join(f"{key}:{val}" for key, val in parts)
    elif isinstance(value, frozenset):
        parts = sorted(
            (stable_key(item) for item in value),
            key=lambda item: (item[0], item[1]),
        )
        key_type = "frozenset"
        key_value = "|".join(f"{key}:{val}" for key, val in parts)
    return (key_type, key_value)


def sorted_keys[K: Hashable](mapping: Mapping[K, object]) -> list[K]:
    """Return mapping keys sorted with a stable cross-type ordering.

    Returns
    -------
    list[K]
        Keys ordered by the stable key function.
    """
    return sorted(mapping.keys(), key=stable_key)


def sorted_mapping[K: Hashable, T](mapping: Mapping[K, T]) -> dict[K, T]:
    """Return a new dict sorted by stable key ordering.

    Returns
    -------
    dict[K, T]
        Mapping sorted by the stable key function.
    """
    return {key: mapping[key] for key in sorted_keys(mapping)}


def sorted_nested_mapping[K: Hashable, V: Hashable, T](
    mapping: Mapping[K, Mapping[V, T]],
) -> dict[K, dict[V, T]]:
    """Return nested mappings sorted by stable key ordering.

    Returns
    -------
    dict[K, dict[V, T]]
        Nested mapping sorted by the stable key function at each level.
    """
    return {
        key: sorted_mapping(value) for key, value in sorted_mapping(mapping).items()
    }


def normalize_float(value: float, *, nan_policy: NanPolicy = "keep") -> float:
    """Normalize floats while enforcing a NaN handling policy.

    Returns
    -------
    float
        Normalized float value honoring the NaN policy.

    Raises
    ------
    ValueError
        If nan_policy is "raise" and the value is NaN.
    """
    normalized = float(value)
    if math.isnan(normalized):
        if nan_policy == "zero":
            return 0.0
        if nan_policy == "raise":
            message = "NaN encountered while normalizing numeric output"
            raise ValueError(message)
    return normalized


def normalize_mapping[K: Hashable](
    mapping: Mapping[K, float],
    *,
    nan_policy: NanPolicy = "keep",
) -> dict[K, float]:
    """Normalize float mappings with deterministic ordering and NaN handling.

    Returns
    -------
    dict[K, float]
        Mapping with normalized float values and stable key ordering.
    """
    return {
        key: normalize_float(mapping[key], nan_policy=nan_policy)
        for key in sorted_keys(mapping)
    }


def edge_weight_from_payload(
    payload: object | None,
    *,
    nan_policy: NanPolicy = "keep",
) -> float:
    """Coerce edge payloads into numeric weights with a NaN policy.

    Returns
    -------
    float
        Normalized numeric edge weight.
    """
    if payload is None:
        return 1.0
    if isinstance(payload, bool):
        return float(int(payload))
    if isinstance(payload, (int, float)):
        return normalize_float(float(payload), nan_policy=nan_policy)
    if isinstance(payload, str):
        try:
            return normalize_float(float(payload), nan_policy=nan_policy)
        except ValueError:
            return 1.0
    return 1.0


__all__ = [
    "NanPolicy",
    "edge_weight_from_payload",
    "normalize_float",
    "normalize_mapping",
    "sorted_keys",
    "sorted_mapping",
    "sorted_nested_mapping",
    "stable_key",
]
