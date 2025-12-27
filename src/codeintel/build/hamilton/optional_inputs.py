"""Registry for optional input table keys per target."""

from __future__ import annotations

from collections.abc import Iterable

_OPTIONAL_INPUTS_BY_TARGET: dict[str, set[str]] = {}


def register_optional_inputs(target: str, table_keys: Iterable[str]) -> None:
    """Register optional input tables for a target."""
    normalized = {str(key) for key in table_keys if key}
    if not normalized:
        return
    existing = _OPTIONAL_INPUTS_BY_TARGET.setdefault(str(target), set())
    existing.update(normalized)


def optional_inputs_for_target(target: str) -> frozenset[str]:
    """Return optional input table keys for a target.

    Returns
    -------
    frozenset[str]
        Optional input table keys registered for the target.
    """
    return frozenset(_OPTIONAL_INPUTS_BY_TARGET.get(str(target), set()))


__all__ = [
    "optional_inputs_for_target",
    "register_optional_inputs",
]
