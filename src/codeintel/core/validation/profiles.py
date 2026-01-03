"""Validation profile definitions and normalization helpers."""

from __future__ import annotations

from typing import Literal, cast

ValidationProfile = Literal[
    "strict",
    "lenient",
    "schema-only",
    "data-light",
    "data-strict",
]

ValidationDepth = Literal[
    "schema-only",
    "data-light",
    "data-strict",
]

VALIDATION_PROFILES = frozenset(
    {
        "strict",
        "lenient",
        "schema-only",
        "data-light",
        "data-strict",
    }
)

_ALIAS_MAP: dict[str, ValidationProfile] = {
    "schema": "schema-only",
    "schema_only": "schema-only",
    "data_light": "data-light",
    "data_strict": "data-strict",
}


def normalize_validation_profile(
    profile: str | None,
    *,
    default: ValidationProfile = "strict",
) -> ValidationProfile:
    """Normalize a validation profile name.

    Parameters
    ----------
    profile
        Raw profile string.
    default
        Default profile to use when input is None.

    Returns
    -------
    ValidationProfile
        Normalized validation profile.

    Raises
    ------
    ValueError
        If the profile is not a supported value.
    """
    if profile is None:
        return default
    normalized = profile.strip().lower()
    if normalized in _ALIAS_MAP:
        return _ALIAS_MAP[normalized]
    if normalized in VALIDATION_PROFILES:
        return cast("ValidationProfile", normalized)
    msg = f"Invalid validation profile: {profile!r}"
    raise ValueError(msg)


def resolve_validation_depth(profile: ValidationProfile) -> ValidationDepth:
    """Resolve the validation depth for a normalized profile.

    Parameters
    ----------
    profile
        Normalized validation profile.

    Returns
    -------
    ValidationDepth
        Validation depth mapped from the profile.
    """
    if profile == "schema-only":
        return "schema-only"
    if profile in {"data-light", "lenient"}:
        return "data-light"
    return "data-strict"


def is_lenient_profile(profile: ValidationProfile) -> bool:
    """Return True if the profile should treat failures as warnings.

    Returns
    -------
    bool
        True when the profile is lenient.
    """
    return profile == "lenient"


__all__ = [
    "VALIDATION_PROFILES",
    "ValidationDepth",
    "ValidationProfile",
    "is_lenient_profile",
    "normalize_validation_profile",
    "resolve_validation_depth",
]
