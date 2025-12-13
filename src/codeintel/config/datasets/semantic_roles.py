"""Semantic role row types and normalization utilities.

This module provides data types for semantic role classification tables:
- FunctionSemanticRoleRow for analytics.semantic_roles_functions
- ModuleSemanticRoleRow for analytics.semantic_roles_modules

These types were originally in analytics.adapters.semantic_roles and were
extracted to support direct usage without the deprecated adapter layer.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence


FUNCTION_COLUMNS: tuple[str, ...] = (
    "repo",
    "commit",
    "function_goid_h128",
    "role",
    "framework",
    "role_confidence",
    "role_sources_json",
    "created_at",
)
MODULE_COLUMNS: tuple[str, ...] = (
    "repo",
    "commit",
    "module",
    "role",
    "role_confidence",
    "role_sources_json",
    "created_at",
)
LEGACY_FUNCTION_TUPLE_LEN = 5
LEGACY_MODULE_TUPLE_LEN = 5


@dataclass(frozen=True)
class FunctionSemanticRoleRow:
    """Normalized row for analytics.semantic_roles_functions.

    Attributes
    ----------
    repo
        Repository identifier.
    commit
        Commit hash.
    function_goid_h128
        Function global ID.
    role
        Semantic role classification.
    framework
        Associated framework (if any).
    role_confidence
        Confidence score for the classification.
    role_sources_json
        JSON-encoded list of evidence sources.
    created_at
        ISO timestamp string.
    """

    repo: str
    commit: str
    function_goid_h128: int
    role: str | None
    framework: str | None
    role_confidence: float | None
    role_sources_json: str
    created_at: str

    def to_tuple(self) -> tuple[object, ...]:
        """Return row values in storage column order.

        Returns
        -------
        tuple[object, ...]
            Row values matching FUNCTION_COLUMNS.
        """
        return (
            self.repo,
            self.commit,
            float(self.function_goid_h128),
            self.role,
            self.framework,
            self.role_confidence,
            self.role_sources_json,
            self.created_at,
        )


@dataclass(frozen=True)
class ModuleSemanticRoleRow:
    """Normalized row for analytics.semantic_roles_modules.

    Attributes
    ----------
    repo
        Repository identifier.
    commit
        Commit hash.
    module
        Module path.
    role
        Semantic role classification.
    role_confidence
        Confidence score for the classification.
    role_sources_json
        JSON-encoded list of evidence sources.
    created_at
        ISO timestamp string.
    """

    repo: str
    commit: str
    module: str
    role: str | None
    role_confidence: float | None
    role_sources_json: str
    created_at: str

    def to_tuple(self) -> tuple[object, ...]:
        """Return row values in storage column order.

        Returns
        -------
        tuple[object, ...]
            Row values matching MODULE_COLUMNS.
        """
        return (
            self.repo,
            self.commit,
            self.module,
            self.role,
            self.role_confidence,
            self.role_sources_json,
            self.created_at,
        )


def timestamp_str(timestamp: datetime | None = None) -> str:
    """Convert timestamp to ISO format string.

    Parameters
    ----------
    timestamp
        Timestamp to convert. If None, uses current UTC time.

    Returns
    -------
    str
        ISO formatted timestamp string.
    """
    return (timestamp or datetime.now(tz=UTC)).isoformat()


def _coerce_int(value: object, field: str) -> int:
    """Coerce value to integer with error handling.

    Parameters
    ----------
    value
        Value to convert.
    field
        Field name for error messages.

    Returns
    -------
    int
        Integer value.

    Raises
    ------
    TypeError
        If value is not convertible to int.
    ValueError
        If conversion fails.
    """
    if not isinstance(value, (int, float, str)):
        message = f"{field} must be int-convertible"
        raise TypeError(message)
    try:
        return int(value)
    except Exception as exc:
        message = f"{field} must be int-convertible"
        raise ValueError(message) from exc


def _coerce_optional_float(value: object | None, field: str) -> float | None:
    """Coerce value to optional float with error handling.

    Parameters
    ----------
    value
        Value to convert.
    field
        Field name for error messages.

    Returns
    -------
    float | None
        Float value or None.

    Raises
    ------
    TypeError
        If value is not convertible to float.
    ValueError
        If conversion fails.
    """
    if value is None:
        return None
    if not isinstance(value, (int, float, str)):
        message = f"{field} must be float-convertible"
        raise TypeError(message)
    try:
        return float(value)
    except Exception as exc:
        message = f"{field} must be float-convertible"
        raise ValueError(message) from exc


def normalize_function_row(
    raw: FunctionSemanticRoleRow | Mapping[str, Any] | Sequence[object],
    repo: str,
    commit: str,
    created_at: str,
) -> FunctionSemanticRoleRow:
    """Normalize a function semantic role row from various input formats.

    Parameters
    ----------
    raw
        Input row as dataclass, mapping, or sequence.
    repo
        Repository identifier.
    commit
        Commit hash.
    created_at
        Default timestamp if not in input.

    Returns
    -------
    FunctionSemanticRoleRow
        Normalized row.

    Raises
    ------
    ValueError
        If required fields are missing or sequence has wrong length.
    """
    if isinstance(raw, FunctionSemanticRoleRow):
        return FunctionSemanticRoleRow(
            repo=repo,
            commit=commit,
            function_goid_h128=raw.function_goid_h128,
            role=raw.role,
            framework=raw.framework,
            role_confidence=raw.role_confidence,
            role_sources_json=raw.role_sources_json,
            created_at=raw.created_at,
        )

    if isinstance(raw, Mapping):
        function_goid_h128 = raw.get("function_goid_h128")
        if function_goid_h128 is None:
            message = "function_goid_h128 is required for semantic role rows"
            raise ValueError(message)
        return FunctionSemanticRoleRow(
            repo=repo,
            commit=commit,
            function_goid_h128=_coerce_int(function_goid_h128, "function_goid_h128"),
            role=raw.get("role"),
            framework=raw.get("framework"),
            role_confidence=_coerce_optional_float(
                raw.get("role_confidence"),
                "role_confidence",
            ),
            role_sources_json=str(raw.get("role_sources_json", "[]")),
            created_at=str(raw.get("created_at", created_at)),
        )

    values = tuple(raw)
    if len(values) == LEGACY_FUNCTION_TUPLE_LEN:
        _, _, function_goid_h128, role, role_confidence = values
        framework = None
        role_sources_json = "[]"
        created = created_at
    elif len(values) == len(FUNCTION_COLUMNS):
        (
            _,
            _,
            function_goid_h128,
            role,
            framework,
            role_confidence,
            role_sources_json,
            created,
        ) = values
    else:
        message = (
            f"Expected {len(FUNCTION_COLUMNS)} values for function role row "
            f"or legacy {LEGACY_FUNCTION_TUPLE_LEN}-tuple, got {len(values)}"
        )
        raise ValueError(message)

    return FunctionSemanticRoleRow(
        repo=repo,
        commit=commit,
        function_goid_h128=_coerce_int(function_goid_h128, "function_goid_h128"),
        role=str(role) if role is not None else None,
        framework=str(framework) if framework is not None else None,
        role_confidence=_coerce_optional_float(role_confidence, "role_confidence"),
        role_sources_json=str(role_sources_json) if role_sources_json is not None else "[]",
        created_at=str(created) if created is not None else created_at,
    )


def normalize_module_row(
    raw: ModuleSemanticRoleRow | Mapping[str, Any] | Sequence[object],
    repo: str,
    commit: str,
    created_at: str,
) -> ModuleSemanticRoleRow:
    """Normalize a module semantic role row from various input formats.

    Parameters
    ----------
    raw
        Input row as dataclass, mapping, or sequence.
    repo
        Repository identifier.
    commit
        Commit hash.
    created_at
        Default timestamp if not in input.

    Returns
    -------
    ModuleSemanticRoleRow
        Normalized row.

    Raises
    ------
    ValueError
        If required fields are missing or sequence has wrong length.
    """
    if isinstance(raw, ModuleSemanticRoleRow):
        return ModuleSemanticRoleRow(
            repo=repo,
            commit=commit,
            module=raw.module,
            role=raw.role,
            role_confidence=raw.role_confidence,
            role_sources_json=raw.role_sources_json,
            created_at=raw.created_at,
        )

    if isinstance(raw, Mapping):
        module = raw.get("module")
        if module is None:
            message = "module is required for semantic role rows"
            raise ValueError(message)
        return ModuleSemanticRoleRow(
            repo=repo,
            commit=commit,
            module=str(module),
            role=raw.get("role"),
            role_confidence=_coerce_optional_float(
                raw.get("role_confidence"),
                "role_confidence",
            ),
            role_sources_json=str(raw.get("role_sources_json", "[]")),
            created_at=str(raw.get("created_at", created_at)),
        )

    values = tuple(raw)
    if len(values) == LEGACY_MODULE_TUPLE_LEN:
        _, _, module, role, role_confidence = values
        role_sources_json = "[]"
        created = created_at
    elif len(values) == len(MODULE_COLUMNS):
        (
            _,
            _,
            module,
            role,
            role_confidence,
            role_sources_json,
            created,
        ) = values
    else:
        message = (
            f"Expected {len(MODULE_COLUMNS)} values for module role row "
            f"or legacy {LEGACY_MODULE_TUPLE_LEN}-tuple, got {len(values)}"
        )
        raise ValueError(message)

    return ModuleSemanticRoleRow(
        repo=repo,
        commit=commit,
        module=str(module),
        role=str(role) if role is not None else None,
        role_confidence=_coerce_optional_float(role_confidence, "role_confidence"),
        role_sources_json=str(role_sources_json) if role_sources_json is not None else "[]",
        created_at=str(created) if created is not None else created_at,
    )


__all__ = [
    "FUNCTION_COLUMNS",
    "LEGACY_FUNCTION_TUPLE_LEN",
    "LEGACY_MODULE_TUPLE_LEN",
    "MODULE_COLUMNS",
    "FunctionSemanticRoleRow",
    "ModuleSemanticRoleRow",
    "normalize_function_row",
    "normalize_module_row",
    "timestamp_str",
]
