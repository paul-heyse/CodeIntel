"""Pandera schema helpers for columnar validation."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import TYPE_CHECKING, Literal, cast

import msgspec

try:
    import pandera.polars as pandera_polars
    from pandera.errors import SchemaError, SchemaErrors
except ImportError:  # pragma: no cover - optional dependency
    pandera_polars = None
    SchemaError = None
    SchemaErrors = None

try:
    import polars as pl
except ImportError:  # pragma: no cover - optional dependency
    pl = None

from codeintel.core.schemas.arrow_gen import EXTRAS_POLICIES, ExtrasPolicy
from codeintel.core.schemas.primitives import TableSchema
from codeintel.core.schemas.schema_catalog_models import SchemaObservationRecord
from codeintel.core.schemas.type_mappings import polars_type_from_column_type
from codeintel.core.serialization.msgspec import to_builtins
from codeintel.core.validation.profiles import (
    ValidationProfile,
    normalize_validation_profile,
    resolve_validation_depth,
)

if TYPE_CHECKING:
    from pandera.polars import Check as PanderaCheck


_PANDERA_MAX_DECIMAL_PRECISION = 28
_DECIMAL_PATTERN = re.compile(r"DECIMAL\\((\\d+)(?:,\\s*(\\d+))?\\)", re.IGNORECASE)


class PanderaDiagnostics(msgspec.Struct, frozen=True):
    """Structured diagnostics for Pandera validation failures."""

    table_key: str
    error: str
    failure_cases: list[dict[str, object]] | str | None = None
    batch_index: int | None = None

    def to_dict(self) -> dict[str, object]:
        """Convert diagnostics to JSON-serializable builtins.

        Returns
        -------
        dict[str, object]
            JSON-ready diagnostics payload.
        """
        return cast("dict[str, object]", to_builtins(self))


def pandera_available() -> bool:
    """Return True when Pandera + Polars are available.

    Returns
    -------
    bool
        True when Pandera + Polars are available.
    """
    return (
        pandera_polars is not None
        and SchemaErrors is not None
        and SchemaError is not None
        and pl is not None
    )


def pandera_error_types() -> tuple[type[BaseException], ...]:
    """Return the Pandera exception types when available.

    Returns
    -------
    tuple[type[BaseException], ...]
        Pandera exception types available in the runtime.
    """
    types: list[type[BaseException]] = []
    if SchemaErrors is not None:
        types.append(SchemaErrors)
    if SchemaError is not None:
        types.append(SchemaError)
    return tuple(types)


def pandera_error_diagnostics(exc: BaseException, *, table_key: str) -> PanderaDiagnostics:
    """Return structured diagnostics for a Pandera validation error.

    Returns
    -------
    PanderaDiagnostics
        Diagnostics payload for logging or telemetry.
    """
    failure_cases: list[dict[str, object]] | str | None = None
    failure_cases = getattr(exc, "failure_cases", None)
    if pl is not None and isinstance(failure_cases, pl.DataFrame):
        failure_cases = failure_cases.head(50).to_dicts()
    elif failure_cases is not None:
        failure_cases = str(failure_cases)
    return PanderaDiagnostics(
        table_key=table_key,
        error=str(exc),
        failure_cases=failure_cases,
    )


def pandera_schema_for_table(
    table_schema: TableSchema,
    observation: SchemaObservationRecord | None,
    extras_policy: ExtrasPolicy | None = None,
    validation_profile: ValidationProfile | None = None,
) -> object | None:
    """Return a Pandera schema for a TableSchema.

    Returns
    -------
    object | None
        Pandera schema object for the table or None if Pandera is unavailable.
    """
    if pandera_polars is None or pl is None:
        return None
    strict = _strict_mode(extras_policy)
    normalized: ValidationProfile = (
        normalize_validation_profile(validation_profile, default="strict")
        if validation_profile is not None
        else "strict"
    )
    depth = resolve_validation_depth(normalized)
    if depth == "schema-only":
        return None
    include_checks = depth != "schema-only"
    include_unique = depth == "data-strict"
    unique = list(table_schema.primary_key) if table_schema.primary_key and include_unique else None
    columns: dict[str, object] = {}
    stats_by_name = _column_stats(observation)
    for column in table_schema.columns:
        polars_type = polars_type_from_column_type(column.type)
        dtype = _pandera_dtype(polars_type) or pl.Object
        precision = _decimal_precision(column.type)
        if precision is not None and precision > _PANDERA_MAX_DECIMAL_PRECISION:
            dtype = pl.Object
        checks = _range_checks(stats_by_name.get(column.name)) if include_checks else []
        columns[column.name] = pandera_polars.Column(
            dtype=dtype,
            nullable=column.nullable,
            required=True,
            checks=checks if checks else None,
            coerce=False,
        )
    return pandera_polars.DataFrameSchema(
        columns,
        strict=strict,
        coerce=False,
        unique=unique,
    )


def _pandera_dtype(polars_type: object | None) -> type | str | None:
    if isinstance(polars_type, str):
        return polars_type
    if isinstance(polars_type, type):
        return polars_type
    dtype_type = getattr(polars_type, "__class__", None)
    if isinstance(dtype_type, type):
        return dtype_type
    return None


def _decimal_precision(column_type: str) -> int | None:
    match = _DECIMAL_PATTERN.search(str(column_type))
    if match is None:
        return None
    try:
        return int(match.group(1))
    except (TypeError, ValueError):
        return None


def _range_checks(stats: Mapping[str, object] | None) -> list[PanderaCheck]:
    if pandera_polars is None or stats is None:
        return []
    min_value = stats.get("min")
    max_value = stats.get("max")
    if min_value is not None and max_value is not None:
        return [
            pandera_polars.Check.in_range(
                min_value,
                max_value,
                include_min=True,
                include_max=True,
            )
        ]
    if min_value is not None:
        return [pandera_polars.Check.ge(min_value)]
    if max_value is not None:
        return [pandera_polars.Check.le(max_value)]
    return []


def _strict_mode(extras_policy: ExtrasPolicy | None) -> Literal["filter"] | bool:
    if extras_policy is None:
        return False
    if extras_policy == "reject":
        return True
    if extras_policy == "drop":
        return "filter"
    return False


def _column_stats(
    observation: SchemaObservationRecord | None,
) -> dict[str, Mapping[str, object]]:
    if observation is None:
        return {}
    raw = observation.column_stats
    if not isinstance(raw, Mapping):
        return {}
    stats: dict[str, Mapping[str, object]] = {}
    for key, value in raw.items():
        if not isinstance(value, Mapping):
            continue
        stats[str(key)] = cast("Mapping[str, object]", value)
    return stats


def resolve_extras_policy(
    observation: SchemaObservationRecord | None,
    *,
    fallback: ExtrasPolicy | None = None,
) -> ExtrasPolicy | None:
    """Return the extras policy derived from observations, if present.

    Returns
    -------
    ExtrasPolicy | None
        Derived extras policy when available, otherwise the fallback.
    """
    if observation is None:
        return fallback
    derived = observation.derived_settings
    if not isinstance(derived, Mapping):
        return fallback
    raw = derived.get("extras_policy")
    if isinstance(raw, str) and raw in EXTRAS_POLICIES:
        return cast("ExtrasPolicy", raw)
    return fallback


__all__ = [
    "PanderaDiagnostics",
    "pandera_available",
    "pandera_error_diagnostics",
    "pandera_error_types",
    "pandera_schema_for_table",
    "resolve_extras_policy",
]
