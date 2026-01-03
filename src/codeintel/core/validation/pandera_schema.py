"""Pandera schema helpers for columnar validation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

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


def pandera_error_diagnostics(exc: BaseException, *, table_key: str) -> dict[str, object]:
    """Return structured diagnostics for a Pandera validation error.

    Returns
    -------
    dict[str, object]
        Diagnostics payload for logging or telemetry.
    """
    diagnostics: dict[str, object] = {"table_key": table_key, "error": str(exc)}
    failure_cases = getattr(exc, "failure_cases", None)
    if pl is not None and isinstance(failure_cases, pl.DataFrame):
        diagnostics["failure_cases"] = failure_cases.head(50).to_dicts()
    elif failure_cases is not None:
        diagnostics["failure_cases"] = str(failure_cases)
    return diagnostics


def pandera_schema_for_table(
    table_schema: TableSchema,
    observation: SchemaObservationRecord | None,
    extras_policy: ExtrasPolicy | None = None,
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
    unique = list(table_schema.primary_key) if table_schema.primary_key else None
    columns: dict[str, object] = {}
    stats_by_name = _column_stats(observation)
    for column in table_schema.columns:
        polars_type = polars_type_from_column_type(column.type)
        dtype = _pandera_dtype(polars_type) or pl.Object
        checks = _range_checks(stats_by_name.get(column.name))
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


def _range_checks(stats: Mapping[str, object] | None) -> list[object]:
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


def _strict_mode(extras_policy: ExtrasPolicy | None) -> bool | str:
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
    "pandera_available",
    "pandera_error_diagnostics",
    "pandera_error_types",
    "pandera_schema_for_table",
    "resolve_extras_policy",
]
