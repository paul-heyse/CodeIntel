"""Hamilton data quality validators for build outputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, cast

import polars as pl
import pyarrow as pa
from hamilton.data_quality.base import DataValidationLevel, DataValidator, ValidationResult
from polars.exceptions import PolarsError

from codeintel.build.schemas import get_schema_provider
from codeintel.core.schemas.arrow_gen import DEFAULT_EXTRAS_POLICY
from codeintel.core.schemas.primitives import TableSchema
from codeintel.core.schemas.resolution import resolve_table_schema
from codeintel.core.validation.pandera_schema import (
    pandera_available,
    pandera_error_diagnostics,
    pandera_error_types,
    pandera_schema_for_table,
    resolve_extras_policy,
)
from codeintel.core.validation.profiles import ValidationProfile, normalize_validation_profile

if TYPE_CHECKING:
    from codeintel.core.schemas.schema_catalog_models import SchemaObservationRecord


class _PanderaSchemaProtocol(Protocol):
    def validate(
        self,
        check_obj: pl.DataFrame | pl.LazyFrame,
        **kwargs: object,
    ) -> object: ...


@dataclass(frozen=True, slots=True)
class _PanderaValidationContext:
    schema_obj: _PanderaSchemaProtocol
    error_types: tuple[type[BaseException], ...]
    primary_keys: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PanderaSchemaValidator(DataValidator):
    """Validate table outputs with a Pandera schema derived from the contract."""

    table_key: str
    validation_profile: ValidationProfile

    def __init__(
        self,
        *,
        table_key: str,
        validation_profile: ValidationProfile,
        importance: str,
    ) -> None:
        object.__setattr__(self, "_importance", DataValidationLevel(importance))
        object.__setattr__(self, "table_key", table_key)
        object.__setattr__(self, "validation_profile", validation_profile)

    @classmethod
    def name(cls) -> str:
        """Return the canonical validator name.

        Returns
        -------
        str
            Validator name identifier.
        """
        return "pandera_schema"

    @staticmethod
    def applies_to(datatype: type[object]) -> bool:
        """Return whether the validator applies to the provided type.

        Parameters
        ----------
        datatype
            Dataset type being validated.

        Returns
        -------
        bool
            True when the validator applies.
        """
        _ = datatype
        return True

    def description(self) -> str:
        """Return a human-readable validator description.

        Returns
        -------
        str
            Validator description for diagnostics.
        """
        return f"Validate Pandera contract for {self.table_key}"

    def validate(self, dataset: object) -> ValidationResult:
        """Validate the dataset using a Pandera schema derived from TableSchema.

        Parameters
        ----------
        dataset
            Dataset to validate.

        Returns
        -------
        ValidationResult
            Validation outcome for the dataset.
        """
        prepared = self._prepare_validation()
        if isinstance(prepared, ValidationResult):
            return prepared
        if isinstance(dataset, pa.RecordBatchReader):
            if prepared.primary_keys:
                return _validate_pandera_reader_full(
                    dataset,
                    context=prepared,
                    table_key=self.table_key,
                )
            return _validate_pandera_reader_stream(
                dataset,
                context=prepared,
                table_key=self.table_key,
            )
        frame = _pandera_frame(dataset)
        if frame is None:
            return ValidationResult(
                passes=True,
                message=f"Skipping Pandera validation for {type(dataset).__name__}",
            )
        return _validate_pandera_frame(
            frame,
            context=prepared,
            table_key=self.table_key,
        )

    def _prepare_validation(self) -> ValidationResult | _PanderaValidationContext:
        if not pandera_available():
            return ValidationResult(
                passes=True,
                message=f"Pandera unavailable for {self.table_key}",
            )
        table_schema, observation = _resolve_table_schema(self.table_key)
        if table_schema is None:
            return ValidationResult(
                passes=True,
                message=f"No Pandera schema for {self.table_key}",
            )
        extras_policy = resolve_extras_policy(observation, fallback=DEFAULT_EXTRAS_POLICY)
        schema = pandera_schema_for_table(
            table_schema=table_schema,
            observation=observation,
            extras_policy=extras_policy,
            validation_profile=self.validation_profile,
        )
        if schema is None:
            return ValidationResult(
                passes=True,
                message=f"Pandera unavailable for {self.table_key}",
            )
        error_types = pandera_error_types()
        if not error_types:
            return ValidationResult(
                passes=True,
                message=f"Pandera error types unavailable for {self.table_key}",
            )
        schema_obj = cast("_PanderaSchemaProtocol", schema)
        primary_keys = tuple(table_schema.primary_key)
        return _PanderaValidationContext(
            schema_obj=schema_obj,
            error_types=error_types,
            primary_keys=primary_keys,
        )


@dataclass(frozen=True, slots=True)
class TableRowCountValidator(DataValidator):
    """Validate minimum row counts for table outputs."""

    table_key: str
    min_rows: int

    def __init__(
        self,
        *,
        table_key: str,
        min_rows: int,
        importance: str,
    ) -> None:
        object.__setattr__(self, "_importance", DataValidationLevel(importance))
        object.__setattr__(self, "table_key", table_key)
        object.__setattr__(self, "min_rows", min_rows)

    @classmethod
    def name(cls) -> str:
        """Return the canonical validator name.

        Returns
        -------
        str
            Validator name identifier.
        """
        return "table_row_count"

    @staticmethod
    def applies_to(datatype: type[object]) -> bool:
        """Return whether the validator applies to the provided type.

        Parameters
        ----------
        datatype
            Dataset type being validated.

        Returns
        -------
        bool
            True when the validator applies.
        """
        _ = datatype
        return True

    def description(self) -> str:
        """Return a human-readable validator description.

        Returns
        -------
        str
            Validator description for diagnostics.
        """
        return f"Validate row count for {self.table_key}"

    def validate(self, dataset: object) -> ValidationResult:
        """Validate that row counts meet the configured minimum.

        Parameters
        ----------
        dataset
            Dataset to validate.

        Returns
        -------
        ValidationResult
            Validation outcome for the dataset.
        """
        if dataset is None:
            return ValidationResult(
                passes=True, message=f"No rows to validate for {self.table_key}"
            )
        if self.min_rows <= 0:
            return ValidationResult(
                passes=True,
                message=f"Row count validation disabled for {self.table_key}",
            )
        count = _row_count(dataset)
        if count is None:
            return ValidationResult(
                passes=True,
                message=f"Skipping row count validation for {type(dataset).__name__}",
            )
        if count < self.min_rows:
            return ValidationResult(
                passes=False,
                message=f"Row count {count} below minimum {self.min_rows} for {self.table_key}",
                diagnostics={
                    "table_key": self.table_key,
                    "row_count": count,
                    "min_rows": self.min_rows,
                },
            )
        return ValidationResult(
            passes=True,
            message=f"Row count validated for {self.table_key}",
        )


def build_table_schema_validators(
    *,
    table_key: str,
    profile: str | None,
    min_rows: int | None = None,
) -> tuple[DataValidator, ...]:
    """Construct table schema validators for a profile.

    Parameters
    ----------
    table_key
        Table key used to resolve schema metadata.
    profile
        Validation profile name ("strict", "lenient", "schema-only", "data-light", "data-strict").
    min_rows
        Optional minimum row count threshold.

    Returns
    -------
    tuple[DataValidator, ...]
        Validators configured for the table.

    Raises
    ------
    ValueError
        If the validation profile is unsupported.
    """
    if profile is None:
        return ()
    try:
        normalized = normalize_validation_profile(profile, default="strict")
    except ValueError as exc:
        msg = f"Unsupported validation profile: {profile}"
        raise ValueError(msg) from exc

    validators: list[DataValidator] = []
    if normalized in {"strict", "data-strict"}:
        importance = DataValidationLevel.FAIL.value
        include_row_count = True
    elif normalized in {"lenient", "data-light"}:
        importance = DataValidationLevel.WARN.value
        include_row_count = True
    else:
        importance = DataValidationLevel.FAIL.value
        include_row_count = False

    validators.append(
        PanderaSchemaValidator(
            table_key=table_key,
            validation_profile=normalized,
            importance=importance,
        )
    )
    if include_row_count and isinstance(min_rows, int) and min_rows > 0:
        validators.append(
            TableRowCountValidator(
                table_key=table_key,
                min_rows=min_rows,
                importance=importance,
            )
        )
    return tuple(validators)


def _resolve_table_schema(
    table_key: str,
) -> tuple[TableSchema | None, SchemaObservationRecord | None]:
    try:
        provider = get_schema_provider()
    except RuntimeError:
        provider = None
    resolution = resolve_table_schema(table_key, schema_provider=provider)
    return resolution.table_schema, resolution.observation


def _pandera_frame(dataset: object) -> pl.LazyFrame | pl.DataFrame | None:
    if isinstance(dataset, (pl.LazyFrame, pl.DataFrame)):
        return dataset
    if isinstance(dataset, pa.Table):
        frame = pl.from_arrow(dataset)
        return frame if isinstance(frame, pl.DataFrame) else None
    if isinstance(dataset, pa.RecordBatch):
        frame = pl.from_arrow(pa.Table.from_batches([dataset]))
        return frame if isinstance(frame, pl.DataFrame) else None
    return None


def _row_count(dataset: object) -> int | None:
    if isinstance(dataset, pl.LazyFrame):
        try:
            frame = dataset.select(pl.len().alias("row_count")).collect()
        except PolarsError:
            return None
        return _frame_scalar(frame)
    if isinstance(dataset, pl.DataFrame):
        return dataset.height
    num_rows = getattr(dataset, "num_rows", None)
    if isinstance(num_rows, int):
        return num_rows
    return None


def _frame_scalar(dataset: pl.DataFrame) -> int | None:
    if dataset.height == 0 or dataset.width == 0:
        return None
    try:
        value = dataset.item()
    except (PolarsError, ValueError):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return None


def _validate_pandera_frame(
    frame: pl.DataFrame | pl.LazyFrame,
    *,
    context: _PanderaValidationContext,
    table_key: str,
) -> ValidationResult:
    try:
        context.schema_obj.validate(frame, lazy=True)
    except context.error_types as exc:
        diagnostics = pandera_error_diagnostics(exc, table_key=table_key)
        return ValidationResult(
            passes=False,
            message=f"Pandera validation failed for {table_key}",
            diagnostics=diagnostics.to_dict(),
        )
    return ValidationResult(
        passes=True,
        message=f"Pandera validation passed for {table_key}",
    )


def _validate_pandera_reader_stream(
    reader: pa.RecordBatchReader,
    *,
    context: _PanderaValidationContext,
    table_key: str,
) -> ValidationResult:
    for batch_index, batch in enumerate(reader):
        frame = _pandera_frame(batch)
        if frame is None:
            return ValidationResult(
                passes=True,
                message=f"Skipping Pandera validation for {type(batch).__name__}",
            )
        try:
            context.schema_obj.validate(frame, lazy=True)
        except context.error_types as exc:
            diagnostics = pandera_error_diagnostics(exc, table_key=table_key)
            diagnostics = diagnostics.__replace__(batch_index=batch_index)
            return ValidationResult(
                passes=False,
                message=f"Pandera validation failed for {table_key}",
                diagnostics=diagnostics.to_dict(),
            )
    return ValidationResult(
        passes=True,
        message=f"Pandera validation passed for {table_key}",
    )


def _validate_pandera_reader_full(
    reader: pa.RecordBatchReader,
    *,
    context: _PanderaValidationContext,
    table_key: str,
) -> ValidationResult:
    try:
        batches = list(reader)
    except (TypeError, ValueError, pa.ArrowInvalid):
        return ValidationResult(
            passes=True,
            message=f"Skipping Pandera validation for {type(reader).__name__}",
        )
    table = pa.Table.from_batches(batches, schema=reader.schema)
    frame = _pandera_frame(table)
    if frame is None:
        return ValidationResult(
            passes=True,
            message=f"Skipping Pandera validation for {type(table).__name__}",
        )
    return _validate_pandera_frame(
        frame,
        context=context,
        table_key=table_key,
    )


__all__ = [
    "PanderaSchemaValidator",
    "TableRowCountValidator",
    "build_table_schema_validators",
]
