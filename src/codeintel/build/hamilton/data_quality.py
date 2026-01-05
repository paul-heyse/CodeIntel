"""Hamilton data quality validators for build outputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import polars as pl
import pyarrow as pa
from hamilton.data_quality.base import DataValidationLevel, DataValidator, ValidationResult
from polars.exceptions import PolarsError

from codeintel.build.schemas import get_schema_provider
from codeintel.build.validation.columnar import (
    ColumnarValidationContext,
    TableValidationError,
    validate_record_batch_reader,
    validate_table,
)
from codeintel.core.schemas.primitives import TableSchema
from codeintel.core.schemas.resolution import resolve_table_schema
from codeintel.core.validation.profiles import ValidationProfile, normalize_validation_profile

if TYPE_CHECKING:
    from codeintel.core.schemas.schema_catalog_models import SchemaObservationRecord


@dataclass(frozen=True, slots=True)
class ColumnarSchemaValidator(DataValidator):
    """Validate table outputs using build columnar validation."""

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
        return "columnar_schema"

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
        return f"Validate columnar contract for {self.table_key}"

    def validate(self, dataset: object) -> ValidationResult:
        """Validate the dataset using storage columnar validation.

        Parameters
        ----------
        dataset
            Dataset to validate.

        Returns
        -------
        ValidationResult
            Validation outcome for the dataset.
        """
        prepared = self._prepare_validation_context()
        if isinstance(prepared, ValidationResult):
            return prepared
        return _validate_columnar_dataset(
            dataset=dataset,
            context=prepared,
            table_key=self.table_key,
        )

    def _prepare_validation_context(self) -> ValidationResult | ColumnarValidationContext:
        table_schema, observation = _resolve_table_schema(self.table_key)
        if table_schema is None:
            return ValidationResult(
                passes=True,
                message=f"No TableSchema available for {self.table_key}",
            )
        return ColumnarValidationContext(
            table_schema=table_schema,
            schema_observation=observation,
            validation_profile=_effective_validation_profile(self.validation_profile),
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
        Build output validators are warn-only to avoid aborting the DAG.

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
    importance = DataValidationLevel.WARN.value
    include_row_count = normalized in {"strict", "data-strict", "lenient", "data-light"}

    validators.append(
        ColumnarSchemaValidator(
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


def _effective_validation_profile(profile: ValidationProfile) -> ValidationProfile:
    if profile == "lenient":
        return "data-light"
    return profile


def _validate_columnar_dataset(
    *,
    dataset: object,
    context: ColumnarValidationContext,
    table_key: str,
) -> ValidationResult:
    try:
        if isinstance(dataset, pa.RecordBatchReader):
            validate_record_batch_reader(table_key, dataset, context=context, mode="strict")
            return ValidationResult(
                passes=True,
                message=f"Columnar validation passed for {table_key}",
            )
        table = _table_from_dataset(dataset)
        if table is None:
            return ValidationResult(
                passes=True,
                message=f"Skipping columnar validation for {type(dataset).__name__}",
            )
        validate_table(table_key, table, context=context, mode="strict")
        return ValidationResult(
            passes=True,
            message=f"Columnar validation passed for {table_key}",
        )
    except TableValidationError as exc:
        return ValidationResult(
            passes=False,
            message=f"Columnar validation failed for {table_key}",
            diagnostics={"errors": list(exc.errors), "table_key": exc.table_key},
        )
    except (PolarsError, pa.ArrowInvalid, TypeError, ValueError) as exc:
        return ValidationResult(
            passes=False,
            message=f"Columnar validation failed for {table_key}: {exc}",
        )


def _table_from_dataset(dataset: object) -> pa.Table | None:
    if isinstance(dataset, pa.Table):
        return dataset
    if isinstance(dataset, pa.RecordBatch):
        return pa.Table.from_batches([dataset])
    if isinstance(dataset, pl.DataFrame):
        return dataset.to_arrow()
    if isinstance(dataset, pl.LazyFrame):
        frame = dataset.collect()
        return frame.to_arrow()
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


__all__ = [
    "ColumnarSchemaValidator",
    "TableRowCountValidator",
    "build_table_schema_validators",
]
