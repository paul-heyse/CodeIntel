"""Hamilton data quality validators for build outputs."""

from __future__ import annotations

from collections.abc import Sequence, Sized
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, cast

import polars as pl
import pyarrow as pa
import pyarrow.compute as pc
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

if TYPE_CHECKING:
    from collections.abc import Iterable

    from codeintel.core.schemas.schema_catalog_models import SchemaObservationRecord


class _PanderaSchemaProtocol(Protocol):
    def validate(
        self,
        check_obj: pl.DataFrame | pl.LazyFrame,
        **kwargs: object,
    ) -> object: ...


@dataclass(frozen=True, slots=True)
class TableSchemaColumnsValidator(DataValidator):
    """Validate table outputs against declared schema columns."""

    table_key: str
    enforce_non_nullable: bool = False

    def __init__(
        self,
        *,
        table_key: str,
        importance: str,
        enforce_non_nullable: bool = False,
    ) -> None:
        object.__setattr__(self, "_importance", DataValidationLevel(importance))
        object.__setattr__(self, "table_key", table_key)
        object.__setattr__(self, "enforce_non_nullable", enforce_non_nullable)

    @classmethod
    def name(cls) -> str:
        """Return the canonical validator name.

        Returns
        -------
        str
            Validator name identifier.
        """
        return "table_schema_columns"

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
        return f"Validate columns for {self.table_key}"

    def validate(self, dataset: object) -> ValidationResult:
        """Validate column presence (and optionally nullability).

        Parameters
        ----------
        dataset
            Dataset to validate.

        Returns
        -------
        ValidationResult
            Validation outcome for the dataset.
        """
        schema, _ = _resolve_table_schema(self.table_key)
        if schema is None:
            return ValidationResult(
                passes=True,
                message=f"No declared schema for {self.table_key}",
            )
        column_names = _column_names(dataset)
        if column_names is None:
            return ValidationResult(
                passes=True,
                message=f"Skipping column validation for {type(dataset).__name__}",
            )

        missing = _missing_columns(column_names, schema)
        if missing:
            return ValidationResult(
                passes=False,
                message=f"Missing required columns: {', '.join(missing)}",
                diagnostics={"table_key": self.table_key, "missing_columns": missing},
            )

        if self.enforce_non_nullable:
            non_nullable = _non_nullable_columns(schema)
            nullable_issues = _non_nullable_violations(dataset, non_nullable)
            if nullable_issues:
                return ValidationResult(
                    passes=False,
                    message=f"Non-nullable columns contain nulls: {', '.join(nullable_issues)}",
                    diagnostics={
                        "table_key": self.table_key,
                        "nullable_violations": nullable_issues,
                    },
                )

        return ValidationResult(
            passes=True,
            message=f"Schema columns validated for {self.table_key}",
        )


@dataclass(frozen=True, slots=True)
class PanderaSchemaValidator(DataValidator):
    """Validate table outputs with a Pandera schema derived from the contract."""

    table_key: str

    def __init__(self, *, table_key: str, importance: str) -> None:
        object.__setattr__(self, "_importance", DataValidationLevel(importance))
        object.__setattr__(self, "table_key", table_key)

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
        result: ValidationResult
        if not pandera_available():
            result = ValidationResult(
                passes=True,
                message=f"Pandera unavailable for {self.table_key}",
            )
        else:
            table_schema, observation = _resolve_table_schema(self.table_key)
            if table_schema is None:
                result = ValidationResult(
                    passes=True,
                    message=f"No Pandera schema for {self.table_key}",
                )
            else:
                extras_policy = resolve_extras_policy(observation, fallback=DEFAULT_EXTRAS_POLICY)
                schema = pandera_schema_for_table(
                    table_schema=table_schema,
                    observation=observation,
                    extras_policy=extras_policy,
                )
                if schema is None:
                    result = ValidationResult(
                        passes=True,
                        message=f"Pandera unavailable for {self.table_key}",
                    )
                else:
                    frame = _pandera_frame(dataset)
                    if frame is None:
                        result = ValidationResult(
                            passes=True,
                            message=f"Skipping Pandera validation for {type(dataset).__name__}",
                        )
                    else:
                        error_types = pandera_error_types()
                        if not error_types:
                            result = ValidationResult(
                                passes=True,
                                message=f"Pandera error types unavailable for {self.table_key}",
                            )
                        else:
                            try:
                                schema_obj = cast("_PanderaSchemaProtocol", schema)
                                schema_obj.validate(frame, lazy=True)
                            except error_types as exc:
                                diagnostics = pandera_error_diagnostics(
                                    exc, table_key=self.table_key
                                )
                                result = ValidationResult(
                                    passes=False,
                                    message=f"Pandera validation failed for {self.table_key}",
                                    diagnostics=diagnostics,
                                )
                            else:
                                result = ValidationResult(
                                    passes=True,
                                    message=f"Pandera validation passed for {self.table_key}",
                                )
        return result


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


@dataclass(frozen=True, slots=True)
class TablePrimaryKeyValidator(DataValidator):
    """Validate primary key uniqueness for table outputs."""

    table_key: str
    primary_keys: tuple[str, ...]

    def __init__(
        self,
        *,
        table_key: str,
        primary_keys: Sequence[str],
        importance: str,
    ) -> None:
        object.__setattr__(self, "_importance", DataValidationLevel(importance))
        object.__setattr__(self, "table_key", table_key)
        object.__setattr__(self, "primary_keys", tuple(primary_keys))

    @classmethod
    def name(cls) -> str:
        """Return the canonical validator name.

        Returns
        -------
        str
            Validator name identifier.
        """
        return "table_primary_key_unique"

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
        return f"Validate primary keys for {self.table_key}"

    def validate(self, dataset: object) -> ValidationResult:
        """Validate primary key uniqueness for the dataset.

        Parameters
        ----------
        dataset
            Dataset to validate.

        Returns
        -------
        ValidationResult
            Validation outcome for the dataset.
        """
        if not self.primary_keys:
            return ValidationResult(
                passes=True,
                message=f"No primary key configured for {self.table_key}",
            )
        if dataset is None:
            return ValidationResult(
                passes=True,
                message=f"No rows to validate for {self.table_key}",
            )
        unique = _primary_key_unique(dataset, self.primary_keys)
        if unique is None:
            return ValidationResult(
                passes=True,
                message=f"Skipping primary key validation for {type(dataset).__name__}",
            )
        if not unique:
            return ValidationResult(
                passes=False,
                message=f"Primary key uniqueness violated for {self.table_key}",
                diagnostics={
                    "table_key": self.table_key,
                    "primary_keys": list(self.primary_keys),
                },
            )
        return ValidationResult(
            passes=True,
            message=f"Primary key validated for {self.table_key}",
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
        Validation profile name ("lenient" or "strict").
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
    normalized = profile.lower()
    if normalized == "strict":
        importance = DataValidationLevel.FAIL.value
        enforce_non_nullable = True
    elif normalized == "lenient":
        importance = DataValidationLevel.WARN.value
        enforce_non_nullable = False
    else:
        msg = f"Unsupported validation profile: {profile}"
        raise ValueError(msg)

    schema, _ = _resolve_table_schema(table_key)
    validators: list[DataValidator] = []
    if pandera_available():
        validators.append(PanderaSchemaValidator(table_key=table_key, importance=importance))
    else:
        validators.append(
            TableSchemaColumnsValidator(
                table_key=table_key,
                importance=importance,
                enforce_non_nullable=enforce_non_nullable,
            )
        )
    if isinstance(min_rows, int) and min_rows > 0:
        validators.append(
            TableRowCountValidator(
                table_key=table_key,
                min_rows=min_rows,
                importance=importance,
            )
        )
    if schema is not None and schema.primary_key and not pandera_available():
        validators.append(
            TablePrimaryKeyValidator(
                table_key=table_key,
                primary_keys=schema.primary_key,
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


def _column_names(dataset: object) -> list[str] | None:
    names: list[str] | None = None
    if isinstance(dataset, pl.LazyFrame):
        try:
            schema = dataset.collect_schema()
        except PolarsError:
            names = None
        else:
            names = _schema_column_names(schema)
    elif isinstance(dataset, pl.DataFrame):
        names = [str(name) for name in dataset.columns]
    else:
        schema = getattr(dataset, "schema", None)
        if schema is not None:
            names = _schema_column_names(schema)
        else:
            columns = getattr(dataset, "columns", None)
            if isinstance(columns, Sequence):
                names = [str(name) for name in columns]
    return names


def _pandera_frame(dataset: object) -> pl.LazyFrame | pl.DataFrame | None:
    if isinstance(dataset, (pl.LazyFrame, pl.DataFrame)):
        return dataset
    if isinstance(dataset, pa.Table):
        frame = pl.from_arrow(dataset)
        return frame if isinstance(frame, pl.DataFrame) else None
    if isinstance(dataset, pa.RecordBatch):
        frame = pl.from_arrow(pa.Table.from_batches([dataset]))
        return frame if isinstance(frame, pl.DataFrame) else None
    if isinstance(dataset, pa.RecordBatchReader):
        try:
            table = pa.Table.from_batches(list(cast("Iterable[pa.RecordBatch]", dataset)))
        except (TypeError, ValueError, pa.ArrowInvalid):
            return None
        frame = pl.from_arrow(table)
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


def _key_list(keys: Sequence[str]) -> list[str]:
    return [str(key) for key in keys]


def _primary_key_unique(dataset: object, keys: Sequence[str]) -> bool | None:
    if not keys:
        return True
    if isinstance(dataset, pl.LazyFrame):
        return _primary_key_unique_lazy(dataset, keys)
    if isinstance(dataset, pl.DataFrame):
        return _primary_key_unique_frame(dataset, keys)
    if isinstance(dataset, pa.Table):
        return _primary_key_unique_arrow(dataset, keys)
    if isinstance(dataset, pa.RecordBatch):
        table = pa.Table.from_batches([dataset])
        return _primary_key_unique_arrow(table, keys)
    return None


def _primary_key_unique_lazy(dataset: pl.LazyFrame, keys: Sequence[str]) -> bool | None:
    total = _row_count(dataset)
    if total is None:
        return None
    key_list = _key_list(keys)
    try:
        unique_count = dataset.select(pl.col(key_list)).unique().collect().height
    except PolarsError:
        return None
    return unique_count == total


def _primary_key_unique_frame(dataset: pl.DataFrame, keys: Sequence[str]) -> bool | None:
    key_list = _key_list(keys)
    try:
        unique_count = dataset.unique(subset=key_list).height
    except PolarsError:
        return None
    return unique_count == dataset.height


def _primary_key_unique_arrow(table: pa.Table, keys: Sequence[str]) -> bool | None:
    if not keys:
        return True
    key_list = _key_list(keys)
    try:
        arrays = [table.column(key).combine_chunks() for key in key_list]
        struct_arr = pa.StructArray.from_arrays(arrays, names=key_list)
    except (KeyError, TypeError, ValueError):
        return None
    hash_fn = getattr(pc, "hash", None)
    unique_fn = getattr(pc, "unique", None)
    if not callable(hash_fn) or not callable(unique_fn):
        return None
    try:
        unique = unique_fn(hash_fn(struct_arr))
    except (TypeError, ValueError):
        return None
    unique_count = _array_length(unique)
    return None if unique_count is None else table.num_rows == unique_count


def _schema_column_names(schema: object) -> list[str]:
    names_fn = getattr(schema, "names", None)
    if callable(names_fn):
        try:
            names = names_fn()
        except (TypeError, ValueError):
            names = None
        if isinstance(names, Sequence):
            return [str(name) for name in names]
        if names is not None:
            return [str(names)]
    if isinstance(schema, dict):
        return [str(name) for name in schema]
    if isinstance(schema, Sequence):
        return [str(name) for name in schema]
    return []


def _array_length(value: object) -> int | None:
    if isinstance(value, (pa.Array, pa.ChunkedArray)):
        return len(cast("Sized", value))
    return None


def _missing_columns(columns: Iterable[str], schema: TableSchema) -> list[str]:
    present = {str(name) for name in columns}
    return [col.name for col in schema.columns if col.name not in present]


def _non_nullable_columns(schema: TableSchema) -> list[str]:
    return [col.name for col in schema.columns if not col.nullable]


def _non_nullable_violations(dataset: object, columns: Sequence[str]) -> list[str]:
    if not columns:
        return []
    if isinstance(dataset, pl.DataFrame):
        return _non_nullable_violations_frame(dataset, columns)
    if isinstance(dataset, pl.LazyFrame):
        return _non_nullable_violations_lazy(dataset, columns)
    return []


def _non_nullable_violations_lazy(dataset: pl.LazyFrame, columns: Sequence[str]) -> list[str]:
    try:
        counts = dataset.select(
            [pl.col(name).null_count().alias(name) for name in columns]
        ).collect()
    except PolarsError:
        return []
    return _non_nullable_violations_frame(counts, columns)


def _non_nullable_violations_frame(dataset: pl.DataFrame, columns: Sequence[str]) -> list[str]:
    violations: list[str] = []
    if dataset.height == 0:
        return violations
    for name in columns:
        if name not in dataset.columns:
            continue
        value = dataset[name][0]
        if isinstance(value, int) and value > 0:
            violations.append(name)
    return violations


__all__ = [
    "PanderaSchemaValidator",
    "TablePrimaryKeyValidator",
    "TableRowCountValidator",
    "TableSchemaColumnsValidator",
    "build_table_schema_validators",
]
