"""Input validation layer for CLI operations.

Provide composable validators for CLI inputs with structured error reporting.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ValidationError:
    """Validation error details.

    Parameters
    ----------
    field
        Field name that failed validation.
    message
        Human-readable error message.
    code
        Machine-readable error code.
    value
        The invalid value (if safe to include).
    """

    field: str
    message: str
    code: str
    value: str | None = None

    def to_dict(self) -> dict[str, str | None]:
        """Convert to dictionary for JSON serialization.

        Returns
        -------
        dict[str, str | None]
            Dictionary representation.
        """
        return {
            "field": self.field,
            "message": self.message,
            "code": self.code,
            "value": self.value,
        }


@dataclass
class ValidationResult[T]:
    """Result of validation.

    Parameters
    ----------
    value
        The validated value (if valid).
    errors
        List of validation errors.
    """

    value: T | None = None
    errors: list[ValidationError] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        """Check if validation passed.

        Returns
        -------
        bool
            True if no errors.
        """
        return len(self.errors) == 0

    @classmethod
    def ok(cls, value: T) -> ValidationResult[T]:
        """Create a successful validation result.

        Parameters
        ----------
        value
            The validated value.

        Returns
        -------
        ValidationResult[T]
            Successful result.
        """
        return cls(value=value, errors=[])

    @classmethod
    def fail(cls, errors: list[ValidationError]) -> ValidationResult[T]:
        """Create a failed validation result.

        Parameters
        ----------
        errors
            List of validation errors.

        Returns
        -------
        ValidationResult[T]
            Failed result.
        """
        return cls(value=None, errors=errors)


class Validator[T](ABC):
    """Base class for validators."""

    @abstractmethod
    def validate(self, value: object, field_name: str) -> ValidationResult[T]:
        """Validate a value.

        Parameters
        ----------
        value
            Value to validate.
        field_name
            Name of the field being validated.

        Returns
        -------
        ValidationResult[T]
            Validation result.
        """
        ...


class StringValidator(Validator[str]):
    """Validate string inputs.

    Parameters
    ----------
    min_length
        Minimum string length (None for no minimum).
    max_length
        Maximum string length (None for no maximum).
    pattern
        Regex pattern to match (None for no pattern).
    allowed_values
        Set of allowed values (None for any value).
    """

    def __init__(
        self,
        *,
        min_length: int | None = None,
        max_length: int | None = None,
        pattern: str | None = None,
        allowed_values: set[str] | None = None,
    ) -> None:
        """Initialize string validator."""
        self._min_length = min_length
        self._max_length = max_length
        self._pattern = re.compile(pattern) if pattern else None
        self._allowed_values = allowed_values

    def validate(self, value: object, field_name: str) -> ValidationResult[str]:
        """Validate a string value.

        Parameters
        ----------
        value
            Value to validate.
        field_name
            Name of the field being validated.

        Returns
        -------
        ValidationResult[str]
            Validation result.
        """
        errors: list[ValidationError] = []

        if not isinstance(value, str):
            errors.append(
                ValidationError(
                    field=field_name,
                    message=f"Expected string, got {type(value).__name__}",
                    code="invalid_type",
                ),
            )
            return ValidationResult.fail(errors)

        str_value: str = value

        if self._min_length is not None and len(str_value) < self._min_length:
            errors.append(
                ValidationError(
                    field=field_name,
                    message=f"String must be at least {self._min_length} characters",
                    code="min_length",
                    value=str_value[:50],
                ),
            )

        if self._max_length is not None and len(str_value) > self._max_length:
            errors.append(
                ValidationError(
                    field=field_name,
                    message=f"String must be at most {self._max_length} characters",
                    code="max_length",
                    value=str_value[:50],
                ),
            )

        if self._pattern is not None and not self._pattern.match(str_value):
            errors.append(
                ValidationError(
                    field=field_name,
                    message=f"String must match pattern: {self._pattern.pattern}",
                    code="pattern",
                    value=str_value[:50],
                ),
            )

        if self._allowed_values is not None and str_value not in self._allowed_values:
            errors.append(
                ValidationError(
                    field=field_name,
                    message=f"Value must be one of: {', '.join(sorted(self._allowed_values))}",
                    code="not_allowed",
                    value=str_value[:50],
                ),
            )

        if errors:
            return ValidationResult.fail(errors)
        return ValidationResult.ok(str_value)


class PathValidator(Validator[Path]):
    """Validate path inputs.

    Parameters
    ----------
    must_exist
        Whether the path must exist.
    must_be_file
        Whether the path must be a file (None for no check).
    must_be_dir
        Whether the path must be a directory (None for no check).
    allowed_extensions
        Set of allowed file extensions (None for any).
    """

    def __init__(
        self,
        *,
        must_exist: bool = False,
        must_be_file: bool | None = None,
        must_be_dir: bool | None = None,
        allowed_extensions: set[str] | None = None,
    ) -> None:
        """Initialize path validator."""
        self._must_exist = must_exist
        self._must_be_file = must_be_file
        self._must_be_dir = must_be_dir
        self._allowed_extensions = allowed_extensions

    def validate(self, value: object, field_name: str) -> ValidationResult[Path]:
        """Validate a path value.

        Parameters
        ----------
        value
            Value to validate (str or Path).
        field_name
            Name of the field being validated.

        Returns
        -------
        ValidationResult[Path]
            Validation result.
        """
        errors: list[ValidationError] = []

        if isinstance(value, str):
            path = Path(value)
        elif isinstance(value, Path):
            path = value
        else:
            errors.append(
                ValidationError(
                    field=field_name,
                    message=f"Expected path, got {type(value).__name__}",
                    code="invalid_type",
                ),
            )
            return ValidationResult.fail(errors)

        if self._must_exist and not path.exists():
            errors.append(
                ValidationError(
                    field=field_name,
                    message="Path does not exist",
                    code="not_exists",
                    value=str(path),
                ),
            )
            return ValidationResult.fail(errors)

        if self._must_be_file is True and path.exists() and not path.is_file():
            errors.append(
                ValidationError(
                    field=field_name,
                    message="Path must be a file",
                    code="not_file",
                    value=str(path),
                ),
            )

        if self._must_be_dir is True and path.exists() and not path.is_dir():
            errors.append(
                ValidationError(
                    field=field_name,
                    message="Path must be a directory",
                    code="not_dir",
                    value=str(path),
                ),
            )

        if self._allowed_extensions is not None:
            ext = path.suffix.lower()
            if ext not in self._allowed_extensions:
                errors.append(
                    ValidationError(
                        field=field_name,
                        message=f"File extension must be one of: {', '.join(sorted(self._allowed_extensions))}",
                        code="invalid_extension",
                        value=str(path),
                    ),
                )

        if errors:
            return ValidationResult.fail(errors)
        return ValidationResult.ok(path)


class IntValidator(Validator[int]):
    """Validate integer inputs.

    Parameters
    ----------
    min_value
        Minimum value (None for no minimum).
    max_value
        Maximum value (None for no maximum).
    """

    def __init__(
        self,
        *,
        min_value: int | None = None,
        max_value: int | None = None,
    ) -> None:
        """Initialize integer validator."""
        self._min_value = min_value
        self._max_value = max_value

    def validate(self, value: object, field_name: str) -> ValidationResult[int]:
        """Validate an integer value.

        Parameters
        ----------
        value
            Value to validate.
        field_name
            Name of the field being validated.

        Returns
        -------
        ValidationResult[int]
            Validation result.
        """
        errors: list[ValidationError] = []

        if not isinstance(value, int) or isinstance(value, bool):
            errors.append(
                ValidationError(
                    field=field_name,
                    message=f"Expected integer, got {type(value).__name__}",
                    code="invalid_type",
                ),
            )
            return ValidationResult.fail(errors)

        int_value: int = value

        if self._min_value is not None and int_value < self._min_value:
            errors.append(
                ValidationError(
                    field=field_name,
                    message=f"Value must be at least {self._min_value}",
                    code="min_value",
                    value=str(int_value),
                ),
            )

        if self._max_value is not None and int_value > self._max_value:
            errors.append(
                ValidationError(
                    field=field_name,
                    message=f"Value must be at most {self._max_value}",
                    code="max_value",
                    value=str(int_value),
                ),
            )

        if errors:
            return ValidationResult.fail(errors)
        return ValidationResult.ok(int_value)


class FloatValidator(Validator[float]):
    """Validate float inputs.

    Parameters
    ----------
    min_value
        Minimum value (None for no minimum).
    max_value
        Maximum value (None for no maximum).
    """

    def __init__(
        self,
        *,
        min_value: float | None = None,
        max_value: float | None = None,
    ) -> None:
        """Initialize float validator."""
        self._min_value = min_value
        self._max_value = max_value

    def validate(self, value: object, field_name: str) -> ValidationResult[float]:
        """Validate a float value.

        Parameters
        ----------
        value
            Value to validate.
        field_name
            Name of the field being validated.

        Returns
        -------
        ValidationResult[float]
            Validation result.
        """
        errors: list[ValidationError] = []

        if not isinstance(value, (int, float)) or isinstance(value, bool):
            errors.append(
                ValidationError(
                    field=field_name,
                    message=f"Expected number, got {type(value).__name__}",
                    code="invalid_type",
                ),
            )
            return ValidationResult.fail(errors)

        float_value: float = float(value)

        if self._min_value is not None and float_value < self._min_value:
            errors.append(
                ValidationError(
                    field=field_name,
                    message=f"Value must be at least {self._min_value}",
                    code="min_value",
                    value=str(float_value),
                ),
            )

        if self._max_value is not None and float_value > self._max_value:
            errors.append(
                ValidationError(
                    field=field_name,
                    message=f"Value must be at most {self._max_value}",
                    code="max_value",
                    value=str(float_value),
                ),
            )

        if errors:
            return ValidationResult.fail(errors)
        return ValidationResult.ok(float_value)


@dataclass
class ValidationSchema:
    """Schema for validating multiple fields.

    Parameters
    ----------
    validators
        Mapping of field names to validators.
    """

    validators: dict[str, Validator[Any]] = field(default_factory=dict)

    def add[T](self, field_name: str, validator: Validator[T]) -> ValidationSchema:
        """Add a validator to the schema.

        Parameters
        ----------
        field_name
            Field name.
        validator
            Validator instance.

        Returns
        -------
        ValidationSchema
            Self for chaining.
        """
        self.validators[field_name] = validator
        return self

    def validate(self, data: dict[str, object]) -> ValidationResult[dict[str, object]]:
        """Validate data against the schema.

        Parameters
        ----------
        data
            Data to validate.

        Returns
        -------
        ValidationResult[dict[str, object]]
            Validation result with validated data or errors.
        """
        all_errors: list[ValidationError] = []
        validated: dict[str, object] = {}

        for field_name, validator in self.validators.items():
            if field_name not in data:
                all_errors.append(
                    ValidationError(
                        field=field_name,
                        message="Required field is missing",
                        code="required",
                    ),
                )
                continue

            result = validator.validate(data[field_name], field_name)
            if result.is_valid:
                validated[field_name] = result.value
            else:
                all_errors.extend(result.errors)

        if all_errors:
            return ValidationResult.fail(all_errors)
        return ValidationResult.ok(validated)


# Common validators
OPERATION_ID_VALIDATOR = StringValidator(
    min_length=1,
    max_length=100,
    pattern=r"^[a-z][a-z0-9_\.]*$",
)

TABLE_KEY_VALIDATOR = StringValidator(
    min_length=1,
    max_length=200,
    pattern=r"^[a-z_][a-z0-9_\.]*$",
)

PATH_VALIDATOR = PathValidator()

EXISTING_PATH_VALIDATOR = PathValidator(must_exist=True)

EXISTING_FILE_VALIDATOR = PathValidator(must_exist=True, must_be_file=True)

EXISTING_DIR_VALIDATOR = PathValidator(must_exist=True, must_be_dir=True)


def validate_operation_id(op_id: str) -> ValidationResult[str]:
    """Validate an operation ID.

    Parameters
    ----------
    op_id
        Operation identifier to validate.

    Returns
    -------
    ValidationResult[str]
        Validation result.
    """
    return OPERATION_ID_VALIDATOR.validate(op_id, "operation_id")


def validate_table_key(table_key: str) -> ValidationResult[str]:
    """Validate a table key.

    Parameters
    ----------
    table_key
        Table key to validate.

    Returns
    -------
    ValidationResult[str]
        Validation result.
    """
    return TABLE_KEY_VALIDATOR.validate(table_key, "table_key")


def validate_path(
    path: str | Path,
    *,
    must_exist: bool = False,
    must_be_file: bool | None = None,
    must_be_dir: bool | None = None,
) -> ValidationResult[Path]:
    """Validate a path.

    Parameters
    ----------
    path
        Path to validate.
    must_exist
        Whether the path must exist.
    must_be_file
        Whether the path must be a file.
    must_be_dir
        Whether the path must be a directory.

    Returns
    -------
    ValidationResult[Path]
        Validation result.
    """
    validator = PathValidator(
        must_exist=must_exist,
        must_be_file=must_be_file,
        must_be_dir=must_be_dir,
    )
    return validator.validate(path, "path")


__all__ = [
    "EXISTING_DIR_VALIDATOR",
    "EXISTING_FILE_VALIDATOR",
    "EXISTING_PATH_VALIDATOR",
    "OPERATION_ID_VALIDATOR",
    "PATH_VALIDATOR",
    "TABLE_KEY_VALIDATOR",
    "FloatValidator",
    "IntValidator",
    "PathValidator",
    "StringValidator",
    "ValidationError",
    "ValidationResult",
    "ValidationSchema",
    "Validator",
    "validate_operation_id",
    "validate_path",
    "validate_table_key",
]
