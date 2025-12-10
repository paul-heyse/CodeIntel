"""Result type and serialization for operations.

Provides a typed Result container and @result_type decorator for
automatic serialization of result dataclasses.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from dataclasses import dataclass, field, is_dataclass
from enum import Enum
from typing import TYPE_CHECKING, Protocol, TextIO, TypeGuard, runtime_checkable

if TYPE_CHECKING:
    from codeintel.operations.errors.problem_detail import ProblemDetail


# Text renderer protocol: callable that takes data and writer
type TextRenderer = Callable[[object, TextIO], None]


class _DataclassInstance(Protocol):
    __dataclass_fields__: dict[str, object]


@runtime_checkable
class Serializable(Protocol):
    """Protocol for types that can serialize to dictionary.

    Result dataclasses should implement this protocol by providing
    a `to_dict()` method. This enables unified JSON serialization.

    Examples
    --------
    >>> from dataclasses import dataclass
    >>> @dataclass
    ... class MyResult:
    ...     value: int
    ...
    ...     def to_dict(self) -> dict[str, object]:
    ...         return {"value": self.value}
    >>> isinstance(MyResult(1), Serializable)
    True
    """

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary for JSON serialization.

        Returns
        -------
        dict[str, object]
            Dictionary representation of the result.
        """
        ...


def result_type[T](cls: type[T]) -> type[T]:
    """Add auto-generated to_dict() that omits None fields.

    Apply this decorator to frozen dataclasses that serve as result types.
    The generated `to_dict()` method serializes all fields, recursively
    handling nested result types and omitting fields with None values.

    Parameters
    ----------
    cls
        The dataclass to enhance with auto-serialization.

    Returns
    -------
    type[T]
        The enhanced dataclass.

    Raises
    ------
    TypeError
        If cls is not a dataclass.
    """
    if not is_dataclass(cls):
        msg = f"@result_type requires a dataclass, got {cls.__name__}"
        raise TypeError(msg)

    def to_dict(self: object) -> dict[str, object]:
        """Auto-generated serialization that omits None fields.

        Returns
        -------
        dict[str, object]
            Dictionary with non-None field values.
        """
        return _serialize_dataclass(self)

    # Avoid overwriting existing to_dict implementations
    if not hasattr(cls, "to_dict") or getattr(cls, "_result_type_generated", False):
        setattr(cls, "to_dict", to_dict)  # noqa: B010
        setattr(cls, "_result_type_generated", True)  # noqa: B010

    return cls


def _is_dataclass_instance(value: object) -> TypeGuard[_DataclassInstance]:
    """Return True when value is a dataclass instance (not a class)."""
    return is_dataclass(value) and not isinstance(value, type)


def _serialize_dataclass(value: _DataclassInstance) -> dict[str, object]:
    """Serialize a dataclass instance to dictionary.

    Parameters
    ----------
    value
        Dataclass instance to serialize.

    Returns
    -------
    dict[str, object]
        Dictionary with non-None, non-private field values.
    """
    from dataclasses import fields as get_fields  # noqa: PLC0415

    result: dict[str, object] = {}
    for fld in get_fields(value):
        if fld.name.startswith("_"):
            continue
        field_value = getattr(value, fld.name)
        if field_value is None:
            continue
        result[fld.name] = _serialize_value(field_value)
    return result


def _serialize_value(value: object) -> object:
    """Recursively serialize a value for JSON output.

    Handle nested dataclasses, lists, dicts, and primitives.

    Parameters
    ----------
    value
        Value to serialize.

    Returns
    -------
    object
        Serialized value suitable for JSON.
    """
    if value is None:
        return None

    if _is_dataclass_instance(value):
        return _serialize_dataclass(value)

    if isinstance(value, (list, tuple)):
        return [_serialize_value(item) for item in value]

    if isinstance(value, dict):
        return {k: _serialize_value(v) for k, v in value.items() if v is not None}

    if isinstance(value, Serializable):
        return value.to_dict()

    if isinstance(value, Enum):
        return value.value

    return _serialize_primitive(value)


def _serialize_primitive(value: object) -> object:
    """Serialize primitive and special types.

    Parameters
    ----------
    value
        Value to serialize.

    Returns
    -------
    object
        Serialized value.
    """
    from pathlib import Path  # noqa: PLC0415

    # Handle Path
    if isinstance(value, Path):
        return str(value)

    # Primitives pass through
    return value


def auto_serialize(data: object) -> object:
    """Automatically serialize data for JSON output.

    Handle different data types with the following precedence:
    1. Serializable protocol (call to_dict())
    2. Dataclass (serialize all fields)
    3. Objects with __dict__ (use dict directly)
    4. Return as-is (primitives, lists, dicts)

    Parameters
    ----------
    data
        Data to serialize.

    Returns
    -------
    object
        Serialized representation suitable for JSON.

    Examples
    --------
    >>> from dataclasses import dataclass
    >>> @dataclass
    ... class Simple:
    ...     x: int
    >>> auto_serialize(Simple(42))
    {'x': 42}
    """
    # Serializable protocol takes precedence
    if isinstance(data, Serializable):
        return data.to_dict()

    # Dataclass serialization
    if _is_dataclass_instance(data):
        return _serialize_dataclass(data)

    # Objects with __dict__
    if hasattr(data, "__dict__") and not isinstance(data, type):
        return data.__dict__

    # Return as-is (primitives, lists, dicts)
    return data


@dataclass
class Result[T]:
    """Typed result from an operation.

    Encapsulates success/failure status, data payload, and warnings
    for consistent rendering and composition.

    Type Parameters
    ---------------
    T
        The type of the data payload on success.

    Parameters
    ----------
    success
        Whether the operation completed successfully.
    data
        Result data payload (type varies by operation).
    error
        Problem details if the operation failed.
    warnings
        Non-fatal warnings to display.
    metadata
        Additional metadata about the operation.

    Example
    -------
    >>> from codeintel.operations.result import Result
    >>> result = Result.ok({"count": 5})
    >>> result.success
    True
    >>> result.data
    {'count': 5}
    """

    success: bool
    data: T | None = None
    error: ProblemDetail | None = None
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary for JSON serialization.

        Returns
        -------
        dict[str, object]
            Dictionary representation with data, metadata, and warnings.
        """
        result: dict[str, object] = {
            "success": self.success,
        }

        if self.data is not None:
            result["data"] = auto_serialize(self.data)

        if self.error is not None:
            result["error"] = self.error.to_dict()

        if self.warnings:
            result["warnings"] = self.warnings

        if self.metadata:
            result["metadata"] = self.metadata

        return result

    def to_json(self, *, indent: int | None = 2) -> str:
        """Serialize to JSON string.

        Parameters
        ----------
        indent
            JSON indentation level (None for compact output).

        Returns
        -------
        str
            JSON representation of the result.
        """
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def render(
        self,
        output_format: str,
        writer: TextIO = sys.stdout,
    ) -> None:
        """Render the result to the specified writer.

        Parameters
        ----------
        output_format
            Output format ("text" or "json").
        writer
            Text writer for output (default: stdout).
        """
        self._write_warnings()

        if output_format.lower() == "json":
            self._render_json(writer)
        elif self.data is not None:
            self._render_text(writer)

    def _write_warnings(self) -> None:
        """Write warnings to stderr."""
        for warning in self.warnings:
            sys.stderr.write(f"Warning: {warning}\n")

    def _render_json(self, writer: TextIO) -> None:
        """Render as JSON."""
        writer.write(self.to_json())
        writer.write("\n")

    def _render_text(self, writer: TextIO) -> None:
        """Render data as text."""
        data = self.data
        if isinstance(data, str):
            writer.write(data)
            if not data.endswith("\n"):
                writer.write("\n")
        elif isinstance(data, list):
            for item in data:
                writer.write(f"{item}\n")
        elif isinstance(data, dict):
            for key, value in data.items():
                writer.write(f"{key}: {value}\n")
        else:
            writer.write(f"{data}\n")

    @classmethod
    def ok(
        cls,
        data: T,
        *,
        metadata: dict[str, object] | None = None,
        warnings: list[str] | None = None,
    ) -> Result[T]:
        """Create a successful result.

        Parameters
        ----------
        data
            Result data payload.
        metadata
            Optional metadata about the operation.
        warnings
            Optional warnings to include.

        Returns
        -------
        Result[T]
            Successful result with the given data.
        """
        return cls(
            success=True,
            data=data,
            metadata=metadata or {},
            warnings=warnings or [],
        )

    @classmethod
    def fail(
        cls,
        error: ProblemDetail,
        *,
        warnings: list[str] | None = None,
    ) -> Result[T]:
        """Create a failed result.

        Parameters
        ----------
        error
            Problem details describing the failure.
        warnings
            Optional warnings to include.

        Returns
        -------
        Result[T]
            Failed result with the given error.
        """
        return cls(
            success=False,
            error=error,
            warnings=warnings or [],
        )


__all__ = [
    "Result",
    "Serializable",
    "auto_serialize",
    "result_type",
]
