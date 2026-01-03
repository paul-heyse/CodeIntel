"""Structured result types for CLI handlers.

This module provides the CliResult protocol for handlers that return
structured results, enabling composition, testing, and consistent output.
All result types implement the SerializableResult protocol for unified
JSON serialization.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    ClassVar,
    Protocol,
    TypeVar,
    runtime_checkable,
)

import msgspec

from codeintel.core.columnar.stream import ColumnarStream
from codeintel.core.constants import DEFAULT_ARROW_BATCH_SIZE
from codeintel.core.query_results import records_from_arrow_reader
from codeintel.core.serialization.msgspec import encode_json_text, to_builtins

if TYPE_CHECKING:
    from codeintel.cli.errors import ProblemDetail


T_co = TypeVar("T_co", covariant=True)


class ResultBase(msgspec.Struct):
    """Base class for msgspec-backed CLI result types."""

    __include_none_fields__: ClassVar[frozenset[str]] = frozenset()
    __result_key_map__: ClassVar[dict[str, str]] = {}

    def to_dict(self) -> dict[str, object]:
        """Serialize the result type to a dict with optional key overrides.

        Returns
        -------
        dict[str, object]
            Serialized result payload.
        """
        return _serialize_result_struct(self)


def _serialize_result_struct(value: ResultBase) -> dict[str, object]:
    """Serialize a msgspec Struct result type to dictionary.

    Parameters
    ----------
    value
        Result type instance to serialize.

    Returns
    -------
    dict[str, object]
        Dictionary with non-None field values.
    """
    result: dict[str, object] = {}
    key_map = value.__result_key_map__
    include_none = value.__include_none_fields__
    for field_info in msgspec.structs.fields(value):
        field_name = field_info.name
        key = key_map.get(field_name, field_name)
        field_value = getattr(value, field_name)
        if field_value is None and field_name not in include_none:
            continue
        result[key] = _serialize_value(field_value)
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
    result: object
    if value is None:
        result = None
    elif isinstance(value, (ResultBase, SerializableResult)):
        result = value.to_dict()
    elif isinstance(value, (list, tuple)):
        result = [_serialize_value(item) for item in value]
    elif isinstance(value, dict):
        result = {key: _serialize_value(item) for key, item in value.items() if item is not None}
    else:
        result = to_builtins(value)
    return result


@runtime_checkable
class SerializableResult(Protocol):
    """Protocol for CLI result types that can serialize to dictionary.

    All result dataclasses should implement this protocol by providing
    a `to_dict()` method. This enables unified JSON serialization.

    Examples
    --------
    >>> from dataclasses import dataclass
    >>> from codeintel.cli.core.results import SerializableResult
    >>> @dataclass
    ... class MyResult:
    ...     value: int
    ...
    ...     def to_dict(self) -> dict[str, object]:
    ...         return {"value": self.value}
    >>> isinstance(MyResult(1), SerializableResult)
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


def auto_serialize(data: object) -> object:
    """Automatically serialize data for JSON output.

    Handle different data types with the following precedence:
    1. SerializableResult protocol (call to_dict())
    2. ColumnarStream (materialize for JSON output)
    3. msgspec-compatible values via to_builtins
    4. Objects with __dict__ (use dict directly)
    5. Return as-is (primitives, lists, dicts)

    Parameters
    ----------
    data
        Data to serialize.

    Returns
    -------
    object
        Serialized representation suitable for JSON.

    """
    if isinstance(data, ColumnarStream):
        reader = data.to_reader(batch_size=DEFAULT_ARROW_BATCH_SIZE)
        return records_from_arrow_reader(reader)
    if isinstance(data, SerializableResult):
        return data.to_dict()
    try:
        return to_builtins(data)
    except TypeError:
        if hasattr(data, "__dict__") and not isinstance(data, type):
            return data.__dict__
        return data


@dataclass
class CliResult[T_co]:
    """Structured result from a CLI handler.

    Encapsulates success/failure status, data payload, and warnings
    for consistent rendering and composition.

    Parameters
    ----------
    success
        Whether the operation completed successfully.
    data
        Result data payload (type varies by handler).
    error
        Problem details if the operation failed.
    warnings
        Non-fatal warnings to display.
    metadata
        Additional metadata about the operation.
    """

    success: bool
    data: T_co | None = None
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
            result["data"] = self._serialize_data(self.data)

        if self.error is not None:
            result["error"] = self.error.to_dict()

        if self.warnings:
            result["warnings"] = self.warnings

        if self.metadata:
            result["metadata"] = self.metadata

        return result

    @staticmethod
    def _serialize_data(data: object) -> object:
        """Serialize data for JSON output.

        Use the unified `auto_serialize` function to handle all result types
        consistently, supporting SerializableResult protocol, dataclasses,
        and plain objects.

        Returns
        -------
        object
            Serialized representation of the data.
        """
        return auto_serialize(data)

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
        return encode_json_text(self.to_dict(), indent=indent)

    @classmethod
    def ok(cls, data: T_co, *, metadata: dict[str, object] | None = None) -> CliResult[T_co]:
        """Create a successful result.

        Parameters
        ----------
        data
            Result data payload.
        metadata
            Optional metadata about the operation.

        Returns
        -------
        CliResult[T]
            Successful result with the given data.
        """
        return cls(
            success=True,
            data=data,
            metadata=metadata or {},
        )

    @classmethod
    def fail(
        cls,
        error: ProblemDetail,
        *,
        warnings: list[str] | None = None,
    ) -> CliResult[T_co]:
        """Create a failed result.

        Parameters
        ----------
        error
            Problem details describing the failure.
        warnings
            Optional warnings to include.

        Returns
        -------
        CliResult[T]
            Failed result with the given error.
        """
        return cls(
            success=False,
            error=error,
            warnings=warnings or [],
        )


__all__ = [
    "CliResult",
    "ResultBase",
    "SerializableResult",
    "auto_serialize",
]
