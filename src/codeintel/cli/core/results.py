"""Structured result types for CLI handlers.

This module provides the CliResult protocol for handlers that return
structured results, enabling composition, testing, and consistent output.
All result types implement the SerializableResult protocol for unified
JSON serialization.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field, is_dataclass
from typing import (
    TYPE_CHECKING,
    ClassVar,
    Protocol,
    TypeVar,
    cast,
    runtime_checkable,
)

from codeintel.core.serialization.converters import (
    serialize_dataclass_to_dict as serialize_result,
)
from codeintel.core.serialization.stable import stable_json_value

if TYPE_CHECKING:
    from codeintel.cli.errors import ProblemDetail


T_co = TypeVar("T_co", covariant=True)


class _DataclassInstance(Protocol):
    __dataclass_fields__: ClassVar[dict[str, dataclasses.Field[object]]]


class _ResultTypeClass(Protocol):
    RESULT_TYPE_GENERATED: ClassVar[bool]
    to_dict: ClassVar[object]


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

    Examples
    --------
    >>> from dataclasses import dataclass
    >>> from codeintel.cli.core.results import result_type
    >>> @result_type
    ... @dataclass(frozen=True)
    ... class MyResult:
    ...     name: str
    ...     count: int
    ...     details: str | None = None
    >>> result = MyResult(name="test", count=5)
    >>> result.to_dict()
    {'name': 'test', 'count': 5}
    >>> result2 = MyResult(name="test", count=5, details="info")
    >>> result2.to_dict()
    {'name': 'test', 'count': 5, 'details': 'info'}
    """
    if not is_dataclass(cls):
        msg = f"@result_type requires a dataclass, got {cls.__name__}"
        raise TypeError(msg)

    result_cls = cast("type[_ResultTypeClass]", cls)

    def to_dict(self: _DataclassInstance) -> dict[str, object]:
        """Auto-generated serialization that omits None fields.

        Returns
        -------
        dict[str, object]
            Dictionary with non-None field values.
        """
        return _serialize_dataclass(self)

    if not hasattr(result_cls, "RESULT_TYPE_GENERATED"):
        result_cls.RESULT_TYPE_GENERATED = False

    if not hasattr(result_cls, "to_dict") or getattr(result_cls, "RESULT_TYPE_GENERATED", False):
        result_cls.to_dict = to_dict
        result_cls.RESULT_TYPE_GENERATED = True

    return cls


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
    serialized = stable_json_value(
        value,
        omit_none=True,
        omit_private_fields=True,
    )
    if isinstance(serialized, dict):
        return cast("dict[str, object]", serialized)
    return {}


def ensure_serializable[T](cls: type[T]) -> type[T]:
    """Ensure a result type has to_dict() - add if missing.

    For gradual migration: existing types with manual to_dict() pass through,
    new types get auto-generated to_dict() via @result_type.

    Parameters
    ----------
    cls
        The class to ensure has serialization.

    Returns
    -------
    type[T]
        The class (possibly enhanced).

    Examples
    --------
    >>> from dataclasses import dataclass
    >>> @dataclass(frozen=True)
    ... class ExistingResult:
    ...     value: int
    ...
    ...     def to_dict(self) -> dict[str, object]:
    ...         return {"val": self.value}
    >>> ensure_serializable(ExistingResult)(1).to_dict()
    {'val': 1}
    """
    if hasattr(cls, "to_dict"):
        return cls
    return result_type(cls)


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
    if isinstance(data, SerializableResult):
        return data.to_dict()

    if is_dataclass(data) and not isinstance(data, type):
        return serialize_result(data)

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
        return json.dumps(self.to_dict(), indent=indent, default=str)

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
    "SerializableResult",
    "auto_serialize",
    "ensure_serializable",
    "result_type",
]
