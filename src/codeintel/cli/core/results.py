"""Structured result types for CLI handlers.

This module provides the CliResult protocol for handlers that return
structured results, enabling composition, testing, and consistent output.
All result types implement the SerializableResult protocol for unified
JSON serialization.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, is_dataclass
from dataclasses import fields as get_fields
from enum import Enum
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    ClassVar,
    Protocol,
    TypeGuard,
    TypeVar,
    cast,
    runtime_checkable,
)

import msgspec

from codeintel.core.columnar.stream import ColumnarStream
from codeintel.core.constants import DEFAULT_ARROW_BATCH_SIZE
from codeintel.core.query_results import records_from_arrow_reader
from codeintel.core.serialization.converters import (
    serialize_dataclass_to_dict as serialize_result,
)
from codeintel.core.serialization.msgspec_json import encode_json_text

if TYPE_CHECKING:
    from codeintel.cli.errors import ProblemDetail


T_co = TypeVar("T_co", covariant=True)


def _encode_hook(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    msg = f"Unsupported type: {type(value).__name__}"
    raise TypeError(msg)


class _DataclassInstance(Protocol):
    __dataclass_fields__: ClassVar[dict[str, dataclasses.Field[object]]]


class _ResultTypeClass(Protocol):
    RESULT_TYPE_GENERATED: ClassVar[bool]
    to_dict: ClassVar[object]


class ResultBase(msgspec.Struct, frozen=True):
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


def _is_dataclass_instance(value: object) -> TypeGuard[_DataclassInstance]:
    """Return True when value is a dataclass instance (not a class).

    Returns
    -------
    bool
        True if value is a dataclass instance.
    """
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
    result: dict[str, object] = {}

    for fld in get_fields(value):
        if fld.name.startswith("_"):
            continue
        key_override = fld.metadata.get("result_key")
        key = key_override if isinstance(key_override, str) else fld.name
        field_value = getattr(value, fld.name)
        if field_value is None:
            if fld.metadata.get("include_none", False):
                result[key] = None
            continue
        result[key] = _serialize_value(field_value)
    return result


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
    for field_info in msgspec.structs.fields(type(value)):
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
    elif isinstance(value, ResultBase) or isinstance(value, SerializableResult):
        result = value.to_dict()
    elif _is_dataclass_instance(value):
        result = _serialize_dataclass(value)
    elif isinstance(value, (list, tuple)):
        result = [_serialize_value(item) for item in value]
    elif isinstance(value, dict):
        result = {key: _serialize_value(item) for key, item in value.items() if item is not None}
    else:
        result = _serialize_primitive(value)
    return result


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
    if isinstance(value, Enum):
        return value.value

    if isinstance(value, Path):
        return str(value)

    return value


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
    if isinstance(data, ColumnarStream):
        reader = data.to_reader(batch_size=DEFAULT_ARROW_BATCH_SIZE)
        return records_from_arrow_reader(reader)
    if isinstance(data, SerializableResult):
        return data.to_dict()
    if is_dataclass(data) and not isinstance(data, type):
        return serialize_result(data)
    if hasattr(data, "__dict__") and not isinstance(data, type):
        return data.__dict__
    try:
        return msgspec.to_builtins(data, enc_hook=_encode_hook)
    except TypeError:
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
    "ensure_serializable",
    "result_type",
]
