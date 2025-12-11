"""Unified parameter coercion service.

Consolidate parameter type coercion from:
- ``handlers/context.py`` (ParameterAccessors)
- ``config/env.py`` (_convert_value)
- ``introspection/params.py`` (coerce_string_param)
- ``commands/decorators.py`` (_convert_to_path, _get_output_format)

All methods delegate to ``core/parsing.py`` internally for consistency.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import TypeVar

from codeintel.cli.core.parsing import is_truthy_string, parse_bool, parse_cli_value
from codeintel.cli.rendering.types import OutputFormat

E = TypeVar("E", bound=Enum)


class ParamError(ValueError):
    """Error raised when parameter coercion fails.

    Parameters
    ----------
    key
        The parameter key that caused the error.
    message
        Human-readable error message.
    """

    def __init__(self, key: str, message: str) -> None:
        """Initialize the error."""
        super().__init__(message)
        self.key = key


class ParamService:
    """Unified parameter access and coercion service.

    Provide type-safe access to raw parameter dictionaries with consistent
    coercion behavior across the CLI.

    Parameters
    ----------
    raw_params
        Raw parameters dictionary.

    Examples
    --------
    >>> params = ParamService({"name": "example", "count": "5", "enabled": "true"})
    >>> params.get_str("name")
    'example'
    >>> params.get_int("count")
    5
    >>> params.get_bool("enabled")
    True
    """

    def __init__(self, raw_params: dict[str, object] | None = None) -> None:
        """Initialize with raw parameters."""
        self._params: dict[str, object] = raw_params or {}

    @property
    def raw(self) -> dict[str, object]:
        """Get raw parameters dictionary (read-only view).

        Returns
        -------
        dict[str, object]
            Raw parameters.
        """
        return self._params

    def get_str(self, key: str, default: str | None = None) -> str | None:
        """Get parameter as string.

        Parameters
        ----------
        key
            Parameter name.
        default
            Default value if missing.

        Returns
        -------
        str | None
            String value or default.

        Examples
        --------
        >>> params = ParamService({"name": "test"})
        >>> params.get_str("name")
        'test'
        >>> params.get_str("missing", "default")
        'default'
        """
        value = self._params.get(key)
        if value is None:
            return default
        return str(value)

    def get_int(self, key: str, default: int = 0) -> int:
        """Get parameter as integer.

        Parameters
        ----------
        key
            Parameter name.
        default
            Default value if missing or invalid.

        Returns
        -------
        int
            Integer value or default.

        Examples
        --------
        >>> params = ParamService({"count": "42"})
        >>> params.get_int("count")
        42
        >>> params.get_int("missing", 10)
        10
        """
        value = self._params.get(key)
        if value is None:
            return default
        if isinstance(value, int) and not isinstance(value, bool):
            return value
        if isinstance(value, bool):
            return default
        if isinstance(value, (str, float)):
            try:
                return int(value)
            except (TypeError, ValueError):
                return default
        return default

    def get_float(self, key: str, default: float = 0.0) -> float:
        """Get parameter as float.

        Parameters
        ----------
        key
            Parameter name.
        default
            Default value if missing or invalid.

        Returns
        -------
        float
            Float value or default.

        Examples
        --------
        >>> params = ParamService({"rate": "3.14"})
        >>> params.get_float("rate")
        3.14
        """
        value = self._params.get(key)
        if value is None:
            return default
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value)
            except (TypeError, ValueError):
                return default
        return default

    def get_bool(self, key: str, *, default: bool = False) -> bool:
        """Get parameter as boolean.

        Use canonical truthy values from core/parsing.py.

        Parameters
        ----------
        key
            Parameter name.
        default
            Default value if missing.

        Returns
        -------
        bool
            Boolean value or default.

        Examples
        --------
        >>> params = ParamService({"enabled": "yes"})
        >>> params.get_bool("enabled")
        True
        >>> params.get_bool("disabled", default=True)
        True
        """
        value = self._params.get(key)
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return is_truthy_string(value)
        return bool(value)

    def get_path(self, key: str, default: Path | None = None) -> Path | None:
        """Get parameter as Path.

        Parameters
        ----------
        key
            Parameter name.
        default
            Default value if missing.

        Returns
        -------
        Path | None
            Path value or default.

        Examples
        --------
        >>> params = ParamService({"file": "/tmp/test.txt"})
        >>> params.get_path("file")
        PosixPath('/tmp/test.txt')
        """
        value = self._params.get(key)
        if value is None:
            return default
        if isinstance(value, Path):
            return value
        return Path(str(value))

    def get_enum(
        self,
        key: str,
        enum_type: type[E],
        default: E | None = None,
    ) -> E | None:
        """Get parameter as enum value.

        Match by value or name (case-insensitive).

        Parameters
        ----------
        key
            Parameter name.
        enum_type
            Enum class for coercion.
        default
            Default value if missing or invalid.

        Returns
        -------
        E | None
            Enum value or default.

        Raises
        ------
        ParamError
            If value cannot be coerced to enum.

        Examples
        --------
        >>> from codeintel.cli.rendering.types import OutputFormat
        >>> params = ParamService({"format": "json"})
        >>> params.get_enum("format", OutputFormat)
        <OutputFormat.JSON: 'json'>
        """
        value = self._params.get(key)
        if value is None:
            return default
        if isinstance(value, enum_type):
            return value
        if isinstance(value, str):
            for member in enum_type:
                if (
                    value.lower() == str(member.value).lower()
                    or value.lower() == member.name.lower()
                ):
                    return member
            return default
        msg = f"Parameter '{key}' must be a string or {enum_type.__name__}."
        raise ParamError(key, msg)

    def get_list(self, key: str, default: list[str] | None = None) -> list[str]:
        """Get parameter as list of strings.

        Parameters
        ----------
        key
            Parameter name.
        default
            Default value if missing.

        Returns
        -------
        list[str]
            List of strings or default.

        Examples
        --------
        >>> params = ParamService({"items": ["a", "b", "c"]})
        >>> params.get_list("items")
        ['a', 'b', 'c']
        """
        value = self._params.get(key)
        if value is None:
            return default or []
        if isinstance(value, list):
            return [str(item) for item in value]
        if isinstance(value, tuple):
            return [str(item) for item in value]
        if isinstance(value, str):
            return [value]
        return [str(value)]

    def get_tuple(
        self,
        key: str,
        default: tuple[str, ...] | None = None,
    ) -> tuple[str, ...]:
        """Get parameter as tuple of strings.

        Parameters
        ----------
        key
            Parameter name.
        default
            Default value if missing.

        Returns
        -------
        tuple[str, ...]
            Tuple of strings or default.
        """
        value = self._params.get(key)
        if value is None:
            return default or ()
        if isinstance(value, tuple):
            return tuple(str(item) for item in value)
        if isinstance(value, list):
            return tuple(str(item) for item in value)
        if isinstance(value, str):
            return (value,)
        return (str(value),)

    def require_str(self, key: str) -> str:
        """Get required string parameter.

        Parameters
        ----------
        key
            Parameter name.

        Returns
        -------
        str
            Non-empty string value.

        Raises
        ------
        ParamError
            If parameter is missing or empty.
        """
        value = self.get_str(key)
        if not value:
            raise ParamError(key, f"Required parameter '{key}' not provided.")
        return value

    def require_int(self, key: str) -> int:
        """Get required integer parameter.

        Parameters
        ----------
        key
            Parameter name.

        Returns
        -------
        int
            Integer value.

        Raises
        ------
        ParamError
            If parameter is missing or not an integer.
        """
        value = self._params.get(key)
        if value is None:
            raise ParamError(key, f"Required parameter '{key}' not provided.")
        if isinstance(value, int) and not isinstance(value, bool):
            return value
        if isinstance(value, bool):
            return 1 if value else 0
        if isinstance(value, (str, float)):
            try:
                return int(value)
            except ValueError as exc:
                raise ParamError(key, f"Parameter '{key}' must be an integer.") from exc
        raise ParamError(key, f"Parameter '{key}' must be an integer.")

    def require_path(self, key: str) -> Path:
        """Get required path parameter.

        Parameters
        ----------
        key
            Parameter name.

        Returns
        -------
        Path
            Path value.

        Raises
        ------
        ParamError
            If parameter is missing.
        """
        value = self.get_path(key)
        if value is None:
            raise ParamError(key, f"Required parameter '{key}' not provided.")
        return value

    def get_output_format(
        self,
        *,
        format_key: str = "output_format",
        json_key: str = "json",
        default: OutputFormat = OutputFormat.TEXT,
    ) -> OutputFormat:
        """Get output format with JSON flag override.

        Combine explicit format and --json flag with proper precedence.

        Parameters
        ----------
        format_key
            Key for explicit format parameter.
        json_key
            Key for JSON flag parameter.
        default
            Default format.

        Returns
        -------
        OutputFormat
            Resolved output format.
        """
        # JSON flag takes precedence
        if self.get_bool(json_key):
            return OutputFormat.JSON

        # Check explicit format
        fmt = self.get_enum(format_key, OutputFormat)
        if fmt is not None:
            return fmt

        return default

    @staticmethod
    def coerce_cli_value(value: str) -> str | int | float | bool:
        """Coerce a CLI string value to appropriate Python type.

        Delegate to canonical parse_cli_value from core/parsing.py.

        Parameters
        ----------
        value
            Raw string from CLI.

        Returns
        -------
        str | int | float | bool
            Coerced value.

        Examples
        --------
        >>> ParamService.coerce_cli_value("42")
        42
        >>> ParamService.coerce_cli_value("true")
        True
        >>> ParamService.coerce_cli_value("hello")
        'hello'
        """
        return parse_cli_value(value)

    @staticmethod
    def coerce_for_type(
        value: str,
        target_type: type[object] | None,
    ) -> str | int | float | bool:
        """Coerce string to specific target type.

        Parameters
        ----------
        value
            String value to coerce.
        target_type
            Target Python type.

        Returns
        -------
        str | int | float | bool
            Coerced value.
        """
        if target_type is None or target_type is str:
            return value
        if target_type is int:
            return int(value)
        if target_type is float:
            return float(value)
        if target_type is bool:
            return parse_bool(value)
        return value

    def merge(self, other: dict[str, object]) -> ParamService:
        """Create new service with merged parameters.

        Other values override existing values.

        Parameters
        ----------
        other
            Additional parameters to merge.

        Returns
        -------
        ParamService
            New service with merged parameters.
        """
        merged = {**self._params, **other}
        return ParamService(merged)


__all__ = [
    "ParamError",
    "ParamService",
]
