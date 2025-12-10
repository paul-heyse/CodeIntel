"""Unified handler context for all CLI operations.

This module provides the single, canonical context type that all CLI handlers
receive. It provides:

- Lazy resource access (gateway, runtime, graph_runtime)
- Typed parameter extraction
- Automatic resource cleanup
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Self, TypeVar

from codeintel.analytics.runtime import (
    GraphRuntime,
    GraphRuntimeOptions,
    build_graph_runtime,
)
from codeintel.cli.handlers._lazy_resources import lazy_resolve_runtime
from codeintel.cli.rendering.types import OutputFormat
from codeintel.storage.gateway import StorageConfig, StorageGateway, open_gateway

if TYPE_CHECKING:
    from collections.abc import Iterator

    from codeintel.cli.config.model import CliConfig
    from codeintel.cli.resolution.types import ResolvedRuntime

LOG = logging.getLogger(__name__)

E = TypeVar("E", bound=Enum)


class ParameterError(ValueError):
    """Error raised when a required parameter is missing or invalid.

    Parameters
    ----------
    key
        The parameter key that caused the error.
    message
        Human-readable error message.

    Examples
    --------
    >>> raise ParameterError("name", "Required parameter 'name' not provided")
    Traceback (most recent call last):
        ...
    codeintel.cli.handlers.context.ParameterError: Required parameter 'name' not provided
    """

    def __init__(self, key: str, message: str) -> None:
        """Initialize the error."""
        super().__init__(message)
        self.key = key


@dataclass(frozen=True)
class HandlerContextOptions:
    """Options for creating a HandlerContext.

    Bundle optional parameters to reduce argument count in factory functions.

    Parameters
    ----------
    output_format
        Requested output format.
    verbosity
        Verbosity level (0=WARNING, 1=INFO, 2+=DEBUG).
    project_root
        Optional project root directory.
    database_path
        Optional database file path.
    """

    output_format: OutputFormat = OutputFormat.TEXT
    verbosity: int = 0
    project_root: Path | None = None
    database_path: Path | None = None


@dataclass
class HandlerContext:
    """Unified context for all CLI handler operations.

    This is the single context type that all handlers receive. It provides:

    - Configuration access
    - Operation metadata
    - Parameter accessors with type conversion
    - Lazy resource loading (runtime, gateway, graph_runtime)
    - Automatic resource cleanup via context manager

    Parameters
    ----------
    config
        CLI configuration.
    operation_id
        Unique identifier for this operation.
    output_format
        Requested output format.
    verbosity
        Verbosity level (0=WARNING, 1=INFO, 2+=DEBUG).
    project_root
        Optional project root directory.
    index_path
        Optional index file path.
    database_path
        Optional database file path.

    Examples
    --------
    >>> from unittest.mock import MagicMock
    >>> config = MagicMock()
    >>> config.log_level = "WARNING"
    >>> ctx = HandlerContext(
    ...     config=config,
    ...     operation_id="test.op",
    ...     _params={"name": "example", "count": 5},
    ... )
    >>> ctx.param_str("name")
    'example'
    >>> ctx.param_int("count")
    5
    >>> ctx.param_str("missing", "default")
    'default'
    """

    # Core configuration
    config: CliConfig
    operation_id: str
    output_format: OutputFormat = OutputFormat.TEXT
    verbosity: int = 0

    # Runtime resolution parameters
    project_root: Path | None = None
    index_path: Path | None = None
    database_path: Path | None = None

    # Internal state
    _params: dict[str, object] = field(default_factory=dict, repr=False)
    _runtime: ResolvedRuntime | None = field(default=None, repr=False)
    _gateway: StorageGateway | None = field(default=None, repr=False)
    _graph_runtime: GraphRuntime | None = field(default=None, repr=False)
    _closed: bool = field(default=False, repr=False)

    # --- Parameter Accessors ---

    def param_str(self, key: str, default: str | None = None) -> str | None:
        """Get string parameter with optional default.

        Parameters
        ----------
        key
            Parameter name.
        default
            Default value if parameter not present.

        Returns
        -------
        str | None
            Parameter value or default.

        Examples
        --------
        >>> from unittest.mock import MagicMock
        >>> ctx = HandlerContext(config=MagicMock(), operation_id="test", _params={"name": "value"})
        >>> ctx.param_str("name")
        'value'
        >>> ctx.param_str("missing", "default")
        'default'
        >>> ctx.param_str("missing") is None
        True
        """
        value = self._params.get(key)
        if value is None:
            return default
        return str(value)

    def param_int(self, key: str, default: int = 0) -> int:
        """Get integer parameter with default.

        Parameters
        ----------
        key
            Parameter name.
        default
            Default value if parameter not present or invalid.

        Returns
        -------
        int
            Parameter value or default.

        Examples
        --------
        >>> from unittest.mock import MagicMock
        >>> ctx = HandlerContext(
        ...     config=MagicMock(), operation_id="test", _params={"count": 42, "text": "5"}
        ... )
        >>> ctx.param_int("count")
        42
        >>> ctx.param_int("text")
        5
        >>> ctx.param_int("missing", 10)
        10
        """
        value = self._params.get(key)
        if value is None:
            return default
        if isinstance(value, int) and not isinstance(value, bool):
            return value
        try:
            return int(str(value))
        except ValueError:
            LOG.warning("Invalid int value for %s: %r, using default %d", key, value, default)
            return default

    def param_bool(self, key: str, *, default: bool = False) -> bool:
        """Get boolean parameter with default.

        Parameters
        ----------
        key
            Parameter name.
        default
            Default value if parameter not present.

        Returns
        -------
        bool
            Parameter value or default.

        Examples
        --------
        >>> from unittest.mock import MagicMock
        >>> ctx = HandlerContext(
        ...     config=MagicMock(), operation_id="test", _params={"flag": True, "text": "yes"}
        ... )
        >>> ctx.param_bool("flag")
        True
        >>> ctx.param_bool("text")
        True
        >>> ctx.param_bool("missing", default=True)
        True
        """
        value = self._params.get(key)
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        # Handle string representations
        if isinstance(value, str):
            return value.lower() in {"true", "1", "yes", "on"}
        return bool(value)

    def param_path(self, key: str, default: Path | None = None) -> Path | None:
        """Get Path parameter with optional default.

        Parameters
        ----------
        key
            Parameter name.
        default
            Default value if parameter not present.

        Returns
        -------
        Path | None
            Parameter value or default.

        Examples
        --------
        >>> from unittest.mock import MagicMock
        >>> from pathlib import Path
        >>> ctx = HandlerContext(
        ...     config=MagicMock(), operation_id="test", _params={"path": "/some/path"}
        ... )
        >>> ctx.param_path("path")
        PosixPath('/some/path')
        >>> ctx.param_path("missing", Path("/default"))
        PosixPath('/default')
        """
        value = self._params.get(key)
        if value is None:
            return default
        if isinstance(value, Path):
            return value
        return Path(str(value))

    def param_enum(self, key: str, enum_type: type[E], default: E | None = None) -> E | None:
        """Get enum parameter with optional default.

        Parameters
        ----------
        key
            Parameter name.
        enum_type
            Enum class to convert to.
        default
            Default value if parameter not present or invalid.

        Returns
        -------
        E | None
            Parameter value or default.

        Examples
        --------
        >>> from unittest.mock import MagicMock
        >>> from enum import Enum
        >>> class Color(Enum):
        ...     RED = "red"
        ...     BLUE = "blue"
        >>> ctx = HandlerContext(config=MagicMock(), operation_id="test", _params={"color": "red"})
        >>> ctx.param_enum("color", Color)
        <Color.RED: 'red'>
        >>> ctx.param_enum("missing", Color, Color.BLUE)
        <Color.BLUE: 'blue'>
        """
        value = self._params.get(key)
        if value is None:
            return default
        if isinstance(value, enum_type):
            return value
        try:
            return enum_type(str(value))
        except ValueError:
            LOG.warning("Invalid enum value for %s: %r, using default", key, value)
            return default

    def param_list(self, key: str, default: list[str] | None = None) -> list[str]:
        """Get list parameter with optional default.

        Parameters
        ----------
        key
            Parameter name.
        default
            Default value if parameter not present.

        Returns
        -------
        list[str]
            Parameter value or default (empty list if None).

        Examples
        --------
        >>> from unittest.mock import MagicMock
        >>> ctx = HandlerContext(
        ...     config=MagicMock(), operation_id="test", _params={"items": ["a", "b"]}
        ... )
        >>> ctx.param_list("items")
        ['a', 'b']
        >>> ctx.param_list("missing")
        []
        """
        value = self._params.get(key)
        if value is None:
            return default if default is not None else []
        if isinstance(value, list):
            return [str(v) for v in value]
        if isinstance(value, tuple):
            return [str(v) for v in value]
        # Single value becomes single-item list
        return [str(value)]

    def param_tuple(self, key: str, default: tuple[str, ...] | None = None) -> tuple[str, ...]:
        """Get tuple parameter with optional default.

        Parameters
        ----------
        key
            Parameter name.
        default
            Default value if parameter not present.

        Returns
        -------
        tuple[str, ...]
            Parameter value or default (empty tuple if None).

        Examples
        --------
        >>> from unittest.mock import MagicMock
        >>> ctx = HandlerContext(
        ...     config=MagicMock(), operation_id="test", _params={"items": ("a", "b")}
        ... )
        >>> ctx.param_tuple("items")
        ('a', 'b')
        >>> ctx.param_tuple("missing")
        ()
        """
        value = self._params.get(key)
        if value is None:
            return default if default is not None else ()
        if isinstance(value, tuple):
            return tuple(str(v) for v in value)
        if isinstance(value, list):
            return tuple(str(v) for v in value)
        # Single value becomes single-item tuple
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
            Parameter value.

        Raises
        ------
        ParameterError
            If parameter is missing.

        Examples
        --------
        >>> from unittest.mock import MagicMock
        >>> ctx = HandlerContext(config=MagicMock(), operation_id="test", _params={"name": "value"})
        >>> ctx.require_str("name")
        'value'
        """
        value = self._params.get(key)
        if value is None:
            msg = f"Required parameter '{key}' not provided"
            raise ParameterError(key, msg)
        return str(value)

    def require_int(self, key: str) -> int:
        """Get required integer parameter.

        Parameters
        ----------
        key
            Parameter name.

        Returns
        -------
        int
            Parameter value.

        Raises
        ------
        ParameterError
            If parameter is missing or not a valid integer.

        Examples
        --------
        >>> from unittest.mock import MagicMock
        >>> ctx = HandlerContext(config=MagicMock(), operation_id="test", _params={"count": 42})
        >>> ctx.require_int("count")
        42
        """
        value = self._params.get(key)
        if value is None:
            msg = f"Required parameter '{key}' not provided"
            raise ParameterError(key, msg)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
        try:
            return int(str(value))
        except ValueError as e:
            msg = f"Parameter '{key}' must be an integer, got: {value!r}"
            raise ParameterError(key, msg) from e

    def require_path(self, key: str) -> Path:
        """Get required Path parameter.

        Parameters
        ----------
        key
            Parameter name.

        Returns
        -------
        Path
            Parameter value.

        Raises
        ------
        ParameterError
            If parameter is missing.

        Examples
        --------
        >>> from unittest.mock import MagicMock
        >>> ctx = HandlerContext(
        ...     config=MagicMock(), operation_id="test", _params={"path": "/some/path"}
        ... )
        >>> ctx.require_path("path")
        PosixPath('/some/path')
        """
        value = self._params.get(key)
        if value is None:
            msg = f"Required parameter '{key}' not provided"
            raise ParameterError(key, msg)
        if isinstance(value, Path):
            return value
        return Path(str(value))

    # --- Lazy Resource Properties ---

    @property
    def runtime(self) -> ResolvedRuntime:
        """Get resolved runtime (lazy).

        Returns
        -------
        ResolvedRuntime
            Fully resolved runtime information.

        Notes
        -----
        Propagates ResolutionError from RuntimeResolver if runtime cannot
        be resolved (e.g., no project file and missing required params).
        """
        if self._runtime is None:
            self._runtime = self._resolve_runtime()
        return self._runtime

    @property
    def gateway(self) -> StorageGateway:
        """Get storage gateway (lazy).

        Gateway is opened on first access. The context manages lifecycle.

        Returns
        -------
        StorageGateway
            Open storage gateway.
        """
        if self._gateway is None:
            self._gateway = self._open_gateway()
        return self._gateway

    @property
    def graph_runtime(self) -> GraphRuntime:
        """Get graph runtime (lazy).

        Returns
        -------
        GraphRuntime
            Graph runtime for graph operations.
        """
        if self._graph_runtime is None:
            self._graph_runtime = self._build_graph_runtime()
        return self._graph_runtime

    # --- Convenience Properties ---

    @property
    def logger(self) -> logging.Logger:
        """Get logger for this operation.

        Returns
        -------
        logging.Logger
            Logger named for this operation.

        Examples
        --------
        >>> from unittest.mock import MagicMock
        >>> ctx = HandlerContext(config=MagicMock(), operation_id="my.operation")
        >>> ctx.logger.name
        'codeintel.cli.handlers.my.operation'
        """
        return logging.getLogger(f"codeintel.cli.handlers.{self.operation_id}")

    @property
    def db_path(self) -> Path | None:
        """Get database path.

        Returns
        -------
        Path | None
            Database path if available.
        """
        if self._runtime is not None:
            return self._runtime.db_path
        return self.database_path

    @property
    def color_enabled(self) -> bool:
        """Check if color output is enabled.

        Returns
        -------
        bool
            True if color is enabled.
        """
        return self.config.color

    @property
    def params(self) -> dict[str, object]:
        """Get operation parameters (read-only view).

        Returns
        -------
        dict[str, object]
            Parameters dictionary.

        Notes
        -----
        Prefer using typed accessor methods (param_str, param_int, etc.)
        for individual parameter access. This property is mainly for
        handlers that need to acknowledge params exist without using them.
        """
        return self._params

    # --- Resource Management ---

    def close(self) -> None:
        """Close managed resources.

        Safe to call multiple times. Called automatically when using
        as a context manager.
        """
        if self._closed:
            return

        if self._gateway is not None:
            try:
                self._gateway.close()
            except Exception:
                LOG.exception("Error closing gateway")
            self._gateway = None

        self._graph_runtime = None
        self._closed = True

    def __enter__(self) -> Self:
        """Enter context manager.

        Returns
        -------
        Self
            Self for use in with block.
        """
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        """Exit context manager, closing resources."""
        self.close()

    # --- Private Methods ---

    def _resolve_runtime(self) -> ResolvedRuntime:
        """Resolve runtime from context parameters.

        Returns
        -------
        ResolvedRuntime
            Resolved runtime.
        """
        return lazy_resolve_runtime(
            operation_id=self.operation_id,
            params=self._params,
            project_root=self.project_root,
            database_path=self.database_path,
        )

    def _open_gateway(self) -> StorageGateway:
        """Open storage gateway.

        Returns
        -------
        StorageGateway
            Open gateway.
        """
        runtime = self.runtime
        storage_config = StorageConfig(db_path=runtime.db_path, read_only=True)
        return open_gateway(storage_config)

    def _build_graph_runtime(self) -> GraphRuntime:
        """Build graph runtime.

        Returns
        -------
        GraphRuntime
            Configured graph runtime.
        """
        options = GraphRuntimeOptions(snapshot=self.runtime.snapshot)
        return build_graph_runtime(
            gateway=self.gateway,
            options=options,
        )


@contextmanager
def handler_context_manager(
    config: CliConfig,
    operation_id: str,
    params: dict[str, object] | None = None,
    options: HandlerContextOptions | None = None,
) -> Iterator[HandlerContext]:
    """Create handler context with automatic resource cleanup.

    Parameters
    ----------
    config
        CLI configuration.
    operation_id
        Operation identifier.
    params
        Operation parameters.
    options
        Optional context options (output_format, verbosity, etc.).

    Yields
    ------
    HandlerContext
        Context for handler use.

    Examples
    --------
    >>> from unittest.mock import MagicMock
    >>> config = MagicMock()
    >>> config.log_level = "WARNING"
    >>> with handler_context_manager(config, "my.op", {"key": "value"}) as ctx:
    ...     ctx.operation_id == "my.op"
    ...     ctx.param_str("key") == "value"
    True
    True
    """
    opts = options or HandlerContextOptions()
    ctx = HandlerContext(
        config=config,
        operation_id=operation_id,
        output_format=opts.output_format,
        verbosity=opts.verbosity,
        project_root=opts.project_root,
        database_path=opts.database_path,
        _params=params or {},
    )
    try:
        yield ctx
    finally:
        ctx.close()


__all__ = [
    "HandlerContext",
    "HandlerContextOptions",
    "ParameterError",
    "handler_context_manager",
]
