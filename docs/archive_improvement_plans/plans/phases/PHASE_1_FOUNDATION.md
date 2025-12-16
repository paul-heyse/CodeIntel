# Phase 1: Foundation Layer — Detailed Implementation Plan

> **Phase:** 1 of 6  
> **Duration:** 3-4 days  
> **Risk Level:** Low  
> **Dependencies:** Phase 0 complete  
> **Parallelizable:** Partially (Tasks P1-1 through P1-5 and P1-6 can run in parallel)  

---

## Table of Contents

1. [Objectives](#1-objectives)
2. [Prerequisites](#2-prerequisites)
3. [Deliverables](#3-deliverables)
4. [Detailed Tasks](#4-detailed-tasks)
5. [File Changes](#5-file-changes)
6. [Testing Requirements](#6-testing-requirements)
7. [Verification Checklist](#7-verification-checklist)
8. [Exit Criteria](#8-exit-criteria)
9. [Rollback Procedure](#9-rollback-procedure)

---

## 1. Objectives

Phase 1 creates the new foundational infrastructure:

1. **Create unified `HandlerContext`** — Single context type for all handlers
2. **Implement `bootstrap_cli()`** — Idempotent CLI initialization
3. **Establish test coverage** — Comprehensive tests for new components
4. **Create migration adapters** — Bridge old contexts to new (temporary)

This phase is **additive only** — no existing code is modified except imports.

---

## 2. Prerequisites

### 2.1 Phase 0 Artifacts

- [ ] Handler inventory complete (`docs/plans/phases/artifacts/handler_inventory.md`)
- [ ] Test baseline captured (`docs/plans/phases/artifacts/test_baseline_output.txt`)
- [ ] Coverage baseline exists (`docs/plans/phases/artifacts/coverage_baseline.json`)

### 2.2 Environment

- [ ] All existing tests passing
- [ ] Clean git working tree

---

## 3. Deliverables

### 3.1 `handlers/context.py` — New HandlerContext

**Purpose:** Single unified context for all CLI handler operations.

**Key Features:**
- Parameter accessor methods (`param_str`, `param_int`, `param_bool`, `param_path`, `param_enum`)
- Required parameter methods (`require_str`, `require_int`, `require_path`)
- Lazy resource properties (`runtime`, `gateway`, `graph_runtime`)
- Context manager protocol for automatic cleanup
- Adapter factory from legacy contexts

### 3.2 `execution/bootstrap.py` — CLI Bootstrap

**Purpose:** Idempotent CLI initialization.

**Key Features:**
- Thread-safe initialization
- Logging configuration
- Signal handler registration
- Configuration loading

### 3.3 Test Files

- `tests/cli/handlers/test_context.py` — HandlerContext unit tests
- `tests/cli/execution/test_bootstrap.py` — bootstrap_cli() unit tests

---

## 4. Detailed Tasks

### Task P1-1: Create `handlers/context.py` Skeleton

**Duration:** 2 hours

**File:** `src/codeintel/cli/handlers/context.py`

```python
"""Unified handler context for all CLI operations.

This module provides the single, canonical context type that all CLI handlers
receive. It consolidates functionality from:

- handlers/base.py (HandlerContext)
- handlers/protocol.py (EnhancedHandlerContext)
- execution/context.py (ExecutionContext)

All handlers should migrate to using this context type.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

from codeintel.cli.rendering.types import OutputFormat

if TYPE_CHECKING:
    from collections.abc import Iterator

    from codeintel.analytics.runtime import GraphRuntime
    from codeintel.cli.config.model import CliConfig
    from codeintel.cli.resolution.types import ResolvedRuntime
    from codeintel.storage.gateway import StorageGateway

LOG = logging.getLogger(__name__)

E = TypeVar("E", bound=Enum)


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
    >>> # In a handler:
    >>> def my_handler(ctx: HandlerContext) -> CliResult[MyData]:
    ...     name = ctx.param_str("name", "default")
    ...     limit = ctx.param_int("limit", 10)
    ...     if ctx.gateway:  # doctest: +SKIP
    ...         data = ctx.gateway.execute("SELECT ...")  # doctest: +SKIP
    ...     return CliResult.ok(MyData(name=name))  # doctest: +SKIP
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
    _params: dict[str, Any] = field(default_factory=dict, repr=False)
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
        """
        value = self._params.get(key)
        if value is None:
            return default
        if isinstance(value, int):
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
        ValueError
            If parameter is missing.
        """
        value = self._params.get(key)
        if value is None:
            msg = f"Required parameter '{key}' not provided"
            raise ValueError(msg)
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
        ValueError
            If parameter is missing or not a valid integer.
        """
        value = self._params.get(key)
        if value is None:
            msg = f"Required parameter '{key}' not provided"
            raise ValueError(msg)
        if isinstance(value, int):
            return value
        try:
            return int(str(value))
        except ValueError as e:
            msg = f"Parameter '{key}' must be an integer, got: {value!r}"
            raise ValueError(msg) from e

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
        ValueError
            If parameter is missing.
        """
        value = self._params.get(key)
        if value is None:
            msg = f"Required parameter '{key}' not provided"
            raise ValueError(msg)
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

        Raises
        ------
        RuntimeError
            If runtime cannot be resolved.
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

    def __enter__(self) -> HandlerContext:
        """Enter context manager.

        Returns
        -------
        HandlerContext
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
        from codeintel.cli.resolution.runtime import RuntimeResolver

        resolver = RuntimeResolver()
        return resolver.resolve(
            project_root=self.project_root,
            db_path=self.database_path,
            repo=self.param_str("repo"),
            commit=self.param_str("commit"),
        )

    def _open_gateway(self) -> StorageGateway:
        """Open storage gateway.

        Returns
        -------
        StorageGateway
            Open gateway.
        """
        from codeintel.storage.gateway import StorageConfig, open_gateway

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
        from codeintel.analytics.runtime import (
            GraphRuntime,
            GraphRuntimeOptions,
            build_graph_runtime,
        )

        options = GraphRuntimeOptions(snapshot=self.runtime.snapshot)
        return build_graph_runtime(
            gateway=self.gateway,
            options=options,
        )

    # --- Adapter Factory (Temporary - Remove in Phase 6) ---

    @classmethod
    def from_enhanced_context(
        cls,
        ctx: Any,  # EnhancedHandlerContext
        operation_id: str,
        params: dict[str, Any] | None = None,
    ) -> HandlerContext:
        """Create HandlerContext from legacy EnhancedHandlerContext.

        This is a temporary adapter for gradual migration. It will be
        removed in Phase 6 when all handlers have been migrated.

        Parameters
        ----------
        ctx
            Legacy EnhancedHandlerContext instance.
        operation_id
            Operation identifier.
        params
            Additional parameters (merged with ctx.params).

        Returns
        -------
        HandlerContext
            New context wrapping the legacy context's resources.

        Notes
        -----
        WARNING: This method is temporary scaffolding. Do not add new
        usages. It will be removed in Phase 6.
        """
        # Import here to avoid circular imports
        from codeintel.cli.handlers.protocol import EnhancedHandlerContext

        if not isinstance(ctx, EnhancedHandlerContext):
            msg = f"Expected EnhancedHandlerContext, got {type(ctx)}"
            raise TypeError(msg)

        # Merge params
        merged_params = dict(ctx.params)
        if params:
            merged_params.update(params)

        return cls(
            config=ctx.config,
            operation_id=operation_id,
            output_format=OutputFormat(ctx.output_format),
            verbosity=ctx.verbosity,
            project_root=ctx.runtime.root if ctx.runtime else None,
            database_path=ctx.runtime.db_path if ctx.runtime else None,
            _params=merged_params,
            _runtime=ctx.runtime,
            # Note: gateway and graph_runtime are not transferred
            # to avoid double-close issues
        )


@contextmanager
def handler_context_manager(
    config: CliConfig,
    operation_id: str,
    params: dict[str, Any] | None = None,
    *,
    output_format: OutputFormat = OutputFormat.TEXT,
    verbosity: int = 0,
    project_root: Path | None = None,
    database_path: Path | None = None,
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
    output_format
        Output format.
    verbosity
        Verbosity level.
    project_root
        Optional project root.
    database_path
        Optional database path.

    Yields
    ------
    HandlerContext
        Context for handler use.

    Examples
    --------
    >>> with handler_context_manager(config, "my.op") as ctx:  # doctest: +SKIP
    ...     result = my_handler(ctx)  # doctest: +SKIP
    """
    ctx = HandlerContext(
        config=config,
        operation_id=operation_id,
        output_format=output_format,
        verbosity=verbosity,
        project_root=project_root,
        database_path=database_path,
        _params=params or {},
    )
    try:
        yield ctx
    finally:
        ctx.close()


__all__ = [
    "HandlerContext",
    "handler_context_manager",
]
```

---

### Task P1-2: Implement Param Accessor Methods

**Duration:** 4 hours

This task is included in P1-1. Ensure all param accessors are implemented and handle edge cases:

**Test Cases to Cover:**

1. `param_str`:
   - Returns value when present
   - Returns default when missing
   - Converts non-string to string
   - Returns None when no default and missing

2. `param_int`:
   - Returns int when present
   - Returns default when missing
   - Converts string to int
   - Returns default on invalid conversion (with warning)

3. `param_bool`:
   - Returns bool when present
   - Returns default when missing
   - Handles string "true"/"false"/"1"/"0"
   - Handles truthy/falsy values

4. `param_path`:
   - Returns Path when present
   - Returns default when missing
   - Converts string to Path

5. `param_enum`:
   - Returns enum when present
   - Returns default when missing
   - Converts string to enum
   - Returns default on invalid conversion (with warning)

6. `require_str`, `require_int`, `require_path`:
   - Returns value when present
   - Raises ValueError when missing

---

### Task P1-3: Implement Lazy Resource Properties

**Duration:** 4 hours

Ensure lazy loading works correctly:

**Implementation Details:**

1. **runtime property**:
   - Resolves once on first access
   - Caches result
   - Uses RuntimeResolver internally
   - Propagates resolution errors

2. **gateway property**:
   - Opens once on first access
   - Requires runtime (triggers resolution)
   - Caches connection
   - Opens in read-only mode by default

3. **graph_runtime property**:
   - Builds once on first access
   - Requires gateway (triggers opening)
   - Caches instance

**Test Scenarios:**

- Access runtime multiple times → same object
- Access gateway multiple times → same object
- Access gateway without accessing runtime → runtime resolved automatically
- Close context → gateway closed, graph_runtime cleared

---

### Task P1-4: Implement Context Manager Protocol

**Duration:** 2 hours

**Implementation:**

```python
def __enter__(self) -> HandlerContext:
    return self

def __exit__(
    self,
    exc_type: type[BaseException] | None,
    exc_val: BaseException | None,
    exc_tb: object,
) -> None:
    self.close()
```

**Test Scenarios:**

```python
# Normal usage
with HandlerContext(...) as ctx:
    # use ctx
# resources automatically closed

# Exception during usage
with HandlerContext(...) as ctx:
    raise ValueError("test")
# resources still closed despite exception

# Nested contexts
with HandlerContext(...) as outer:
    with HandlerContext(...) as inner:
        pass
    # inner closed
# outer closed
```

---

### Task P1-5: Implement `from_enhanced_context` Adapter

**Duration:** 2 hours

This is a **temporary** method that will be removed in Phase 6.

**Purpose:** Allow gradual migration by creating new `HandlerContext` from existing `EnhancedHandlerContext` instances.

**Implementation Notes:**

1. Import `EnhancedHandlerContext` inside the method to avoid circular imports
2. Merge params from legacy context with any additional params
3. Transfer `config`, `runtime`, `verbosity`
4. Do NOT transfer `_gateway` or `_graph_runtime` to avoid double-close
5. Log a deprecation warning on each call (optional)

---

### Task P1-6: Create `execution/bootstrap.py`

**Duration:** 4 hours

**File:** `src/codeintel/cli/execution/bootstrap.py`

```python
"""CLI bootstrap - single entry point for CLI initialization.

This module provides bootstrap_cli(), the idempotent initialization function
that all CLI entry points should call. It consolidates:

- Logging configuration (from handlers/base.py)
- Signal handler registration
- Rich console setup

Call bootstrap_cli() once at CLI startup. Subsequent calls are no-ops.
"""

from __future__ import annotations

import logging
import signal
import sys
import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from codeintel.cli.config.model import CliConfig

LOG = logging.getLogger(__name__)

# Thread-safe initialization guard
_BOOTSTRAP_LOCK = threading.Lock()
_BOOTSTRAP_COMPLETE = False
_BOOTSTRAP_CONFIG: CliConfig | None = None

# Verbosity thresholds (same as handlers/base.py)
VERBOSITY_DEBUG = 2
VERBOSITY_INFO = 1


def bootstrap_cli(
    verbosity: int = 0,
    config: CliConfig | None = None,
) -> CliConfig:
    """Initialize CLI subsystems exactly once.

    This function is idempotent and thread-safe. It should be called at the
    start of every CLI command. Subsequent calls return the cached config.

    Initializes:

    - Logging configuration based on verbosity
    - Signal handlers for graceful shutdown (SIGINT, SIGTERM)
    - Rich console theming (via lazy import)

    Parameters
    ----------
    verbosity
        Logging verbosity level:
        - 0 = WARNING (or config default)
        - 1 = INFO
        - 2+ = DEBUG
    config
        Optional pre-loaded configuration. If None, loads from environment.

    Returns
    -------
    CliConfig
        The active CLI configuration.

    Examples
    --------
    >>> config = bootstrap_cli(verbosity=1)  # doctest: +SKIP
    >>> config.output_format  # doctest: +SKIP
    'text'
    """
    global _BOOTSTRAP_COMPLETE, _BOOTSTRAP_CONFIG  # noqa: PLW0603

    # Fast path for already initialized
    if _BOOTSTRAP_COMPLETE:
        if _BOOTSTRAP_CONFIG is not None:
            return _BOOTSTRAP_CONFIG
        # Shouldn't happen, but handle gracefully
        from codeintel.cli.config import load_config
        return load_config(validate=False)

    with _BOOTSTRAP_LOCK:
        # Double-check after acquiring lock
        if _BOOTSTRAP_COMPLETE and _BOOTSTRAP_CONFIG is not None:
            return _BOOTSTRAP_CONFIG

        # Load configuration if not provided
        if config is None:
            from codeintel.cli.config import load_config
            config = load_config(validate=False)

        # Configure logging
        _configure_logging(verbosity, config)

        # Register signal handlers
        _register_signal_handlers()

        # Mark as complete
        _BOOTSTRAP_CONFIG = config
        _BOOTSTRAP_COMPLETE = True

        LOG.debug("CLI bootstrap complete (verbosity=%d)", verbosity)

        return config


def _configure_logging(verbosity: int, config: CliConfig) -> None:
    """Configure logging based on verbosity.

    Parameters
    ----------
    verbosity
        Verbosity level from CLI.
    config
        CLI configuration.
    """
    level = _determine_log_level(verbosity, config)

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        force=True,  # Reconfigure if already configured
    )


def _determine_log_level(verbosity: int, config: CliConfig) -> int:
    """Determine log level from verbosity and config.

    Parameters
    ----------
    verbosity
        Verbosity level from CLI.
    config
        CLI configuration.

    Returns
    -------
    int
        Logging level constant.
    """
    if verbosity >= VERBOSITY_DEBUG:
        return logging.DEBUG
    if verbosity >= VERBOSITY_INFO:
        return logging.INFO
    # Use config default
    return getattr(logging, config.log_level, logging.WARNING)


def _register_signal_handlers() -> None:
    """Register signal handlers for graceful shutdown."""
    # Only register on main thread
    if threading.current_thread() is not threading.main_thread():
        return

    def _handle_signal(signum: int, frame: object) -> None:
        """Handle termination signal."""
        LOG.info("Received signal %d, initiating shutdown", signum)
        sys.exit(128 + signum)

    # Register handlers (ignore if not supported)
    try:
        signal.signal(signal.SIGINT, _handle_signal)
        signal.signal(signal.SIGTERM, _handle_signal)
    except (ValueError, OSError):
        # Signal registration may fail in some environments
        pass


def reset_bootstrap() -> None:
    """Reset bootstrap state (for testing only).

    WARNING: This function is for testing purposes only. Do not call
    in production code.
    """
    global _BOOTSTRAP_COMPLETE, _BOOTSTRAP_CONFIG  # noqa: PLW0603

    with _BOOTSTRAP_LOCK:
        _BOOTSTRAP_COMPLETE = False
        _BOOTSTRAP_CONFIG = None


__all__ = [
    "bootstrap_cli",
    "reset_bootstrap",
]
```

---

### Task P1-7: Write Unit Tests for `HandlerContext`

**Duration:** 4 hours

**File:** `tests/cli/handlers/test_context.py`

```python
"""Tests for HandlerContext."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

from codeintel.cli.handlers.context import HandlerContext, handler_context_manager
from codeintel.cli.rendering.types import OutputFormat

if TYPE_CHECKING:
    from codeintel.cli.config.model import CliConfig


class TestEnum(Enum):
    """Test enum for param_enum tests."""
    
    VALUE_A = "a"
    VALUE_B = "b"


@pytest.fixture
def mock_config() -> MagicMock:
    """Create mock CLI config."""
    config = MagicMock(spec=["output_format", "log_level", "color", "progress"])
    config.output_format = "text"
    config.log_level = "WARNING"
    return config


@pytest.fixture
def basic_context(mock_config: MagicMock) -> HandlerContext:
    """Create basic HandlerContext for testing."""
    return HandlerContext(
        config=mock_config,
        operation_id="test.operation",
        output_format=OutputFormat.TEXT,
        verbosity=0,
        _params={"name": "test", "count": 10, "enabled": True},
    )


class TestParamStr:
    """Tests for param_str method."""

    def test_returns_value_when_present(self, basic_context: HandlerContext) -> None:
        """Return string value when parameter exists."""
        assert basic_context.param_str("name") == "test"

    def test_returns_default_when_missing(self, basic_context: HandlerContext) -> None:
        """Return default when parameter missing."""
        assert basic_context.param_str("missing", "default") == "default"

    def test_returns_none_when_missing_no_default(
        self, basic_context: HandlerContext
    ) -> None:
        """Return None when parameter missing and no default."""
        assert basic_context.param_str("missing") is None

    def test_converts_int_to_string(self, basic_context: HandlerContext) -> None:
        """Convert non-string values to string."""
        assert basic_context.param_str("count") == "10"


class TestParamInt:
    """Tests for param_int method."""

    def test_returns_value_when_present(self, basic_context: HandlerContext) -> None:
        """Return integer value when parameter exists."""
        assert basic_context.param_int("count") == 10

    def test_returns_default_when_missing(self, basic_context: HandlerContext) -> None:
        """Return default when parameter missing."""
        assert basic_context.param_int("missing", 42) == 42

    def test_converts_string_to_int(self, mock_config: MagicMock) -> None:
        """Convert string values to integer."""
        ctx = HandlerContext(
            config=mock_config,
            operation_id="test",
            _params={"count": "20"},
        )
        assert ctx.param_int("count") == 20

    def test_returns_default_on_invalid(self, mock_config: MagicMock) -> None:
        """Return default when conversion fails."""
        ctx = HandlerContext(
            config=mock_config,
            operation_id="test",
            _params={"count": "not-a-number"},
        )
        assert ctx.param_int("count", 5) == 5


class TestParamBool:
    """Tests for param_bool method."""

    def test_returns_value_when_present(self, basic_context: HandlerContext) -> None:
        """Return boolean value when parameter exists."""
        assert basic_context.param_bool("enabled") is True

    def test_returns_default_when_missing(self, basic_context: HandlerContext) -> None:
        """Return default when parameter missing."""
        assert basic_context.param_bool("missing", default=True) is True

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("true", True),
            ("True", True),
            ("TRUE", True),
            ("1", True),
            ("yes", True),
            ("on", True),
            ("false", False),
            ("False", False),
            ("0", False),
            ("no", False),
            ("off", False),
        ],
    )
    def test_handles_string_values(
        self, mock_config: MagicMock, value: str, expected: bool
    ) -> None:
        """Handle various string representations of boolean."""
        ctx = HandlerContext(
            config=mock_config,
            operation_id="test",
            _params={"flag": value},
        )
        assert ctx.param_bool("flag") is expected


class TestParamPath:
    """Tests for param_path method."""

    def test_returns_path_when_present(self, mock_config: MagicMock) -> None:
        """Return Path when parameter exists."""
        ctx = HandlerContext(
            config=mock_config,
            operation_id="test",
            _params={"path": Path("/some/path")},
        )
        assert ctx.param_path("path") == Path("/some/path")

    def test_converts_string_to_path(self, mock_config: MagicMock) -> None:
        """Convert string values to Path."""
        ctx = HandlerContext(
            config=mock_config,
            operation_id="test",
            _params={"path": "/some/path"},
        )
        assert ctx.param_path("path") == Path("/some/path")

    def test_returns_default_when_missing(self, mock_config: MagicMock) -> None:
        """Return default when parameter missing."""
        ctx = HandlerContext(
            config=mock_config,
            operation_id="test",
            _params={},
        )
        assert ctx.param_path("path", Path("/default")) == Path("/default")


class TestParamEnum:
    """Tests for param_enum method."""

    def test_returns_enum_when_present(self, mock_config: MagicMock) -> None:
        """Return enum when parameter exists."""
        ctx = HandlerContext(
            config=mock_config,
            operation_id="test",
            _params={"choice": TestEnum.VALUE_A},
        )
        assert ctx.param_enum("choice", TestEnum) == TestEnum.VALUE_A

    def test_converts_string_to_enum(self, mock_config: MagicMock) -> None:
        """Convert string values to enum."""
        ctx = HandlerContext(
            config=mock_config,
            operation_id="test",
            _params={"choice": "a"},
        )
        assert ctx.param_enum("choice", TestEnum) == TestEnum.VALUE_A

    def test_returns_default_on_invalid(self, mock_config: MagicMock) -> None:
        """Return default when conversion fails."""
        ctx = HandlerContext(
            config=mock_config,
            operation_id="test",
            _params={"choice": "invalid"},
        )
        assert ctx.param_enum("choice", TestEnum, TestEnum.VALUE_B) == TestEnum.VALUE_B


class TestRequireStr:
    """Tests for require_str method."""

    def test_returns_value_when_present(self, basic_context: HandlerContext) -> None:
        """Return string value when parameter exists."""
        assert basic_context.require_str("name") == "test"

    def test_raises_when_missing(self, basic_context: HandlerContext) -> None:
        """Raise ValueError when parameter missing."""
        with pytest.raises(ValueError, match="Required parameter 'missing'"):
            basic_context.require_str("missing")


class TestRequireInt:
    """Tests for require_int method."""

    def test_returns_value_when_present(self, basic_context: HandlerContext) -> None:
        """Return integer value when parameter exists."""
        assert basic_context.require_int("count") == 10

    def test_raises_when_missing(self, basic_context: HandlerContext) -> None:
        """Raise ValueError when parameter missing."""
        with pytest.raises(ValueError, match="Required parameter 'missing'"):
            basic_context.require_int("missing")


class TestContextManager:
    """Tests for context manager protocol."""

    def test_enter_returns_self(self, basic_context: HandlerContext) -> None:
        """__enter__ returns the context."""
        with basic_context as ctx:
            assert ctx is basic_context

    def test_exit_calls_close(self, mock_config: MagicMock) -> None:
        """__exit__ closes resources."""
        ctx = HandlerContext(
            config=mock_config,
            operation_id="test",
        )
        with ctx:
            pass
        assert ctx._closed is True

    def test_close_on_exception(self, mock_config: MagicMock) -> None:
        """Resources closed even on exception."""
        ctx = HandlerContext(
            config=mock_config,
            operation_id="test",
        )
        with pytest.raises(ValueError):
            with ctx:
                raise ValueError("test error")
        assert ctx._closed is True


class TestHandlerContextManager:
    """Tests for handler_context_manager function."""

    def test_creates_context(self, mock_config: MagicMock) -> None:
        """Create context with correct parameters."""
        with handler_context_manager(
            mock_config,
            "test.op",
            params={"key": "value"},
            verbosity=1,
        ) as ctx:
            assert ctx.operation_id == "test.op"
            assert ctx.verbosity == 1
            assert ctx.param_str("key") == "value"

    def test_closes_on_exit(self, mock_config: MagicMock) -> None:
        """Close context on exit."""
        with handler_context_manager(mock_config, "test.op") as ctx:
            pass
        assert ctx._closed is True
```

---

### Task P1-8: Write Unit Tests for `bootstrap_cli`

**Duration:** 2 hours

**File:** `tests/cli/execution/test_bootstrap.py`

```python
"""Tests for CLI bootstrap."""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import pytest

from codeintel.cli.execution.bootstrap import bootstrap_cli, reset_bootstrap


@pytest.fixture(autouse=True)
def _reset_bootstrap_state() -> None:
    """Reset bootstrap state before each test."""
    reset_bootstrap()


@pytest.fixture
def mock_config() -> MagicMock:
    """Create mock CLI config."""
    config = MagicMock()
    config.log_level = "WARNING"
    return config


class TestBootstrapCli:
    """Tests for bootstrap_cli function."""

    def test_returns_config(self, mock_config: MagicMock) -> None:
        """Return the provided config."""
        result = bootstrap_cli(config=mock_config)
        assert result is mock_config

    def test_loads_config_when_not_provided(self) -> None:
        """Load config from environment when not provided."""
        with patch("codeintel.cli.execution.bootstrap.load_config") as mock_load:
            mock_load.return_value = MagicMock(log_level="INFO")
            result = bootstrap_cli()
            mock_load.assert_called_once_with(validate=False)
            assert result is mock_load.return_value

    def test_idempotent_second_call_returns_cached(
        self, mock_config: MagicMock
    ) -> None:
        """Second call returns cached config without re-initialization."""
        first = bootstrap_cli(config=mock_config)
        
        # Create different config for second call
        other_config = MagicMock()
        second = bootstrap_cli(config=other_config)
        
        # Should return first config, not second
        assert second is first
        assert second is mock_config

    def test_configures_logging_at_debug(self, mock_config: MagicMock) -> None:
        """Configure DEBUG logging when verbosity >= 2."""
        bootstrap_cli(verbosity=2, config=mock_config)
        assert logging.getLogger().level == logging.DEBUG

    def test_configures_logging_at_info(self, mock_config: MagicMock) -> None:
        """Configure INFO logging when verbosity == 1."""
        bootstrap_cli(verbosity=1, config=mock_config)
        assert logging.getLogger().level == logging.INFO

    def test_configures_logging_at_warning(self, mock_config: MagicMock) -> None:
        """Configure WARNING logging when verbosity == 0."""
        mock_config.log_level = "WARNING"
        bootstrap_cli(verbosity=0, config=mock_config)
        assert logging.getLogger().level == logging.WARNING


class TestResetBootstrap:
    """Tests for reset_bootstrap function."""

    def test_allows_reinitialize(self, mock_config: MagicMock) -> None:
        """Reset allows re-initialization."""
        first = bootstrap_cli(config=mock_config)
        reset_bootstrap()
        
        other_config = MagicMock(log_level="DEBUG")
        second = bootstrap_cli(config=other_config)
        
        # After reset, should use new config
        assert second is other_config
        assert second is not first
```

---

### Task P1-9: Integration Test - New Context in Isolation

**Duration:** 2 hours

Create a simple integration test that verifies the new context works end-to-end without touching old code paths.

**File:** `tests/cli/handlers/test_context_integration.py`

```python
"""Integration tests for new HandlerContext."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from codeintel.cli.handlers.context import HandlerContext
from codeintel.cli.rendering.types import OutputFormat


class TestHandlerContextIntegration:
    """Integration tests for HandlerContext."""

    def test_full_param_workflow(self) -> None:
        """Test complete parameter workflow."""
        config = MagicMock()
        config.log_level = "WARNING"

        ctx = HandlerContext(
            config=config,
            operation_id="test.integration",
            output_format=OutputFormat.JSON,
            verbosity=1,
            project_root=Path("/test/project"),
            _params={
                "name": "test-name",
                "count": 42,
                "enabled": True,
                "path": "/test/path",
            },
        )

        # Test all param accessors
        assert ctx.param_str("name") == "test-name"
        assert ctx.param_int("count") == 42
        assert ctx.param_bool("enabled") is True
        assert ctx.param_path("path") == Path("/test/path")

        # Test require variants
        assert ctx.require_str("name") == "test-name"
        assert ctx.require_int("count") == 42

        # Test context properties
        assert ctx.operation_id == "test.integration"
        assert ctx.output_format == OutputFormat.JSON
        assert ctx.verbosity == 1
        assert ctx.project_root == Path("/test/project")

    def test_context_manager_closes_resources(self) -> None:
        """Test context manager properly closes resources."""
        config = MagicMock()
        config.log_level = "WARNING"

        with HandlerContext(
            config=config,
            operation_id="test.cleanup",
            _params={},
        ) as ctx:
            assert ctx._closed is False

        assert ctx._closed is True

    def test_logger_property(self) -> None:
        """Test logger property returns correct logger."""
        config = MagicMock()
        config.log_level = "WARNING"

        ctx = HandlerContext(
            config=config,
            operation_id="my.operation",
            _params={},
        )

        logger = ctx.logger
        assert logger.name == "codeintel.cli.handlers.my.operation"
```

---

### Task P1-10: Code Review and Refinement

**Duration:** 2 hours

1. Run all quality checks:
   ```bash
   uv run ruff check --fix src/codeintel/cli/handlers/context.py
   uv run ruff check --fix src/codeintel/cli/execution/bootstrap.py
   uv run pyright --warnings --pythonversion=3.13
   uv run pyrefly check
   ```

2. Ensure docstrings are complete and follow NumPy style

3. Verify test coverage:
   ```bash
   uv run pytest tests/cli/handlers/test_context.py tests/cli/execution/test_bootstrap.py \
     --cov=src/codeintel/cli/handlers/context \
     --cov=src/codeintel/cli/execution/bootstrap \
     --cov-report=term-missing
   ```

4. Target: >90% coverage on new modules

---

## 5. File Changes

### 5.1 New Files Created

| File | Type | Lines (approx) |
|------|------|----------------|
| `src/codeintel/cli/handlers/context.py` | Python | 350-400 |
| `src/codeintel/cli/execution/bootstrap.py` | Python | 150-180 |
| `tests/cli/handlers/test_context.py` | Python | 250-300 |
| `tests/cli/execution/test_bootstrap.py` | Python | 80-100 |
| `tests/cli/handlers/test_context_integration.py` | Python | 60-80 |

### 5.2 Files Modified

None — Phase 1 is purely additive.

### 5.3 Files Deleted

None.

---

## 6. Testing Requirements

### 6.1 Unit Test Coverage

| Module | Target Coverage |
|--------|-----------------|
| `handlers/context.py` | >90% |
| `execution/bootstrap.py` | >90% |

### 6.2 Test Categories

- Parameter accessor tests (all types)
- Required parameter tests (error cases)
- Context manager tests (cleanup, exceptions)
- Bootstrap idempotency tests
- Thread-safety tests (optional)

### 6.3 Regression Testing

All existing CLI tests must still pass:

```bash
uv run pytest tests/cli/ -x -q
```

---

## 7. Verification Checklist

### 7.1 Code Quality

- [ ] `handlers/context.py` passes ruff check
- [ ] `handlers/context.py` passes pyright strict
- [ ] `handlers/context.py` passes pyrefly check
- [ ] `execution/bootstrap.py` passes ruff check
- [ ] `execution/bootstrap.py` passes pyright strict
- [ ] `execution/bootstrap.py` passes pyrefly check
- [ ] All docstrings complete (NumPy style)

### 7.2 Test Coverage

- [ ] `test_context.py` covers all param accessors
- [ ] `test_context.py` covers require methods (success and error)
- [ ] `test_context.py` covers context manager
- [ ] `test_bootstrap.py` covers idempotency
- [ ] `test_bootstrap.py` covers verbosity levels
- [ ] Integration test passes

### 7.3 No Regressions

- [ ] All existing CLI tests pass
- [ ] No existing code modified (except additions)

---

## 8. Exit Criteria

Phase 1 is complete when:

| Criterion | Status |
|-----------|--------|
| `handlers/context.py` implemented | ⬜ |
| `execution/bootstrap.py` implemented | ⬜ |
| Unit tests for HandlerContext (>90% coverage) | ⬜ |
| Unit tests for bootstrap_cli (>90% coverage) | ⬜ |
| Integration test passes | ⬜ |
| All quality checks pass (ruff, pyright, pyrefly) | ⬜ |
| All existing CLI tests pass | ⬜ |
| Code review complete | ⬜ |

---

## 9. Rollback Procedure

Phase 1 is low-risk because it's purely additive.

**To rollback:**

1. Delete new files:
   ```bash
   rm src/codeintel/cli/handlers/context.py
   rm src/codeintel/cli/execution/bootstrap.py
   rm tests/cli/handlers/test_context.py
   rm tests/cli/handlers/test_context_integration.py
   rm tests/cli/execution/test_bootstrap.py
   ```

2. Verify no other changes:
   ```bash
   git status
   git diff
   ```

3. Commit rollback if needed

---

**Previous Phase:** [Phase 0: Preparation](./PHASE_0_PREPARATION.md)  
**Next Phase:** [Phase 2: Rendering Consolidation](./PHASE_2_RENDERING.md)
