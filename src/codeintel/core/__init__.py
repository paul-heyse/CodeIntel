"""Core infrastructure shared between graphs and analytics subsystems.

This package contains unified protocols, types, and utilities that are
used by both the graphs and analytics subsystems, eliminating duplication
and ensuring consistency.

Subpackages
-----------
- config: Configuration accessor protocol and registry
- execution: Runtime execution infrastructure (telemetry, retry, timing)
- plugins: Unified plugin protocol, result types, and registry
- recipes: Unified recipe DSL and executor
- resources: Unified resource provider protocol and registry
- types: Common type definitions (status types)

Modules
-------
- singleton: Thread-safe singleton holder pattern
"""

from __future__ import annotations

from codeintel.core.config.accessor import ConfigAccessor
from codeintel.core.config.registry import (
    ConfigNotFoundError,
    ConfigRegistry,
    ConfigTypeError,
    ConfigValidationError,
)
from codeintel.core.process import (
    CommandExecutionError,
    CommandExecutor,
    CommandNotAllowedError,
    CommandResult,
)
from codeintel.core.singleton import SingletonHolder

__all__ = [
    "CommandExecutionError",
    "CommandExecutor",
    "CommandNotAllowedError",
    "CommandResult",
    "ConfigAccessor",
    "ConfigNotFoundError",
    "ConfigRegistry",
    "ConfigTypeError",
    "ConfigValidationError",
    "SingletonHolder",
]
