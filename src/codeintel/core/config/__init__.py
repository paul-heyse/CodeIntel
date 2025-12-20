"""Configuration infrastructure for CodeIntel.

This package provides the configuration accessor protocol and registry:

- **ConfigAccessor**: Protocol for typed configuration access
- **ConfigRegistry**: Full-featured registry with runtime validation
- **ConfigNotFoundError**: Raised when requested config is not registered
- **ConfigTypeError**: Raised when config type doesn't match
- **ConfigValidationError**: Raised when config validation fails

Example
-------
>>> from codeintel.core.config import ConfigRegistry
>>> from dataclasses import dataclass
>>> @dataclass
... class AppConfig:
...     debug: bool
>>> registry = ConfigRegistry()
>>> registry.register(AppConfig, AppConfig(debug=True))
>>> config = registry.get(AppConfig)
>>> config.debug
True
"""

from __future__ import annotations

from codeintel.core.config.accessor import ConfigAccessor
from codeintel.core.config.registry import (
    ConfigNotFoundError,
    ConfigRegistry,
    ConfigTypeError,
    ConfigValidationError,
)
from codeintel.core.config.settings import (
    BuildSettings,
    ExportAuditSettings,
    HamiltonExecutionSettings,
    ServingSettings,
)

__all__ = [
    "BuildSettings",
    "ConfigAccessor",
    "ConfigNotFoundError",
    "ConfigRegistry",
    "ConfigTypeError",
    "ConfigValidationError",
    "ExportAuditSettings",
    "HamiltonExecutionSettings",
    "ServingSettings",
]
